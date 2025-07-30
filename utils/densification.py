import imageio
import cv2
from utils import safe_math
from typing import NamedTuple, List
import gc
from delaunay_rasterization.internal.render_err import render_err
from delaunay_rasterization import render_debug
import torch
from icecream import ic

def get_approx_ray_intersections(split_rays_data, epsilon=1e-7):
    """
    Calculates the approximate intersection point for pairs of line segments.

    The intersection is defined as the midpoint of the shortest segment
    connecting the two input line segments.

    Args:
        split_rays_data (torch.Tensor): Tensor of shape (N, 2, 6).
            - N: Number of segment pairs.
            - 2: Represents the two segments in a pair.
            - 6: Contains [Ax, Ay, Az, Bx, By, Bz] for each segment,
                 where A and B are the segment endpoints.
                 Based on current Python code:
                 A = average_P_exit, B = average_P_entry
        epsilon (float): Small value to handle parallel lines and avoid
                         division by zero if a segment has zero length.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) representing the approximate
                      "intersection" points (midpoints of closest approach).
    """
    # Segment 1 endpoints
    p1_a = split_rays_data[:, 0, 0:3]  # Endpoint A of first segments (N, 3)
    p1_b = split_rays_data[:, 0, 3:6]  # Endpoint B of first segments (N, 3)
    # Segment 2 endpoints
    p2_a = split_rays_data[:, 1, 0:3]  # Endpoint A of second segments (N, 3)
    p2_b = split_rays_data[:, 1, 3:6]  # Endpoint B of second segments (N, 3)

    # Define segment origins and direction vectors
    # Segment S1: o1 + s * d1, for s in [0, 1]
    # Segment S2: o2 + t * d2, for t in [0, 1]
    o1 = p1_a
    d1 = p1_b - p1_a  # Direction vector for segment 1 (from A to B)
    o2 = p2_a
    d2 = p2_b - p2_a  # Direction vector for segment 2 (from A to B)

    # Calculate terms for finding closest points on the infinite lines
    # containing the segments (based on standard formulas, e.g., Christer Ericson's "Real-Time Collision Detection")
    v_o = o1 - o2 # Vector from origin of line 2 to origin of line 1

    a = torch.sum(d1 * d1, dim=1)  # Squared length of d1
    b = torch.sum(d1 * d2, dim=1)  # Dot product of d1 and d2
    c = torch.sum(d2 * d2, dim=1)  # Squared length of d2
    d = torch.sum(d1 * v_o, dim=1) # d1 dot (o1 - o2)
    e = torch.sum(d2 * v_o, dim=1) # d2 dot (o1 - o2)

    denom = a * c - b * b
    s_line_num = (b * e) - (c * d)
    t_line_num = (a * e) - (b * d) # This corresponds to t_c = (a*e - b*d)/denom from previous thoughts for P(t) = O2 + tD2

    # Handle near-zero denominator (lines are parallel or one segment is a point)
    # We compute with a safe denominator, then clamp. Clamping is key for segments.
    denom_safe = torch.where(denom.abs() < epsilon, torch.ones_like(denom), denom)
    
    s_line = s_line_num / denom_safe
    t_line = t_line_num / denom_safe # Note: This t_line is for the parameter of d2 (from o2)

    # Clamp parameters to [0, 1] to stay within the segments
    bad_intersect = (s_line < 0) | (t_line < 0) | (s_line > 1) | (t_line > 1)
    s_seg = torch.clamp(s_line, 0.0, 1.0)
    t_seg = torch.clamp(t_line, 0.0, 1.0)

    # Points of closest approach on the segments
    pc1 = o1 + s_seg.unsqueeze(1) * d1
    pc2 = o2 + t_seg.unsqueeze(1) * d2
    
    p_int = (pc1 + pc2) / 2.0
                        
    return p_int, bad_intersect

# -----------------------------------------------------------------------------
# 1.  Aggregation helper
# -----------------------------------------------------------------------------
class RenderStats(NamedTuple):
    within_var_rays: torch.Tensor
    total_var_moments: torch.Tensor
    tet_moments: torch.Tensor
    tet_view_count: torch.Tensor
    total_err: torch.Tensor
    top_ssim: torch.Tensor
    top_size: torch.Tensor
    total_count: torch.Tensor
    peak_contrib: torch.Tensor
    alphas: torch.Tensor
    density: torch.Tensor


@torch.no_grad()
def collect_render_stats(
    sampled_cameras: List["Camera"],
    model,
    glo_list,
    args,
    device: torch.device,
):
    """Accumulate densification statistics for one iteration."""
    n_tets = model.indices.shape[0]
    tet_moments = torch.zeros((n_tets, 4), device=device)
    tet_view_count = torch.zeros((n_tets,), device=device)

    peak_contrib = torch.zeros((n_tets), device=device)
    top_ssim = torch.zeros((n_tets, 2), device=device)
    top_size = torch.zeros((n_tets, 2), device=device)
    total_err = torch.zeros((n_tets), device=device)
    total_count = torch.zeros((n_tets), device=device, dtype=int)
    within_var_rays = torch.zeros((n_tets, 2, 6), device=device)
    total_var_moments = torch.zeros((n_tets, 3), device=device)
    top_moments = torch.zeros((n_tets, 2, 4), device=device)

    for cam in sampled_cameras:
        target = cam.original_image.cuda()

        image_votes, extras = render_err(
            target, cam, model,
            scene_scaling=model.scene_scaling,
            tile_size=args.tile_size,
            lambda_ssim=0,
            glo=glo_list(torch.LongTensor([cam.uid]).to(device))
        )

        tc = extras["tet_count"]
        
        update_mask = (tc >= args.min_tet_count) & (tc < 8000)

        # --- Moments (s0: sum of T, s1: sum of err, s2: sum of err^2)
        image_T, image_err, image_err2 = image_votes[:, 0], image_votes[:, 1], image_votes[:, 2]
        _, image_Terr, image_ssim = image_votes[:, 3], image_votes[:, 4], image_votes[:, 5]
        total_err += image_Terr
        peak_contrib = torch.maximum(image_T / tc.clip(min=1), peak_contrib)

        # ray buffer: (enter | exit) → (N, 6)
        w = image_votes[:, 12:13]
        seg_exit = safe_math.safe_div(image_votes[:, 9:12], w)
        seg_enter = safe_math.safe_div(image_votes[:, 6:9], w)

        image_ssim[~update_mask] = 0
        top_ssim, idx_sorted = torch.cat([top_ssim[:, :2], image_ssim.reshape(-1, 1)], dim=1).sort(1, descending=True)
        top_size = torch.gather(
            torch.cat([top_size, tc.reshape(-1, 1)], dim=1), 1,
            idx_sorted[:, :2]
        )

        # -------- Other stats -------------------------------------------------
        tet_moments[update_mask, :3] += image_votes[update_mask, 13:16]
        tet_moments[update_mask, 3] += w[update_mask].reshape(-1)
        moments = torch.cat([
            image_votes[:, 13:16],
            w.reshape(-1, 1)
        ], dim=1)
        moments_3 = torch.cat([top_moments, moments.reshape(-1, 1, 4)], dim=1)
        top_moments = torch.gather(
            moments_3, 1,
            idx_sorted[:, :2, None].expand(-1, -1, 4)
        )

        rays = torch.cat([seg_enter, seg_exit], dim=1)
        rays_3 = torch.cat([within_var_rays, rays[:, None]], dim=1)
        within_var_rays = torch.gather(
            rays_3, 1,
            idx_sorted[:, :2, None].expand(-1, -1, 6)
        )

        # -------- Total Variance (accumulated across images) ------------------
        total_var_moments[update_mask, 0] += image_T[update_mask]
        total_var_moments[update_mask, 1] += image_err[update_mask]
        total_var_moments[update_mask, 2] += image_err2[update_mask]
        # total_count += N
        total_count[update_mask] += 1

        # -------- Other stats -------------------------------------------------
        tet_moments[update_mask, :3] += image_votes[update_mask, 13:16]
        tet_moments[update_mask, 3] += w[update_mask].reshape(-1)

        tet_view_count[update_mask] += 1 # Count views per tet

    tet_density = model.calc_tet_density()
    alphas = model.calc_tet_alpha(mode="max", density=tet_density)
    # done
    return RenderStats(
        within_var_rays = within_var_rays,
        total_var_moments = total_var_moments,
        tet_moments = tet_moments,
        tet_view_count = tet_view_count,
        total_err = total_err,
        total_count = total_count,
        top_ssim = top_ssim[:, :2],
        top_size = top_size[:, :2],
        peak_contrib = peak_contrib,
        density=tet_density,
        alphas=alphas
    )

@torch.no_grad()
def apply_densification(
    stats: RenderStats,
    model,
    tet_optim,
    args,
    iteration: int,
    device: torch.device,
    sample_cam,
    sample_image,
    budget: int
):
    s0_t, s1_t, s2_t = stats.total_var_moments.T
    total_var_mu = safe_math.safe_div(s1_t, s0_t)
    total_var_std = (safe_math.safe_div(s2_t, s0_t) - total_var_mu**2).clip(min=0)
    total_var_std[s0_t < 1] = 0

    within_var = stats.top_ssim.sum(dim=1) / stats.top_size.sum(dim=1).clip(min=1).sqrt()

    total_var = (stats.total_err / stats.total_count.clip(min=1)) * total_var_std
    # N_b = stats.tet_view_count # Num views
    # total_var[(N_b < 2) | (s0_t < 1)] = 0

    mask_alive = (stats.alphas >= args.clone_min_alpha) & (stats.density.reshape(-1) >= args.clone_min_density)


    mask_alive2 = ((stats.peak_contrib > args.contrib_threshold) | (stats.alphas > args.clone_min_alpha)).int()
    keep_verts = torch.zeros((model.vertices.shape[0]), dtype=torch.int, device=stats.alphas.device)
    indices = model.indices.long()
    reduce_type = "sum"
    keep_verts.scatter_reduce_(dim=0, index=indices[..., 0], src=mask_alive2, reduce=reduce_type)
    keep_verts.scatter_reduce_(dim=0, index=indices[..., 1], src=mask_alive2, reduce=reduce_type)
    keep_verts.scatter_reduce_(dim=0, index=indices[..., 2], src=mask_alive2, reduce=reduce_type)
    keep_verts.scatter_reduce_(dim=0, index=indices[..., 3], src=mask_alive2, reduce=reduce_type)


    total_var[~mask_alive] = 0
    within_var[~mask_alive] = 0
    within_mask = (within_var > args.within_thresh)
    total_mask = (total_var > args.total_thresh)
    clone_mask = within_mask | total_mask
    if clone_mask.sum() > budget:
        true_indices = clone_mask.nonzero().squeeze(-1)
        perm = torch.randperm(true_indices.size(0))
        selected_indices = true_indices[perm[:budget]]
        
        clone_mask = torch.zeros_like(clone_mask, dtype=torch.bool)
        clone_mask[selected_indices] = True

    if args.output_path is not None:

        f = mask_alive.float().unsqueeze(1).expand(-1, 4).clone()
        color = torch.rand_like(f[:, :3])
        f[:, :3] = color
        f[:, 3] *= 2.0    # alpha
        # imageio.imwrite(args.output_path / f"alive_mask{iteration}.png",
        #                 render_debug(f, model, sample_cam, 10))
        f = clone_mask.float().unsqueeze(1).expand(-1, 4).clone()
        f[:, :3] = color
        f[:, 3] *= 2.0    # alpha
        # imageio.imwrite(args.output_path / f"densify{iteration}.png",
        #                 render_debug(f, model, sample_cam, 10))
        # imageio.imwrite(args.output_path / f"total_var{iteration}.png",
        #                 render_debug(total_var[:, None],
        #                              model, sample_cam))
        # imageio.imwrite(args.output_path / f"within_var{iteration}.png",
        #                 render_debug(within_var[:, None],
        #                              model, sample_cam))
        imageio.imwrite(args.output_path / f"im{iteration}.png",
                        cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))

    clone_indices = model.indices[clone_mask]
    split_point, bad = get_approx_ray_intersections(stats.within_var_rays)
    grow_point = safe_math.safe_div(
        stats.tet_moments[:, :3],
        stats.tet_moments[:, 3:4]
    )
    split_point[bad] = grow_point[bad]
    split_point = split_point[clone_mask]
    bad = bad[clone_mask]
    barycentric = torch.rand((clone_indices.shape[0], clone_indices.shape[1], 1), device=device).clip(min=0.01, max=0.99)
    barycentric_weights = barycentric / (1e-3+barycentric.sum(dim=1, keepdim=True))
    random_locations = (model.vertices[clone_indices] * barycentric_weights).sum(dim=1)
    split_point[bad] = random_locations[bad]    # fall back

    keep_verts = keep_verts > 0
    print(f"Pruned: {(~keep_verts).sum()}")
    # tet_optim.remove_points(keep_verts.reshape(-1))
    tet_optim.split(clone_indices,
                    split_point,
                    **args.as_dict())

    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"#Within: {within_mask.sum():4d} #Total: {total_mask.sum():4d} | "
        f"Total Avg: {total_var.mean():.4f} Within Avg: {within_var.mean():.4f}"
    )
