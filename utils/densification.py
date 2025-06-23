import imageio
import cv2
from utils import safe_math
from typing import NamedTuple, List
import gc
from delaunay_rasterization.internal.render_err import render_err
import torch
from utils.train_util import *
from utils import topo_utils


# -----------------------------------------------------------------------------
# 1.  Aggregation helper
# -----------------------------------------------------------------------------
class RenderStats(NamedTuple):
    total_within_var_votes: torch.Tensor    # (T, 2)
    total_within_var: torch.Tensor    # (T, 2)
    within_var_rays: torch.Tensor         # (T, 2, 6)
    total_var_moments: torch.Tensor     # (T, 3)
    between_var_moments: torch.Tensor   # (T, 3)
    tet_moments: torch.Tensor           # (T, 4)
    tet_view_count: torch.Tensor             # (T,)
    total_var_count: torch.Tensor         # (T,)
    tet_size: torch.Tensor              # (T,)
    peak_contrib: torch.Tensor              # (T,)
    total_T: torch.Tensor
    total_err: torch.Tensor
    total_ssim: torch.Tensor
    max_ssim: torch.Tensor


@torch.no_grad()
def collect_render_stats(
    sampled_cameras: List["Camera"],
    model,
    args,
    device: torch.device,
):
    """Accumulate densification statistics for one iteration."""
    n_tets = model.indices.shape[0]

    # Pre-allocate accumulators ------------------------------------------------
    tet_moments = torch.zeros((n_tets, 4), device=device)
    tet_view_count = torch.zeros((n_tets,), device=device)
    tet_size = torch.zeros_like(tet_view_count)
    total_var_count = torch.zeros((n_tets,), device=device)

    total_within_var = torch.zeros((n_tets), device=device)
    max_ssim = torch.zeros((n_tets), device=device)
    total_ssim = torch.zeros((n_tets), device=device)
    total_T = torch.zeros((n_tets), device=device)
    total_err = torch.zeros((n_tets), device=device)
    peak_contrib = torch.zeros((n_tets), device=device)
    total_within_var_votes = torch.zeros((n_tets, 2), device=device)
    within_var_rays = torch.zeros((n_tets, 2, 6), device=device)
    total_var_moments = torch.zeros((n_tets, 3), device=device)
    between_var_moments = torch.zeros((n_tets, 3), device=device)


    # Main per-camera loop -----------------------------------------------------
    for cam in sampled_cameras:
        target = cam.original_image.cuda()

        image_votes, extras = render_err(
            target, cam, model,
            scene_scaling=model.scene_scaling,
            tile_size=args.tile_size,
            lambda_ssim=args.clone_lambda_ssim
        )

        tc = extras["tet_count"]
        
        # --- Create a single mask for valid updates ---
        # Mask for tets that have a reasonable number of samples in the current view
        update_mask = (tc >= args.min_tet_count) & (tc < 8000)

        # --- Moments (s0: sum of T, s1: sum of err, s2: sum of err^2)
        image_T, image_err, image_err2 = image_votes[:, 0], image_votes[:, 1], image_votes[:, 2]
        # total_T_p, image_err, image_err2 = image_votes[:, 3], image_votes[:, 4], image_votes[:, 5]
        _, image_Terr, image_ssim = image_votes[:, 3], image_votes[:, 4], image_votes[:, 5]
        N = tc
        peak_contrib = torch.maximum(image_T, peak_contrib)
        total_T += image_T
        total_err += image_Terr
        total_ssim += image_ssim
        max_ssim = torch.maximum(image_ssim, max_ssim)

        # -------- Within-Image Variance (Top-2 per tet) -----------------------
        within_var_mu = safe_math.safe_div(image_err, N)
        within_var_std = (safe_math.safe_div(image_err2, N) - within_var_mu**2).clip(min=0)
        within_var_std[N < 10] = 0
        within_var_std[~update_mask] = 0 # Use the unified mask

        within_var_votes = image_T * within_var_std

        # ray buffer: (enter | exit) → (N, 6)
        w = image_votes[:, 12:13]
        seg_exit = safe_math.safe_div(image_votes[:, 9:12], w)
        seg_enter = safe_math.safe_div(image_votes[:, 6:9], w)
        rays = torch.cat([seg_enter, seg_exit], dim=1)

        # keep top-2 candidates per tet across all views
        total_within_var += within_var_votes
        votes_3 = torch.cat([total_within_var_votes, within_var_votes[:, None]], dim=1)
        rays_3 = torch.cat([within_var_rays, rays[:, None]], dim=1)
        votes_sorted, idx_sorted = votes_3.sort(1, descending=True)

        total_within_var_votes = votes_sorted[:, :2]
        within_var_rays = torch.gather(
            rays_3, 1,
            idx_sorted[:, :2, None].expand(-1, -1, 6)
        )

        # -------- Total Variance (accumulated across images) ------------------
        total_var_moments[update_mask, 0] += N[update_mask]
        # total_var_moments[update_mask, 0] += image_T[update_mask]
        total_var_moments[update_mask, 1] += image_err[update_mask]
        total_var_moments[update_mask, 2] += image_err2[update_mask]
        total_var_count[update_mask] += N[update_mask]

        # -------- Between-Image Variance (accumulated across images) ----------
        # We compute the variance of the mean error across different views
        mean_err_per_view = within_var_mu
        mean_err_per_view[N < 10] = 0

        between_var_moments[update_mask, 0] += image_T[update_mask] # Use summed image_T as weight
        between_var_moments[update_mask, 1] += mean_err_per_view[update_mask]
        between_var_moments[update_mask, 2] += (mean_err_per_view[update_mask])**2

        # -------- Other stats -------------------------------------------------
        tet_moments[update_mask, :3] += image_votes[update_mask, 13:16]
        tet_moments[update_mask, 3] += w[update_mask].reshape(-1)

        tet_view_count[update_mask] += 1 # Count views per tet
        tet_size += tc

    # done
    return RenderStats(
        total_within_var_votes = total_within_var_votes,
        total_within_var = total_within_var,
        within_var_rays = within_var_rays,
        total_var_moments = total_var_moments,
        between_var_moments = between_var_moments,
        tet_moments = tet_moments,
        tet_view_count = tet_view_count,
        total_var_count = total_var_count,
        tet_size = tet_size,
        peak_contrib = peak_contrib,
        total_T = total_T,
        total_err = total_err,
        total_ssim = total_ssim,
        max_ssim = max_ssim,
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
    target_addition
):
    """Turns accumulated statistics into actual vertex cloning / splitting."""
    # ---------- Calculate scores from variances ------------------------------
    total_T, s1_b, s2_b = stats.between_var_moments.T
    # 1. Total Variance Score (for growing)
    s0_t, s1_t, s2_t = stats.total_var_moments.T
    N_t = stats.total_var_count
    total_var_mu = safe_math.safe_div(s1_t, s0_t)
    total_var_std = (safe_math.safe_div(s2_t, s0_t) - total_var_mu**2).clip(min=0)
    total_var_std[s0_t < 1] = 0

    # 2. Between-Image Variance Score (for splitting)
    N_b = stats.tet_view_count # Num views
    between_var_mu = safe_math.safe_div(s1_b, N_b)
    between_var_std = (safe_math.safe_div(s2_b, N_b) - between_var_mu**2).clip(min=0)
    between_var_std[N_b < 2] = 0 # Need at least 2 views for variance

    # 3. Within-Image Variance Score (for splitting)
    within_var = stats.total_within_var_votes[:, 0]
    within_var = stats.total_ssim / stats.tet_size.clip(min=1).sqrt()

    between_var = stats.total_T * between_var_std # Weighted by summed s0
    total_var = stats.total_err * total_var_std
    # total_var = stats.total_T * safe_math.safe_div(total_var_std, between_var_std).clip(min=0, max=10)
    total_var[(N_b < 2) | (s0_t < 1)] = 0
    # within_var = stats.total_within_var / stats.tet_view_count.clip(min=1)
    # within_var = safe_math.safe_div(stats.total_within_var, total_T)
    # total_var += within_var
    vertices = model.vertices
    # circumcenters, _, tet_density, rgb, grd, sh = model.compute_batch_features(vertices, model.indices, 0, model.indices.shape[0])

    # --- Masking and target calculation --------------------------------------
    tet_density = model.calc_tet_density()
    alphas = model.calc_tet_alpha(mode="max", density=tet_density)
    # mask_alive = alphas >= args.clone_min_alpha
    # mask_alive = (alphas >= args.clone_min_alpha) & (stats.peak_contrib >= args.clone_min_density)
    mask_alive = (alphas >= args.clone_min_alpha) & (tet_density.reshape(-1) >= args.clone_min_density)
    total_var[~mask_alive] = 0
    within_var[~mask_alive] = 0
    between_var[~mask_alive] = 0
    # total_var = (total_var - between_var).clip(min=0)

    target_addition = min(target_addition, stats.tet_view_count.shape[0])
    if target_addition < 0:
        return

    # Assume args.percent_total_var and args.percent_within_var exist
    target_total = int(args.percent_total * target_addition)
    target_within = int(args.percent_within * target_addition)
    target_between = int(max(0, target_addition - target_total - target_within))

    grow_mask = torch.zeros_like(total_var, dtype=torch.bool)
    within_mask = torch.zeros_like(grow_mask)
    between_mask = torch.zeros_like(grow_mask)

    # To prevent overlap, we select candidates sequentially
    # and zero out the scores of selected tets in other categories.
    temp_total_score = total_var.clone()
    temp_within_score = within_var.clone()
    temp_between_score = between_var.clone()

    if target_total > 0:
        top_total = torch.topk(temp_total_score, target_total).indices
        grow_mask[top_total] = temp_total_score[top_total] > 0
        temp_within_score[grow_mask] = 0
        temp_between_score[grow_mask] = 0

    if target_within > 0:
        top_within = torch.topk(temp_within_score, target_within).indices
        within_mask[top_within] = temp_within_score[top_within] > 0
        temp_between_score[within_mask] = 0
    
    if target_between > 0:
        top_between = torch.topk(temp_between_score, target_between).indices
        between_mask[top_between] = temp_between_score[top_between] > 0

    split_mask = within_mask | between_mask
    clone_mask = split_mask | grow_mask

    # ---------- debug renders -------------------------------------------------
    if args.output_path is not None:

        f = mask_alive.float().unsqueeze(1).expand(-1, 4).clone()
        color = torch.rand_like(f[:, :3])
        # color = rgb + 0.5#torch.rand_like(f[:, :3])
        f[:, :3] = color
        f[:, 3] *= 2.0    # alpha
        imageio.imwrite(args.output_path / f"alive_mask{iteration}.png",
                        render_debug(f, model, sample_cam, 10))
        f = clone_mask.float().unsqueeze(1).expand(-1, 4).clone()
        f[:, :3] = color
        f[:, 3] *= 2.0    # alpha
        imageio.imwrite(args.output_path / f"densify{iteration}.png",
                        render_debug(f, model, sample_cam, 10))
        imageio.imwrite(args.output_path / f"total_var{iteration}.png",
                        render_debug(total_var[:, None],
                                     model, sample_cam))
        imageio.imwrite(args.output_path / f"within_var{iteration}.png",
                        render_debug(within_var[:, None],
                                     model, sample_cam))
        imageio.imwrite(args.output_path / f"between_var{iteration}.png",
                        render_debug(between_var[:, None],
                                     model, sample_cam))
        imageio.imwrite(args.output_path / f"im{iteration}.png",
                        cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))


    # ---------- pick clone positions -----------------------------------------
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

    tet_optim.split(clone_indices,
                    split_point,
                    **args.as_dict())

    # ---------- velocity-based additions -------------------------------------
    raw_verts = model.contracted_vertices
    vstate = tet_optim.vertex_optim.get_state_by_name("contracted_vertices")
    velocity = vstate["exp_avg"] * args.speed_mul

    if model.contract_vertices:
        J_d = topo_utils.contraction_jacobian_d_in_chunks(
            model.vertices[:raw_verts.shape[0]]).view(-1, 1)
        speed = torch.linalg.norm(velocity * J_d, dim=1)
    else:
        speed = torch.linalg.norm(velocity, dim=1)
        
    if args.clone_velocity > 0:
        new_verts = (raw_verts + velocity)[speed > args.clone_velocity]
        tet_optim.add_points(new_verts, raw_verts=True)
        num_cloned = new_verts.shape[0]
    else:
        num_cloned = 0

    # ---------- housekeeping --------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"#Grow: {grow_mask.sum():4d} #Split: {split_mask.sum():4d} | "
        f"T_Total: {target_total:4d} T_Within: {target_within:4d} T_Between: {target_between:4d} | "
        f"Total Avg: {total_var.mean():.4f} Within Avg: {within_var.mean():.4f} Between Avg: {between_var.mean():.4f}  | "
        f"By Vel: {num_cloned}"
    )
