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
    within_var_rays: torch.Tensor
    total_var_moments: torch.Tensor
    tet_moments: torch.Tensor
    tet_view_count: torch.Tensor
    total_err: torch.Tensor
    top_ssim: torch.Tensor
    top_size: torch.Tensor
    total_count: torch.Tensor


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
            lambda_ssim=args.clone_lambda_ssim,
            glo=glo_list(torch.LongTensor([cam.uid]).to(device))
        )

        tc = extras["tet_count"]
        
        update_mask = (tc >= args.min_tet_count) & (tc < 8000)

        # --- Moments (s0: sum of T, s1: sum of err, s2: sum of err^2)
        image_T, image_err, image_err2 = image_votes[:, 0], image_votes[:, 1], image_votes[:, 2]
        _, image_Terr, image_ssim = image_votes[:, 3], image_votes[:, 4], image_votes[:, 5]
        total_err += image_Terr

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

    tet_density = model.calc_tet_density()
    alphas = model.calc_tet_alpha(mode="max", density=tet_density)
    mask_alive = (alphas >= args.clone_min_alpha) & (tet_density.reshape(-1) >= args.clone_min_density)

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

    tet_optim.split(clone_indices,
                    split_point,
                    **args.as_dict())

    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"#Within: {within_mask.sum():4d} #Total: {total_mask.sum():4d} | "
        f"Total Avg: {total_var.mean():.4f} Within Avg: {within_var.mean():.4f}"
    )
