import imageio
import cv2
from utils import safe_math
from typing import NamedTuple, List
import gc
from delaunay_rasterization.internal.render_err import render_err
import torch
from utils.train_util import *


# -----------------------------------------------------------------------------  
# 1.  Aggregation helper  
# -----------------------------------------------------------------------------  
class RenderStats(NamedTuple):
    total_split_votes : torch.Tensor   # (T, 2)
    split_rays        : torch.Tensor   # (T, 2, 6)
    total_grow_moments: torch.Tensor   # (T, 3)
    tet_moments       : torch.Tensor   # (T, 4)
    tet_count         : torch.Tensor   # (T,)
    tet_size          : torch.Tensor   # (T,)


@torch.no_grad()
def collect_render_stats(
    sampled_cameras: List["Camera"],
    model,
    args,
    device: torch.device,
):
    """Accumulate split / grow statistics for one densification iteration."""
    n_tets = model.indices.shape[0]

    # Pre-allocate accumulators ------------------------------------------------
    tet_moments        = torch.zeros((n_tets, 4),     device=device)
    tet_votes          = torch.zeros((n_tets, 4),     device=device)  # <-- kept for parity
    tet_count          = torch.zeros((n_tets,),       device=device)
    tet_size           = torch.zeros_like(tet_count)

    total_split_votes  = torch.zeros((n_tets, 2),     device=device)
    total_grow_moments = torch.zeros((n_tets, 3),     device=device)
    split_rays         = torch.zeros((n_tets, 2, 6),  device=device)

    # Main per-camera loop -----------------------------------------------------
    for cam in sampled_cameras:
        target = cam.original_image.cuda()

        image_votes, extras = render_err(
            target, cam, model,
            scene_scaling=model.scene_scaling,
            tile_size=args.tile_size,
            density_t=args.density_t,
            lambda_ssim=args.clone_lambda_ssim
        )

        tc      = extras["tet_count"]
        density = extras["cell_values"][:, 0]          # kept for completeness

        # -------- Split -------------------------------------------------------
        split_mask = (tc > 2000) | (tc < 4)

        s0, s1, s2  = image_votes[:, 0], image_votes[:, 1], image_votes[:, 2]
        split_mu    = safe_math.safe_div(s1, s0)
        split_std   = safe_math.safe_div(s2, s0) - split_mu ** 2
        split_std[s0 < 1]   = 0
        split_std[split_mask] = 0

        split_votes = s0 * split_std

        # ray buffer: (enter | exit) → (N, 6)
        w          = image_votes[:, 12:13]
        seg_exit   = safe_math.safe_div(image_votes[:, 9:12],  w)
        seg_enter  = safe_math.safe_div(image_votes[:, 6:9],   w)
        rays       = torch.cat([seg_enter, seg_exit], dim=1)

        # keep top-2 split candidates per tet
        votes_3        = torch.cat([total_split_votes, split_votes[:, None]], dim=1)
        rays_3         = torch.cat([split_rays,        rays[:, None]],       dim=1)
        votes_sorted, idx_sorted = votes_3.sort(1, descending=True)

        total_split_votes = votes_sorted[:, :2]
        split_rays        = torch.gather(
            rays_3, 1,
            idx_sorted[:, :2, None].expand(-1, -1, 6)
        )

        # -------- Grow --------------------------------------------------------
        grow_mask = (tc < 2000) & (tc > 4)
        total_grow_moments[grow_mask] += image_votes[grow_mask, 3:6]

        tet_moments[grow_mask, :3] += image_votes[grow_mask, 13:16]
        tet_moments[grow_mask, 3]  += image_votes[grow_mask, 3]

        tet_count += (tc > 0)
        tet_size  += tc

    # done
    return RenderStats(
        total_split_votes  = total_split_votes,
        split_rays         = split_rays,
        total_grow_moments = total_grow_moments,
        tet_moments        = tet_moments,
        tet_count          = tet_count,
        tet_size           = tet_size,
    )


# -----------------------------------------------------------------------------  
# 2.  Densification / cloning step  
# -----------------------------------------------------------------------------  
@torch.no_grad()
def apply_densification(
    stats       : RenderStats,
    model,
    tet_optim,
    args,
    iteration   : int,
    device      : torch.device,
    sample_cam,                     # used for render_debug
    sample_image,
    target_addition
):
    """Turns accumulated statistics into actual vertex cloning / splitting."""
    # ---------- split & grow scores ------------------------------------------
    split_score = stats.total_split_votes.sum(1)

    s0, s1, s2  = stats.total_grow_moments.t()
    grow_mu     = safe_math.safe_div(s1, s0)
    grow_std    = safe_math.safe_div(s2, s0) - grow_mu**2
    grow_std[s0 < 1] = 0
    grow_score  = s0 * grow_std

    alphas = compute_alpha(model.indices,
                           model.vertices,
                           model.calc_tet_density()).view(-1)
    mask_alive = alphas >= args.clone_min_alpha
    grow_score [~mask_alive] = 0
    split_score[~mask_alive] = 0

    # ---------- quota for this iteration -------------------------------------
    if target_addition <= 0:   # nothing to do
        return

    target_split = int(args.percent_split * target_addition)
    target_grow  = target_addition - target_split

    split_mask = torch.zeros_like(split_score, dtype=torch.bool)
    grow_mask  = torch.zeros_like(split_mask)

    if target_grow > 0:
        top_grow = torch.topk(grow_score, target_grow).indices
        grow_mask[top_grow] = grow_score[top_grow] > 0
        split_score[grow_mask] = 0      # exclude from split

    if target_split > 0:
        top_split = torch.topk(split_score, target_split).indices
        split_mask[top_split] = split_score[top_split] > 0

    clone_mask = split_mask | grow_mask
    clone_indices = model.indices[clone_mask]

    # ---------- debug renders -------------------------------------------------
    if args.output_path is not None:
        f = clone_mask.float().unsqueeze(1).expand(-1, 4).clone()
        f[:, :3] = torch.rand_like(f[:, :3])
        f[:, 3] *= 1.0     # alpha

        imageio.imwrite(args.output_path / f"densify{iteration}.png",
                        render_debug(f, model, sample_cam, 10))

        imageio.imwrite(args.output_path / f"split{iteration}.png",
                        render_debug(split_score[:, None],
                                     model, sample_cam))
        imageio.imwrite(args.output_path / f"grow{iteration}.png",
                        render_debug(grow_score[:, None],
                                     model, sample_cam))
        imageio.imwrite(args.output_path / f"im{iteration}.png",
                        cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))

    # ---------- pick clone positions -----------------------------------------
    split_point, bad = get_approx_ray_intersections(stats.split_rays)
    grow_point       = safe_math.safe_div(
        stats.tet_moments[:, :3],
        stats.tet_moments[:, 3:4]
    )
    split_point[bad]      = grow_point[bad]   # fall back
    split_point[grow_mask] = grow_point[grow_mask]

    tet_optim.split(clone_indices,
                    split_point[clone_mask],
                    args.split_mode)

    # ---------- velocity-based additions -------------------------------------
    raw_verts    = model.contracted_vertices
    vstate       = tet_optim.vertex_optim.get_state_by_name("contracted_vertices")
    velocity     = vstate["exp_avg"] * args.speed_mul

    J_d   = topo_utils.contraction_jacobian_d_in_chunks(
                model.vertices[:raw_verts.shape[0]]).view(-1, 1)
    speed = torch.linalg.norm(velocity * J_d, dim=1)
    new_verts = (raw_verts + velocity)[speed > args.clone_velocity]

    tet_optim.add_points(new_verts, raw_verts=True)

    # ---------- housekeeping --------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"#Split: {split_mask.sum():4d}  "
        f"#Grow: {grow_mask.sum():4d}  "
        f"#T_Split: {target_split:4d}  "
        f"#T_Grow: {target_grow:4d}  "
        f"Grow Avg: {grow_score[grow_mask].mean():.4f}  "
        f"Split Avg: {split_score[split_mask].mean():.4f}  "
        f"By Vel: {new_verts.shape[0]}  "
        f"Added: {target_addition}"
    )
