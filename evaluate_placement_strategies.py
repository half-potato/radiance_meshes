#!/usr/bin/env python3
"""
Evaluate 4 vertex placement strategies for densification.

Strategies:
  A. current       - Ray intersection + random barycentric for bad (baseline)
  B. clamped       - Use clamped intersection for ALL cases (no fallback)
  C. wt_centroid   - Ray intersection + weighted centroid fallback for bad
  D. angular_div   - Select top-2 rays by angular diversity (from top-5 SSIM),
                     clamped fallback for bad

For each strategy: add all candidates -> retrain backbone -> measure PSNR.

Usage:
  python evaluate_placement_strategies.py --ckpt output/bicycle_base
  python evaluate_placement_strategies.py --ckpt output/bicycle_base --retrain-iters 500 --topk 5
"""

import os
import sys
import gc
import json
import math
import random
import argparse
import time
from pathlib import Path
from typing import List

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import loader
from models.ingp_color import Model, TetOptimizer
from utils.args import Args
from utils.train_util import render, SimpleSampler
from utils.densification import (
    collect_render_stats, get_approx_ray_intersections, RenderStats
)
from utils import safe_math
from delaunay_rasterization.internal.render_err import render_err
from fused_ssim import fused_ssim

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_model_and_config(ckpt_path, device):
    config = Args.load_from_json(str(Path(ckpt_path) / "config.json"))
    model = Model.load_ckpt(Path(ckpt_path), device)
    return model, config


@torch.no_grad()
def compute_mean_psnr(model, cameras, tile_size=4, min_t=0.4, max_cameras=30):
    model.eval()
    if max_cameras > 0 and max_cameras < len(cameras):
        indices = list(range(len(cameras)))
        random.shuffle(indices)
        cameras = [cameras[i] for i in indices[:max_cameras]]

    psnrs = []
    for cam in cameras:
        target = cam.original_image.cuda()
        gt_mask = cam.gt_alpha_mask.cuda()
        render_pkg = render(cam, model, tile_size=tile_size, min_t=min_t)
        image = render_pkg['render']
        l2 = ((target - image) ** 2 * gt_mask).mean().item()
        psnr = -20 * math.log10(math.sqrt(max(l2, 1e-10)))
        psnrs.append(psnr)
        del render_pkg, image, target, gt_mask
        torch.cuda.empty_cache()
    return np.mean(psnrs)


# ---------------------------------------------------------------------------
# Modified stats collection with top-K rays
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_render_stats_topk(sampled_cameras, model, args, device, K=5):
    """Like collect_render_stats but keeps top-K rays per tet.

    Returns:
        stats: RenderStats with top-2 (for candidate selection)
        rays_topk: (T, K, 6) full top-K rays for angular diversity selection
    """
    n_tets = model.indices.shape[0]

    tet_moments = torch.zeros((n_tets, 4), device=device)
    tet_view_count = torch.zeros((n_tets,), device=device)
    top_ssim_k = torch.zeros((n_tets, K), device=device)
    peak_contrib = torch.zeros((n_tets,), device=device)
    within_var_rays_k = torch.zeros((n_tets, K, 6), device=device)
    total_var_moments = torch.zeros((n_tets, 3), device=device)

    for cam in tqdm(sampled_cameras, desc="Collecting top-K stats"):
        target = cam.original_image.cuda()
        gt_mask = cam.gt_alpha_mask.cuda()
        image_votes, extras = render_err(
            target, gt_mask, cam, model, tile_size=args.tile_size)

        tc = extras["tet_count"][..., 0]
        max_T = extras["tet_count"][..., 1].float() / 65535
        peak_contrib = torch.maximum(max_T, peak_contrib)

        update_mask = (tc >= args.min_tet_count) & (tc < 8000)

        image_T, image_err, image_err2 = (
            image_votes[:, 0], image_votes[:, 1], image_votes[:, 2])
        _, _, image_ssim = image_votes[:, 3], image_votes[:, 4], image_votes[:, 5]
        N = tc
        image_ssim[~update_mask] = 0

        within_var_mu = safe_math.safe_div(image_err, N)
        within_var_std = (
            safe_math.safe_div(image_err2, N) - within_var_mu ** 2
        ).clip(min=0)
        within_var_std[N < 10] = 0
        within_var_std[~update_mask] = 0

        w = image_votes[:, 12:13]
        seg_exit = safe_math.safe_div(image_votes[:, 9:12], w)
        seg_enter = safe_math.safe_div(image_votes[:, 6:9], w)

        image_ssim = image_ssim / tc.clip(min=1)

        # Keep top-K by SSIM
        combined = torch.cat(
            [top_ssim_k, image_ssim.reshape(-1, 1)], dim=1)  # (T, K+1)
        top_ssim_k, idx_sorted = combined.sort(1, descending=True)
        top_ssim_k = top_ssim_k[:, :K]
        idx_sorted_k = idx_sorted[:, :K]

        # Gather rays using same sort indices
        rays = torch.cat([seg_enter, seg_exit], dim=1)  # (T, 6)
        rays_kp1 = torch.cat(
            [within_var_rays_k, rays[:, None]], dim=1)  # (T, K+1, 6)
        within_var_rays_k = torch.gather(
            rays_kp1, 1,
            idx_sorted_k[:, :, None].expand(-1, -1, 6))

        # Accumulate standard stats (matches original double-accumulation)
        tet_moments[update_mask, :3] += image_votes[update_mask, 13:16]
        tet_moments[update_mask, 3] += w[update_mask].reshape(-1)

        total_var_moments[update_mask, 0] += image_T[update_mask]
        total_var_moments[update_mask, 1] += image_err[update_mask]
        total_var_moments[update_mask, 2] += image_err2[update_mask]

        tet_moments[update_mask, :3] += image_votes[update_mask, 13:16]
        tet_moments[update_mask, 3] += w[update_mask].reshape(-1)

        tet_view_count[update_mask] += 1

    # Build standard RenderStats with top-2 for candidate selection
    stats = RenderStats(
        within_var_rays=within_var_rays_k[:, :2],
        total_var_moments=total_var_moments,
        tet_moments=tet_moments,
        tet_view_count=tet_view_count,
        top_ssim=top_ssim_k[:, :2],
        top_size=torch.zeros((n_tets, 2), device=device),
        peak_contrib=peak_contrib,
    )

    return stats, within_var_rays_k


# ---------------------------------------------------------------------------
# Candidate selection (same for all strategies)
# ---------------------------------------------------------------------------

def compute_candidates(stats, model, args, device):
    """Compute candidate mask and scores from stats."""
    s0_t, s1_t, s2_t = stats.total_var_moments.T
    total_var_mu = safe_math.safe_div(s1_t, s0_t)
    total_var_std = (
        safe_math.safe_div(s2_t, s0_t) - total_var_mu ** 2
    ).clip(min=0)
    total_var_std[s0_t < 1] = 0

    N_b = stats.tet_view_count
    within_var = stats.top_ssim.sum(dim=1)
    total_var = s0_t * total_var_std
    total_var[(N_b < 2) | (s0_t < 1)] = 0

    total_var[stats.peak_contrib < args.clone_min_contrib] = 0
    within_var[stats.peak_contrib < args.split_min_contrib] = 0

    within_mask = within_var > args.within_thresh
    total_mask = total_var > args.total_thresh
    clone_mask = within_mask | total_mask

    return clone_mask, within_var, total_var


# ---------------------------------------------------------------------------
# Angular diversity selection from top-K rays
# ---------------------------------------------------------------------------

def select_diverse_pair(rays_topk, epsilon=1e-7):
    """Select the 2 most angularly diverse rays from top-K per tet.

    Args:
        rays_topk: (N, K, 6) - K ray segments [enter_xyz, exit_xyz]
    Returns:
        diverse_rays: (N, 2, 6)
        best_cos: (N,) - |cos(angle)| of selected pair (lower = more diverse)
    """
    N, K, _ = rays_topk.shape
    device = rays_topk.device

    # Compute directions for all K rays
    enters = rays_topk[:, :, :3]
    exits = rays_topk[:, :, 3:]
    dirs = exits - enters  # (N, K, 3)
    lengths = torch.norm(dirs, dim=2, keepdim=True)  # (N, K, 1)
    dirs_norm = dirs / (lengths + epsilon)

    # Find pair with minimum |cos(angle)| (most diverse)
    best_i = torch.zeros(N, dtype=torch.long, device=device)
    best_j = torch.ones(N, dtype=torch.long, device=device)
    best_cos = torch.ones(N, device=device)

    for i in range(K):
        for j in range(i + 1, K):
            # Both rays must have nonzero length
            valid = (lengths[:, i, 0] > epsilon) & (lengths[:, j, 0] > epsilon)
            cos_ij = (dirs_norm[:, i] * dirs_norm[:, j]).sum(dim=1).abs()
            cos_ij[~valid] = 1.0  # Treat invalid as parallel

            better = cos_ij < best_cos
            best_cos[better] = cos_ij[better]
            best_i[better] = i
            best_j[better] = j

    # Gather selected pairs
    arange = torch.arange(N, device=device)
    result = torch.zeros(N, 2, 6, device=device)
    result[:, 0] = rays_topk[arange, best_i]
    result[:, 1] = rays_topk[arange, best_j]

    return result, best_cos


# ---------------------------------------------------------------------------
# Placement strategies
# ---------------------------------------------------------------------------

def compute_split_points_current(stats, clone_mask, model, device):
    """Strategy A: Current (ray intersection + random barycentric fallback)."""
    split_point_all, bad_all = get_approx_ray_intersections(stats.within_var_rays)
    split_points = split_point_all[clone_mask]
    bad = bad_all[clone_mask]

    clone_indices = model.indices[clone_mask]
    barycentric = torch.rand(
        (clone_indices.shape[0], clone_indices.shape[1], 1), device=device
    ).clip(min=0.01, max=0.99)
    bary_w = barycentric / (1e-3 + barycentric.sum(dim=1, keepdim=True))
    random_locations = (model.vertices[clone_indices] * bary_w).sum(dim=1)
    split_points[bad] = random_locations[bad]

    return split_points, bad


def compute_split_points_clamped(stats, clone_mask, model, device):
    """Strategy B: Use clamped intersection for ALL cases (no fallback)."""
    split_point_all, bad_all = get_approx_ray_intersections(stats.within_var_rays)
    split_points = split_point_all[clone_mask]
    bad = bad_all[clone_mask]
    # No fallback - the clamped point is already computed by get_approx_ray_intersections
    return split_points, bad


def compute_split_points_weighted_centroid(stats, clone_mask, model, device):
    """Strategy C: Ray intersection + weighted centroid fallback for bad."""
    split_point_all, bad_all = get_approx_ray_intersections(stats.within_var_rays)
    split_points = split_point_all[clone_mask]
    bad = bad_all[clone_mask]

    tet_mom = stats.tet_moments[clone_mask]
    weighted_centroid = safe_math.safe_div(tet_mom[:, :3], tet_mom[:, 3:4])
    valid_wc = tet_mom[:, 3] > 0

    # Use weighted centroid for bad intersections with valid moments
    bad_valid = bad & valid_wc
    split_points[bad_valid] = weighted_centroid[bad_valid]

    # For bad with invalid moments, use tet centroid
    bad_invalid = bad & ~valid_wc
    if bad_invalid.any():
        clone_indices = model.indices[clone_mask]
        centroids = model.vertices[clone_indices].mean(dim=1)
        split_points[bad_invalid] = centroids[bad_invalid]

    return split_points, bad


def compute_split_points_angular_diversity(rays_topk_all, clone_mask, model, device):
    """Strategy D: Select diverse pair from top-K, then ray intersection."""
    rays_topk = rays_topk_all[clone_mask]

    # Select most angularly diverse pair
    diverse_rays, best_cos = select_diverse_pair(rays_topk)

    # Compute intersection using the diverse pair
    split_points, bad = get_approx_ray_intersections(diverse_rays)
    # Clamped fallback is already built into get_approx_ray_intersections
    return split_points, bad


# ---------------------------------------------------------------------------
# Placement quality statistics
# ---------------------------------------------------------------------------

def placement_statistics(split_points, clone_mask, model, bad, device):
    """Compute placement quality metrics."""
    clone_indices = model.indices[clone_mask]
    verts = model.vertices[clone_indices]
    centroids = verts.mean(dim=1)
    tet_min = verts.min(dim=1).values
    tet_max = verts.max(dim=1).values
    margin = (tet_max - tet_min) * 0.1

    inside_bbox = (
        (split_points >= tet_min - margin) &
        (split_points <= tet_max + margin)
    ).all(dim=1)

    dist_to_centroid = torch.norm(split_points - centroids, dim=1)
    tet_diameter = torch.norm(tet_max - tet_min, dim=1)
    relative_dist = dist_to_centroid / (tet_diameter + 1e-10)

    return {
        "inside_bbox_pct": 100 * inside_bbox.float().mean().item(),
        "mean_rel_dist": relative_dist.mean().item(),
        "median_rel_dist": relative_dist.median().item(),
        "bad_pct": 100 * bad.float().mean().item(),
        "n_bad": bad.sum().item(),
        "n_total": bad.shape[0],
    }


# ---------------------------------------------------------------------------
# Retrain and evaluate
# ---------------------------------------------------------------------------

def retrain_and_evaluate(
    strategy_name, split_points, ckpt_path, train_cameras, args, device,
    retrain_iters=500, eval_interval=100, max_eval_cameras=30,
):
    """Load fresh model, add points, retrain backbone, measure PSNR."""
    print(f"\n--- {strategy_name} ({split_points.shape[0]} points) ---")

    model, _ = load_model_and_config(ckpt_path, device)
    tet_optim = TetOptimizer(model, **args.as_dict())

    # Baseline PSNR
    base_psnr = compute_mean_psnr(
        model, train_cameras, tile_size=args.tile_size, min_t=args.min_t,
        max_cameras=max_eval_cameras)
    print(f"  Baseline PSNR: {base_psnr:.2f}")

    # Add ALL candidate points
    tet_optim.split(split_points, **args.as_dict())
    n_verts_after = model.vertices.shape[0]
    n_tets_after = model.indices.shape[0]
    print(f"  After split: {n_verts_after} vertices, {n_tets_after} tets")

    # PSNR right after adding (before retraining)
    psnr_after_add = compute_mean_psnr(
        model, train_cameras, tile_size=args.tile_size, min_t=args.min_t,
        max_cameras=max_eval_cameras)
    print(f"  After adding:  {psnr_after_add:.2f}")

    # Retrain backbone
    model.train()
    cam_indices = list(range(len(train_cameras)))
    eval_points = set(range(eval_interval, retrain_iters + 1, eval_interval))

    psnr_curve = [(0, psnr_after_add)]
    t0 = time.time()

    for it in range(1, retrain_iters + 1):
        idx = random.choice(cam_indices)
        camera = train_cameras[idx]
        target = camera.original_image.cuda()
        gt_mask = camera.gt_alpha_mask.cuda()

        ray_jitter = torch.rand(
            (camera.image_height, camera.image_width, 2), device=device)
        render_pkg = render(camera, model, ray_jitter=ray_jitter, **args.as_dict())
        image = render_pkg['render']

        l1_loss = ((target - image).abs() * gt_mask).mean()
        ssim_loss = (
            1 - fused_ssim(image.unsqueeze(0), target.unsqueeze(0))
        ).clip(min=0, max=1)
        reg = tet_optim.regularizer(render_pkg, **args.as_dict())
        loss = ((1 - args.lambda_ssim) * l1_loss
                + args.lambda_ssim * ssim_loss + reg)

        loss.backward()
        tet_optim.main_step()
        tet_optim.main_zero_grad()

        if it in eval_points:
            psnr = compute_mean_psnr(
                model, train_cameras, tile_size=args.tile_size,
                min_t=args.min_t, max_cameras=max_eval_cameras)
            psnr_curve.append((it, psnr))
            elapsed = time.time() - t0
            print(f"  Iter {it:4d}: PSNR={psnr:.2f}  ({elapsed:.0f}s)")
            model.train()

    del model, tet_optim
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": strategy_name,
        "baseline_psnr": base_psnr,
        "psnr_after_add": psnr_after_add,
        "psnr_curve": psnr_curve,
        "final_psnr": psnr_curve[-1][1],
        "num_points": int(split_points.shape[0]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_results, placement_stats, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    # 1. PSNR convergence curves
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, res in enumerate(all_results):
        iters = [p[0] for p in res['psnr_curve']]
        psnrs = [p[1] for p in res['psnr_curve']]
        ax.plot(iters, psnrs, color=colors[i], marker=markers[i],
                markersize=6, linewidth=2,
                label=f"{res['name']} (final={res['final_psnr']:.2f})")
        ax.axhline(res['baseline_psnr'], color=colors[i],
                   linestyle=':', alpha=0.3)
    ax.set_xlabel('Retraining Iterations')
    ax.set_ylabel('Training PSNR (dB)')
    ax.set_title('Placement Strategy Comparison:\nPSNR After Adding All Candidates + Backbone Retraining')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "placement_psnr_convergence.png", dpi=150)
    plt.close()

    # 2. Improvement bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [r['name'] for r in all_results]
    baseline = all_results[0]['baseline_psnr']
    improvements = [r['final_psnr'] - baseline for r in all_results]
    bars = ax.bar(names, improvements, color=colors[:len(names)])
    ax.set_ylabel('PSNR Improvement over Baseline (dB)')
    ax.set_title('Total PSNR Improvement by Placement Strategy')
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{imp:+.3f}', ha='center', va='bottom', fontsize=11)
    ax.axhline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "placement_improvement.png", dpi=150)
    plt.close()

    # 3. Placement quality comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names_short = [r['name'].split(': ')[1] if ': ' in r['name']
                   else r['name'] for r in all_results]
    x = np.arange(len(names_short))

    # Bad intersection %
    bad_pcts = [placement_stats[r['name']]['bad_pct'] for r in all_results]
    axes[0].bar(x, bad_pcts, color=colors[:len(names_short)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names_short, rotation=15)
    axes[0].set_ylabel('% Bad Intersections')
    axes[0].set_title('Bad Intersection Rate')

    # Inside bbox %
    bbox_pcts = [placement_stats[r['name']]['inside_bbox_pct']
                 for r in all_results]
    axes[1].bar(x, bbox_pcts, color=colors[:len(names_short)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names_short, rotation=15)
    axes[1].set_ylabel('% Inside BBox')
    axes[1].set_title('Points Inside Tet BBox')

    # Mean relative distance
    rel_dists = [placement_stats[r['name']]['mean_rel_dist']
                 for r in all_results]
    axes[2].bar(x, rel_dists, color=colors[:len(names_short)])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names_short, rotation=15)
    axes[2].set_ylabel('Mean Relative Distance')
    axes[2].set_title('Distance to Centroid / Tet Diameter')

    plt.tight_layout()
    plt.savefig(output_dir / "placement_quality.png", dpi=150)
    plt.close()

    # Save raw results
    json_results = []
    for r in all_results:
        rr = dict(r)
        rr['psnr_curve'] = [[it, psnr] for it, psnr in r['psnr_curve']]
        rr['placement_stats'] = placement_stats[r['name']]
        json_results.append(rr)
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_results, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate vertex placement strategies")
    parser.add_argument("--ckpt", type=str, default="output/bicycle_base")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--output-dir", type=str,
                        default="output/placement_eval")
    parser.add_argument("--retrain-iters", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--max-eval-cameras", type=int, default=30)
    parser.add_argument("--topk", type=int, default=5,
                        help="K for angular diversity (top-K by SSIM)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = Args.load_from_json(str(Path(args.ckpt) / "config.json"))
    dataset_path = args.dataset if args.dataset else config.dataset_path

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    train_cameras, test_cameras, _ = loader.load_dataset(
        Path(dataset_path), config.image_folder, data_device='cpu',
        eval=config.eval, resolution=config.resolution)
    print(f"  {len(train_cameras)} train cameras")

    # Load model for stats collection
    print("Loading checkpoint...")
    model, _ = load_model_and_config(args.ckpt, device)
    print(f"  {model.vertices.shape[0]} vertices, {model.indices.shape[0]} tets")

    # Collect stats with top-K (K=5 by default)
    print(f"\nCollecting densification stats (top-{args.topk})...")
    num_stat_cams = min(200, len(train_cameras))
    sampler = SimpleSampler(len(train_cameras), num_stat_cams, device)
    sampled_cams = [train_cameras[i] for i in sampler.nextids()]

    model.eval()
    stats, rays_topk_all = collect_render_stats_topk(
        sampled_cams, model, config, device, K=args.topk)

    # Compute candidates (same mask for all strategies)
    clone_mask, within_var, total_var = compute_candidates(
        stats, model, config, device)
    n_cand = clone_mask.sum().item()
    print(f"  {n_cand} candidates identified")

    # --- Compute split points for each strategy ---
    print("\nComputing split points for each strategy...")
    strategies = {}
    all_placement_stats = {}

    # A: Current (random barycentric fallback)
    torch.manual_seed(args.seed)
    pts_A, bad_A = compute_split_points_current(
        stats, clone_mask, model, device)
    st_A = placement_statistics(pts_A, clone_mask, model, bad_A, device)
    strategies["A: current"] = pts_A
    all_placement_stats["A: current"] = st_A
    print(f"  A (current):     {st_A['n_bad']}/{st_A['n_total']} bad ({st_A['bad_pct']:.1f}%), "
          f"{st_A['inside_bbox_pct']:.1f}% in bbox, rel_dist={st_A['mean_rel_dist']:.4f}")

    # B: Clamped for all
    pts_B, bad_B = compute_split_points_clamped(
        stats, clone_mask, model, device)
    st_B = placement_statistics(pts_B, clone_mask, model, bad_B, device)
    strategies["B: clamped"] = pts_B
    all_placement_stats["B: clamped"] = st_B
    print(f"  B (clamped):     {st_B['n_bad']}/{st_B['n_total']} bad ({st_B['bad_pct']:.1f}%), "
          f"{st_B['inside_bbox_pct']:.1f}% in bbox, rel_dist={st_B['mean_rel_dist']:.4f}")

    # C: Weighted centroid fallback
    pts_C, bad_C = compute_split_points_weighted_centroid(
        stats, clone_mask, model, device)
    st_C = placement_statistics(pts_C, clone_mask, model, bad_C, device)
    strategies["C: wt_centroid"] = pts_C
    all_placement_stats["C: wt_centroid"] = st_C
    print(f"  C (wt_centroid): {st_C['n_bad']}/{st_C['n_total']} bad ({st_C['bad_pct']:.1f}%), "
          f"{st_C['inside_bbox_pct']:.1f}% in bbox, rel_dist={st_C['mean_rel_dist']:.4f}")

    # D: Angular diversity
    pts_D, bad_D = compute_split_points_angular_diversity(
        rays_topk_all, clone_mask, model, device)
    st_D = placement_statistics(pts_D, clone_mask, model, bad_D, device)
    strategies["D: angular_div"] = pts_D
    all_placement_stats["D: angular_div"] = st_D
    print(f"  D (angular_div): {st_D['n_bad']}/{st_D['n_total']} bad ({st_D['bad_pct']:.1f}%), "
          f"{st_D['inside_bbox_pct']:.1f}% in bbox, rel_dist={st_D['mean_rel_dist']:.4f}")

    # --- Angular diversity analysis ---
    rays_topk_cand = rays_topk_all[clone_mask]
    _, best_cos = select_diverse_pair(rays_topk_cand)

    # Compare top-2-by-SSIM angle vs top-2-by-diversity angle
    top2_rays = stats.within_var_rays[clone_mask]
    d1_ssim = top2_rays[:, 0, 3:] - top2_rays[:, 0, :3]
    d2_ssim = top2_rays[:, 1, 3:] - top2_rays[:, 1, :3]
    cos_ssim = (
        (d1_ssim * d2_ssim).sum(dim=1)
        / (torch.norm(d1_ssim, dim=1) * torch.norm(d2_ssim, dim=1) + 1e-10)
    ).abs()

    print(f"\n--- Angular diversity improvement ---")
    print(f"  SSIM-selected pair |cos|:    mean={cos_ssim.mean():.4f}, "
          f">0.95: {(cos_ssim > 0.95).sum().item()}, "
          f">0.99: {(cos_ssim > 0.99).sum().item()}")
    print(f"  Diverse-selected pair |cos|: mean={best_cos.mean():.4f}, "
          f">0.95: {(best_cos > 0.95).sum().item()}, "
          f">0.99: {(best_cos > 0.99).sum().item()}")

    # How many points differ between strategies?
    diff_AB = torch.norm(pts_A - pts_B, dim=1)
    diff_AC = torch.norm(pts_A - pts_C, dim=1)
    diff_AD = torch.norm(pts_A - pts_D, dim=1)
    print(f"\n--- Point differences from baseline (A) ---")
    print(f"  A vs B (clamped):     {(diff_AB > 1e-6).sum().item()} differ "
          f"({100*(diff_AB > 1e-6).float().mean():.1f}%), "
          f"mean dist={diff_AB[diff_AB > 1e-6].mean():.6f}" if (diff_AB > 1e-6).any()
          else "  A vs B: identical")
    print(f"  A vs C (wt_centroid): {(diff_AC > 1e-6).sum().item()} differ "
          f"({100*(diff_AC > 1e-6).float().mean():.1f}%), "
          f"mean dist={diff_AC[diff_AC > 1e-6].mean():.6f}" if (diff_AC > 1e-6).any()
          else "  A vs C: identical")
    print(f"  A vs D (angular_div): {(diff_AD > 1e-6).sum().item()} differ "
          f"({100*(diff_AD > 1e-6).float().mean():.1f}%), "
          f"mean dist={diff_AD[diff_AD > 1e-6].mean():.6f}" if (diff_AD > 1e-6).any()
          else "  A vs D: identical")

    # Free model before retraining loop
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Evaluate each strategy ---
    print(f"\n{'='*60}")
    print("Starting placement quality evaluation...")
    print(f"  Retrain iterations: {args.retrain_iters}")
    print(f"  Eval interval:      {args.eval_interval}")
    print(f"  Max eval cameras:   {args.max_eval_cameras}")
    print(f"{'='*60}")

    all_results = []

    for name, pts in strategies.items():
        # Reset seeds for reproducible retraining
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        result = retrain_and_evaluate(
            name, pts, args.ckpt, train_cameras, config, device,
            retrain_iters=args.retrain_iters,
            eval_interval=args.eval_interval,
            max_eval_cameras=args.max_eval_cameras)
        result['placement_stats'] = all_placement_stats[name]
        all_results.append(result)

        # Save intermediate
        with open(output_dir / f"result_{name.split(': ')[1]}.json", "w") as f:
            rr = dict(result)
            rr['psnr_curve'] = [[it, p] for it, p in result['psnr_curve']]
            json.dump(rr, f, indent=2)

    # --- Summary ---
    print(f"\n{'='*90}")
    print(f"{'Strategy':20s} | {'Baseline':>8s} | {'After Add':>9s} | "
          f"{'Final':>8s} | {'Delta':>8s} | "
          f"{'%Bad':>6s} | {'%InBBox':>7s} | {'RelDist':>7s}")
    print("-" * 90)
    for r in all_results:
        delta = r['final_psnr'] - r['baseline_psnr']
        ps = r['placement_stats']
        print(f"{r['name']:20s} | {r['baseline_psnr']:8.2f} | "
              f"{r['psnr_after_add']:9.2f} | {r['final_psnr']:8.2f} | "
              f"{delta:+8.3f} | {ps['bad_pct']:5.1f}% | "
              f"{ps['inside_bbox_pct']:6.1f}% | {ps['mean_rel_dist']:7.4f}")
    print(f"{'='*90}")

    # Plot
    plot_results(all_results, all_placement_stats, output_dir)
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
