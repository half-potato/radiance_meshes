"""
Investigate why 19% of ray intersections are "bad" and test alternative
placement strategies for densification.

Failure modes to check:
  1. Zero-length segments (averaged entry ≈ exit)
  2. Near-parallel segments (top-2 views from similar angles)
  3. Segments outside tet bounds
  4. Degenerate tets (near-zero volume)
"""

import os
import sys
import gc
import math
import random
from pathlib import Path
from typing import List, Tuple

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
from utils.topo_utils import tet_volumes

torch.set_num_threads(1)


def load_model_and_config(ckpt_path, device):
    config = Args.load_from_json(str(Path(ckpt_path) / "config.json"))
    model = Model.load_ckpt(Path(ckpt_path), device)
    return model, config


@torch.no_grad()
def investigate_bad_intersections(model, train_cameras, args, device, output_dir):
    """Deep investigation of why ray intersections fail."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_tets = model.indices.shape[0]
    num_stat_cameras = min(200, len(train_cameras))
    sampler = SimpleSampler(len(train_cameras), num_stat_cameras, device)
    sampled_cams = [train_cameras[i] for i in sampler.nextids()]

    model.eval()
    stats = collect_render_stats(sampled_cams, model, args, device)

    # --- Compute candidates (same as apply_densification) ---
    s0_t, s1_t, s2_t = stats.total_var_moments.T
    total_var_mu = safe_math.safe_div(s1_t, s0_t)
    total_var_std = (safe_math.safe_div(s2_t, s0_t) - total_var_mu ** 2).clip(min=0)
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

    rays_all = stats.within_var_rays  # (T, 2, 6)
    split_point_all, bad_all = get_approx_ray_intersections(rays_all)

    # Focus on candidates
    rays_cand = rays_all[clone_mask]       # (N, 2, 6)
    bad_cand = bad_all[clone_mask]         # (N,)
    split_pts_cand = split_point_all[clone_mask]
    cand_indices = model.indices[clone_mask]  # (N, 4)

    n_cand = rays_cand.shape[0]
    n_bad = bad_cand.sum().item()
    n_good = n_cand - n_bad

    print(f"Candidates: {n_cand}, Bad: {n_bad} ({100*n_bad/n_cand:.1f}%), Good: {n_good}")

    # =========================================================================
    # Analysis 1: Segment lengths
    # =========================================================================
    seg1 = rays_cand[:, 0]  # (N, 6) = [enter_xyz, exit_xyz]
    seg2 = rays_cand[:, 1]

    seg1_enter = seg1[:, :3]
    seg1_exit = seg1[:, 3:]
    seg2_enter = seg2[:, :3]
    seg2_exit = seg2[:, 3:]

    len1 = torch.norm(seg1_exit - seg1_enter, dim=1)
    len2 = torch.norm(seg2_exit - seg2_enter, dim=1)

    print(f"\n=== Segment Lengths ===")
    print(f"Seg1 - mean: {len1.mean():.6f}, median: {len1.median():.6f}, "
          f"zero (<1e-6): {(len1 < 1e-6).sum().item()}")
    print(f"Seg2 - mean: {len2.mean():.6f}, median: {len2.median():.6f}, "
          f"zero (<1e-6): {(len2 < 1e-6).sum().item()}")
    print(f"Bad seg1 len:  mean={len1[bad_cand].mean():.6f}, median={len1[bad_cand].median():.6f}")
    print(f"Good seg1 len: mean={len1[~bad_cand].mean():.6f}, median={len1[~bad_cand].median():.6f}")
    print(f"Bad seg2 len:  mean={len2[bad_cand].mean():.6f}, median={len2[bad_cand].median():.6f}")
    print(f"Good seg2 len: mean={len2[~bad_cand].mean():.6f}, median={len2[~bad_cand].median():.6f}")

    # Check for zero segments (both segments are zero → no ray data)
    both_zero = (len1 < 1e-6) & (len2 < 1e-6)
    one_zero = ((len1 < 1e-6) | (len2 < 1e-6)) & ~both_zero
    print(f"Both zero-length: {both_zero.sum().item()} ({100*both_zero.float().mean():.1f}%)")
    print(f"One zero-length:  {one_zero.sum().item()} ({100*one_zero.float().mean():.1f}%)")
    print(f"Bad AND both_zero: {(bad_cand & both_zero).sum().item()}")
    print(f"Bad AND one_zero:  {(bad_cand & one_zero).sum().item()}")

    # =========================================================================
    # Analysis 2: Parallelism between segments
    # =========================================================================
    d1 = seg1_exit - seg1_enter
    d2 = seg2_exit - seg2_enter
    d1_norm = d1 / (torch.norm(d1, dim=1, keepdim=True) + 1e-10)
    d2_norm = d2 / (torch.norm(d2, dim=1, keepdim=True) + 1e-10)
    cos_angle = (d1_norm * d2_norm).sum(dim=1).abs()

    print(f"\n=== Segment Parallelism (|cos(angle)|) ===")
    print(f"All  - mean: {cos_angle.mean():.4f}, >0.99: {(cos_angle > 0.99).sum().item()}, "
          f">0.95: {(cos_angle > 0.95).sum().item()}")
    if n_bad > 0:
        print(f"Bad  - mean: {cos_angle[bad_cand].mean():.4f}, "
              f">0.99: {(cos_angle[bad_cand] > 0.99).sum().item()}, "
              f">0.95: {(cos_angle[bad_cand] > 0.95).sum().item()}")
    if n_good > 0:
        print(f"Good - mean: {cos_angle[~bad_cand].mean():.4f}, "
              f">0.99: {(cos_angle[~bad_cand] > 0.99).sum().item()}, "
              f">0.95: {(cos_angle[~bad_cand] > 0.95).sum().item()}")

    # =========================================================================
    # Analysis 3: Are split points inside their tets?
    # =========================================================================
    verts = model.vertices[cand_indices]  # (N, 4, 3)
    centroids = verts.mean(dim=1)         # (N, 3)

    # Check if split points are within tet bounding boxes
    tet_min = verts.min(dim=1).values    # (N, 3)
    tet_max = verts.max(dim=1).values    # (N, 3)
    margin = (tet_max - tet_min) * 0.1   # 10% margin

    inside_bbox = (
        (split_pts_cand >= tet_min - margin) &
        (split_pts_cand <= tet_max + margin)
    ).all(dim=1)

    print(f"\n=== Split Point Inside Tet BBox (with 10% margin) ===")
    print(f"All:  {inside_bbox.sum().item()}/{n_cand} ({100*inside_bbox.float().mean():.1f}%)")
    print(f"Bad:  {inside_bbox[bad_cand].sum().item()}/{n_bad} ({100*inside_bbox[bad_cand].float().mean():.1f}%)")
    print(f"Good: {inside_bbox[~bad_cand].sum().item()}/{n_good} ({100*inside_bbox[~bad_cand].float().mean():.1f}%)")

    # Distance from split point to centroid vs tet diameter
    tet_diameter = torch.norm(tet_max - tet_min, dim=1)
    dist_to_centroid = torch.norm(split_pts_cand - centroids, dim=1)
    relative_dist = dist_to_centroid / (tet_diameter + 1e-10)

    print(f"\n=== Split Point Distance to Centroid (relative to tet diameter) ===")
    print(f"All:  mean={relative_dist.mean():.4f}, median={relative_dist.median():.4f}, "
          f">1.0: {(relative_dist > 1.0).sum().item()}, >5.0: {(relative_dist > 5.0).sum().item()}")
    if n_bad > 0:
        print(f"Bad:  mean={relative_dist[bad_cand].mean():.4f}, median={relative_dist[bad_cand].median():.4f}, "
              f">1.0: {(relative_dist[bad_cand] > 1.0).sum().item()}")
    if n_good > 0:
        print(f"Good: mean={relative_dist[~bad_cand].mean():.4f}, median={relative_dist[~bad_cand].median():.4f}, "
              f">1.0: {(relative_dist[~bad_cand] > 1.0).sum().item()}")

    # =========================================================================
    # Analysis 4: Tet volumes (degenerate tets?)
    # =========================================================================
    vols = tet_volumes(model.vertices[model.indices])
    cand_vols = vols[clone_mask]

    print(f"\n=== Tet Volumes ===")
    print(f"All tets - mean: {vols.abs().mean():.8f}, "
          f"near-zero (<1e-10): {(vols.abs() < 1e-10).sum().item()}")
    print(f"Candidates - mean: {cand_vols.abs().mean():.8f}")
    if n_bad > 0:
        print(f"Bad  - mean: {cand_vols[bad_cand].abs().mean():.8f}, "
              f"near-zero: {(cand_vols[bad_cand].abs() < 1e-10).sum().item()}")
    if n_good > 0:
        print(f"Good - mean: {cand_vols[~bad_cand].abs().mean():.8f}, "
              f"near-zero: {(cand_vols[~bad_cand].abs() < 1e-10).sum().item()}")

    # =========================================================================
    # Analysis 5: Detailed breakdown of bad_intersect conditions
    # =========================================================================
    # Recompute the intersection parameters to see which condition triggers
    o1 = seg1_enter
    d1_vec = seg1_exit - seg1_enter
    o2 = seg2_enter
    d2_vec = seg2_exit - seg2_enter

    v_o = o1 - o2
    a = (d1_vec * d1_vec).sum(dim=1)
    b = (d1_vec * d2_vec).sum(dim=1)
    c = (d2_vec * d2_vec).sum(dim=1)
    d_val = (d1_vec * v_o).sum(dim=1)
    e_val = (d2_vec * v_o).sum(dim=1)

    denom = a * c - b * b
    denom_safe = torch.where(denom.abs() < 1e-7, torch.ones_like(denom), denom)

    s_line = ((b * e_val) - (c * d_val)) / denom_safe
    t_line = ((a * e_val) - (b * d_val)) / denom_safe

    parallel = denom.abs() < 1e-7
    s_neg = s_line < 0
    s_over = s_line > 1
    t_neg = t_line < 0
    t_over = t_line > 1

    print(f"\n=== Bad Intersection Breakdown (among candidates) ===")
    print(f"Parallel (denom < 1e-7):     {parallel.sum().item()}")
    print(f"s < 0:                       {(s_neg & ~parallel).sum().item()}")
    print(f"s > 1:                       {(s_over & ~parallel).sum().item()}")
    print(f"t < 0:                       {(t_neg & ~parallel).sum().item()}")
    print(f"t > 1:                       {(t_over & ~parallel).sum().item()}")
    print(f"Both s,t in [0,1] (good):    {(~s_neg & ~s_over & ~t_neg & ~t_over & ~parallel).sum().item()}")

    # Among bad candidates specifically
    if n_bad > 0:
        print(f"\nAmong BAD candidates ({n_bad}):")
        print(f"  Parallel:  {parallel[bad_cand].sum().item()}")
        print(f"  s < 0:     {s_neg[bad_cand].sum().item()}")
        print(f"  s > 1:     {s_over[bad_cand].sum().item()}")
        print(f"  t < 0:     {t_neg[bad_cand].sum().item()}")
        print(f"  t > 1:     {t_over[bad_cand].sum().item()}")

        # How far outside [0,1] are s and t?
        s_bad = s_line[bad_cand]
        t_bad = t_line[bad_cand]
        print(f"\n  s_line stats: mean={s_bad.mean():.4f}, min={s_bad.min():.4f}, max={s_bad.max():.4f}")
        print(f"  t_line stats: mean={t_bad.mean():.4f}, min={t_bad.min():.4f}, max={t_bad.max():.4f}")

        # How many are just slightly outside?
        s_near = ((s_line > -0.5) & (s_line < 1.5))[bad_cand].sum().item()
        t_near = ((t_line > -0.5) & (t_line < 1.5))[bad_cand].sum().item()
        print(f"  s in [-0.5, 1.5]: {s_near}/{n_bad}")
        print(f"  t in [-0.5, 1.5]: {t_near}/{n_bad}")

    # =========================================================================
    # Analysis 6: What does the clamped point look like for "bad" cases?
    # =========================================================================
    s_clamped = s_line.clamp(0, 1)
    t_clamped = t_line.clamp(0, 1)
    pc1 = o1 + s_clamped.unsqueeze(1) * d1_vec
    pc2 = o2 + t_clamped.unsqueeze(1) * d2_vec
    clamped_point = (pc1 + pc2) / 2.0

    # For bad intersections, how far is the clamped point from centroid?
    clamped_dist = torch.norm(clamped_point - centroids, dim=1)
    clamped_rel = clamped_dist / (tet_diameter + 1e-10)

    if n_bad > 0:
        print(f"\n=== Clamped Point Quality (bad candidates) ===")
        print(f"Clamped dist to centroid (relative): "
              f"mean={clamped_rel[bad_cand].mean():.4f}, "
              f"median={clamped_rel[bad_cand].median():.4f}")
        print(f"Centroid dist to centroid (should be 0): "
              f"mean={(torch.norm(centroids - centroids, dim=1)[bad_cand]).mean():.4f}")

        # Compare: clamped point vs centroid vs random barycentric
        # The clamped point IS what get_approx_ray_intersections returns
        # (it clamps s,t to [0,1] and returns midpoint regardless of bad flag)
        # But then apply_densification replaces bad ones with random barycentric

        # Check if clamped is actually inside bbox
        clamped_inside = (
            (clamped_point >= tet_min - margin) &
            (clamped_point <= tet_max + margin)
        ).all(dim=1)
        print(f"Clamped inside bbox: {clamped_inside[bad_cand].sum().item()}/{n_bad} "
              f"({100*clamped_inside[bad_cand].float().mean():.1f}%)")

    # =========================================================================
    # Analysis 7: Alternative placement strategies and their quality
    # =========================================================================
    print(f"\n=== Alternative Placement Strategies ===")

    # Strategy A: Centroid (always inside tet by convexity)
    centroid_dist_to_bbox_center = torch.norm(
        centroids - (tet_min + tet_max) / 2, dim=1)
    print(f"A. Centroid: always inside, dist to bbox center: "
          f"mean={centroid_dist_to_bbox_center.mean():.6f}")

    # Strategy B: Weighted centroid using moments (tet_moments[:, :3] / tet_moments[:, 3])
    tet_mom = stats.tet_moments[clone_mask]
    weighted_centroid = safe_math.safe_div(tet_mom[:, :3], tet_mom[:, 3:4])
    wc_valid = (tet_mom[:, 3] > 0)
    wc_dist = torch.norm(weighted_centroid - centroids, dim=1)
    wc_inside_bbox = (
        (weighted_centroid >= tet_min - margin) &
        (weighted_centroid <= tet_max + margin)
    ).all(dim=1)
    print(f"B. Weighted centroid (from tet_moments): valid={wc_valid.sum().item()}/{n_cand}, "
          f"inside bbox: {(wc_inside_bbox & wc_valid).sum().item()}, "
          f"dist to centroid: mean={wc_dist[wc_valid].mean():.6f}")

    # Strategy C: Use clamped intersection for all (don't fall back to random)
    clamped_inside_all = (
        (clamped_point >= tet_min - margin) &
        (clamped_point <= tet_max + margin)
    ).all(dim=1)
    print(f"C. Clamped intersection (all): inside bbox: {clamped_inside_all.sum().item()}/{n_cand} "
          f"({100*clamped_inside_all.float().mean():.1f}%)")

    # Strategy D: Hybrid - use ray intersection when good, weighted centroid when bad
    hybrid_pts = split_pts_cand.clone()
    if wc_valid.any():
        use_wc = bad_cand & wc_valid
        hybrid_pts[use_wc] = weighted_centroid[use_wc]
    hybrid_inside = (
        (hybrid_pts >= tet_min - margin) &
        (hybrid_pts <= tet_max + margin)
    ).all(dim=1)
    print(f"D. Hybrid (ray when good, weighted centroid when bad): "
          f"inside bbox: {hybrid_inside.sum().item()}/{n_cand} "
          f"({100*hybrid_inside.float().mean():.1f}%)")

    # Strategy E: Use entry point midpoint of the two segments
    entry_mid = (seg1_enter + seg2_enter) / 2
    entry_inside = (
        (entry_mid >= tet_min - margin) &
        (entry_mid <= tet_max + margin)
    ).all(dim=1)
    print(f"E. Entry midpoint: inside bbox: {entry_inside.sum().item()}/{n_cand} "
          f"({100*entry_inside.float().mean():.1f}%)")

    # Strategy F: Segment midpoint of the longer segment
    longer_mask = len1 > len2
    longer_mid = torch.where(longer_mask.unsqueeze(1),
                              (seg1_enter + seg1_exit) / 2,
                              (seg2_enter + seg2_exit) / 2)
    longer_inside = (
        (longer_mid >= tet_min - margin) &
        (longer_mid <= tet_max + margin)
    ).all(dim=1)
    print(f"F. Longer segment midpoint: inside bbox: {longer_inside.sum().item()}/{n_cand} "
          f"({100*longer_inside.float().mean():.1f}%)")

    # =========================================================================
    # Plots
    # =========================================================================

    # Plot 1: Segment length distribution (good vs bad)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(len1[~bad_cand].cpu().numpy(), bins=80, alpha=0.6, label='Good', color='steelblue', density=True)
    if n_bad > 0:
        axes[0].hist(len1[bad_cand].cpu().numpy(), bins=80, alpha=0.6, label='Bad', color='coral', density=True)
    axes[0].set_xlabel('Segment 1 Length')
    axes[0].set_title('Segment 1 Length Distribution')
    axes[0].legend()

    axes[1].hist(len2[~bad_cand].cpu().numpy(), bins=80, alpha=0.6, label='Good', color='steelblue', density=True)
    if n_bad > 0:
        axes[1].hist(len2[bad_cand].cpu().numpy(), bins=80, alpha=0.6, label='Bad', color='coral', density=True)
    axes[1].set_xlabel('Segment 2 Length')
    axes[1].set_title('Segment 2 Length Distribution')
    axes[1].legend()

    axes[2].hist(cos_angle[~bad_cand].cpu().numpy(), bins=80, alpha=0.6, label='Good', color='steelblue', density=True)
    if n_bad > 0:
        axes[2].hist(cos_angle[bad_cand].cpu().numpy(), bins=80, alpha=0.6, label='Bad', color='coral', density=True)
    axes[2].set_xlabel('|cos(angle)| between segments')
    axes[2].set_title('Segment Parallelism')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "segment_analysis.png", dpi=150)
    plt.close()

    # Plot 2: s_line and t_line distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    s_np = s_line.cpu().numpy()
    t_np = t_line.cpu().numpy()
    # Clip for visualization
    s_clip = np.clip(s_np, -3, 4)
    t_clip = np.clip(t_np, -3, 4)

    axes[0].hist(s_clip[~bad_cand.cpu().numpy()], bins=100, alpha=0.6, label='Good', color='steelblue', density=True)
    if n_bad > 0:
        axes[0].hist(s_clip[bad_cand.cpu().numpy()], bins=100, alpha=0.6, label='Bad', color='coral', density=True)
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[0].axvline(1, color='black', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('s parameter (clipped to [-3, 4])')
    axes[0].set_title('s_line Distribution')
    axes[0].legend()

    axes[1].hist(t_clip[~bad_cand.cpu().numpy()], bins=100, alpha=0.6, label='Good', color='steelblue', density=True)
    if n_bad > 0:
        axes[1].hist(t_clip[bad_cand.cpu().numpy()], bins=100, alpha=0.6, label='Bad', color='coral', density=True)
    axes[1].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[1].axvline(1, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('t parameter (clipped to [-3, 4])')
    axes[1].set_title('t_line Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_distributions.png", dpi=150)
    plt.close()

    # Plot 3: Relative distance comparison of strategies
    strategies = {}
    strategies['Ray intersection (good)'] = relative_dist[~bad_cand].cpu().numpy() if n_good > 0 else np.array([])
    strategies['Ray intersection (bad)'] = relative_dist[bad_cand].cpu().numpy() if n_bad > 0 else np.array([])
    strategies['Clamped (bad)'] = clamped_rel[bad_cand].cpu().numpy() if n_bad > 0 else np.array([])
    if wc_valid.any():
        wc_rel = torch.norm(weighted_centroid - centroids, dim=1) / (tet_diameter + 1e-10)
        strategies['Weighted centroid'] = wc_rel[wc_valid].cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    data = [v for v in strategies.values() if len(v) > 0]
    labels = [k for k, v in strategies.items() if len(v) > 0]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel('Distance to centroid / tet diameter')
    ax.set_title('Placement Quality by Strategy')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "strategy_comparison.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="output/bicycle_base")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="output/placement_investigation")
    cli = parser.parse_args()

    device = torch.device('cuda')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model, config = load_model_and_config(cli.ckpt, device)
    dataset_path = cli.dataset if cli.dataset else config.dataset_path
    train_cameras, test_cameras, scene_info = loader.load_dataset(
        Path(dataset_path), config.image_folder, data_device='cpu',
        eval=config.eval, resolution=config.resolution)
    print(f"Loaded {len(train_cameras)} train cameras")
    print(f"Model: {model.vertices.shape[0]} vertices, {model.indices.shape[0]} tets")

    investigate_bad_intersections(model, train_cameras, config, device, cli.output_dir)
