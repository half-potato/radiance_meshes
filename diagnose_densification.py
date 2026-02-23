"""
Densification Diagnostic Script
================================
Loads a pre-densification checkpoint and analyzes the densification heuristic:
  Part 1: Static diagnostics (error heatmaps, decision maps, statistics)
  Part 2: Batch ordering experiment (PSNR vs #points for different strategies)
  Part 3: Placement quality analysis (ray-intersection vs centroid vs circumcenter)

Usage:
  python diagnose_densification.py --ckpt output/bicycle_base --dataset /data/nerf_datasets/360/bicycle
  python diagnose_densification.py --ckpt output/bicycle_base --dataset /data/nerf_datasets/360/bicycle --static-only
  python diagnose_densification.py --ckpt output/bicycle_base --dataset /data/nerf_datasets/360/bicycle --ordering-only
"""

import os
import sys
import gc
import json
import math
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, NamedTuple

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Project imports
from data import loader
from models.ingp_color import Model, TetOptimizer
from utils.args import Args
from utils.train_util import render, SimpleSampler, render_debug
from utils.densification import (
    collect_render_stats, get_approx_ray_intersections, RenderStats
)
from utils import safe_math
from fused_ssim import fused_ssim

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class CandidateInfo(NamedTuple):
    clone_mask: torch.Tensor       # (T,) bool
    within_var: torch.Tensor       # (T,) float - all tets
    total_var: torch.Tensor        # (T,) float - all tets
    within_mask: torch.Tensor      # (T,) bool
    total_mask: torch.Tensor       # (T,) bool
    top_ssim: torch.Tensor         # (T, 2) float
    peak_contrib: torch.Tensor     # (T,) float
    split_points: torch.Tensor     # (N_cand, 3)
    bad_intersections: torch.Tensor  # (N_cand,) bool
    tet_centroids: torch.Tensor    # (N_cand, 3)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_model_and_config(ckpt_path: str, device: torch.device):
    """Load model and config from checkpoint directory."""
    config = Args.load_from_json(str(Path(ckpt_path) / "config.json"))
    model = Model.load_ckpt(Path(ckpt_path), device)
    return model, config


@torch.no_grad()
def compute_mean_psnr(model, cameras, tile_size=4, min_t=0.4, max_cameras=-1):
    """Compute mean PSNR, L1, and SSIM across cameras."""
    model.eval()
    if max_cameras > 0 and max_cameras < len(cameras):
        indices = list(range(len(cameras)))
        random.shuffle(indices)
        cameras = [cameras[i] for i in indices[:max_cameras]]

    psnrs, l1s, ssims = [], [], []
    for cam in cameras:
        target = cam.original_image.cuda()
        gt_mask = cam.gt_alpha_mask.cuda()
        render_pkg = render(cam, model, tile_size=tile_size, min_t=min_t)
        image = render_pkg['render']

        l2 = ((target - image) ** 2 * gt_mask).mean().item()
        l1 = ((target - image).abs() * gt_mask).mean().item()
        ssim_val = fused_ssim(image.unsqueeze(0), target.unsqueeze(0)).item()
        psnr_val = -20 * math.log10(math.sqrt(max(l2, 1e-10)))

        psnrs.append(psnr_val)
        l1s.append(l1)
        ssims.append(ssim_val)

        del render_pkg, image, target, gt_mask
        torch.cuda.empty_cache()

    return np.mean(psnrs), np.mean(l1s), np.mean(ssims)


@torch.no_grad()
def collect_candidates(model, train_cameras, args, device, num_stat_cameras=200):
    """Collect densification stats and identify all candidates.

    Replicates the scoring logic from apply_densification() without
    actually modifying the model.
    """
    num_stat_cameras = min(num_stat_cameras, len(train_cameras))
    sampler = SimpleSampler(len(train_cameras), num_stat_cameras, device)
    sampled_cams = [train_cameras[i] for i in sampler.nextids()]

    model.eval()
    stats = collect_render_stats(sampled_cams, model, args, device)

    # --- Score computation (mirrors apply_densification lines 260-308) ---
    s0_t, s1_t, s2_t = stats.total_var_moments.T
    total_var_mu = safe_math.safe_div(s1_t, s0_t)
    total_var_std = (safe_math.safe_div(s2_t, s0_t) - total_var_mu ** 2).clip(min=0)
    total_var_std[s0_t < 1] = 0

    N_b = stats.tet_view_count
    within_var = stats.top_ssim.sum(dim=1)
    total_var = s0_t * total_var_std
    total_var[(N_b < 2) | (s0_t < 1)] = 0

    # Alive / contribution masking
    total_var[stats.peak_contrib < args.clone_min_contrib] = 0
    within_var[stats.peak_contrib < args.split_min_contrib] = 0

    within_mask = within_var > args.within_thresh
    total_mask = total_var > args.total_thresh
    clone_mask = within_mask | total_mask

    # --- Compute split points ---
    split_point_all, bad_all = get_approx_ray_intersections(stats.within_var_rays)
    split_points = split_point_all[clone_mask]
    bad = bad_all[clone_mask]

    # Fallback to random barycentric for bad intersections
    clone_indices = model.indices[clone_mask]
    barycentric = torch.rand(
        (clone_indices.shape[0], clone_indices.shape[1], 1), device=device
    ).clip(min=0.01, max=0.99)
    barycentric_weights = barycentric / (1e-3 + barycentric.sum(dim=1, keepdim=True))
    random_locations = (model.vertices[clone_indices] * barycentric_weights).sum(dim=1)
    split_points[bad] = random_locations[bad]

    # Tet centroids
    tet_centroids = model.vertices[clone_indices].mean(dim=1)

    candidates = CandidateInfo(
        clone_mask=clone_mask,
        within_var=within_var,
        total_var=total_var,
        within_mask=within_mask,
        total_mask=total_mask,
        top_ssim=stats.top_ssim,
        peak_contrib=stats.peak_contrib,
        split_points=split_points,
        bad_intersections=bad,
        tet_centroids=tet_centroids,
    )
    return stats, candidates


@torch.no_grad()
def project_points_to_image(points, camera, device):
    """Project 3D world points to 2D pixel coordinates.

    Returns (px, py, valid_mask).
    """
    N = points.shape[0]
    p_homo = torch.cat([points, torch.ones(N, 1, device=device)], dim=1)
    p_view = p_homo @ camera.world_view_transform.to(device)

    z = p_view[:, 2]
    cx = camera.cx if camera.cx != -1 else camera.image_width / 2
    cy = camera.cy if camera.cy != -1 else camera.image_height / 2
    px = camera.fx * p_view[:, 0] / z + cx
    py = camera.fy * p_view[:, 1] / z + cy

    valid = (z > 0.1) & (px >= 0) & (px < camera.image_width) & (py >= 0) & (py < camera.image_height)
    return px, py, valid


# ---------------------------------------------------------------------------
# Part 1: Static Diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_static_diagnostics(model, train_cameras, stats, candidates, args, output_dir, device):
    """Generate all static diagnostic outputs (no retraining)."""
    static_dir = Path(output_dir) / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    n_tets = model.indices.shape[0]
    n_alive = (candidates.peak_contrib > args.clone_min_contrib).sum().item()
    n_within = candidates.within_mask.sum().item()
    n_total = candidates.total_mask.sum().item()
    n_candidates = candidates.clone_mask.sum().item()
    n_bad = candidates.bad_intersections.sum().item()
    n_verts = model.vertices.shape[0]

    wv = candidates.within_var
    tv = candidates.total_var
    cand_wv = wv[candidates.clone_mask]
    cand_tv = tv[candidates.clone_mask]

    # --- 1. Statistics table ---
    lines = [
        "=== Densification Diagnostics ===",
        f"Total tetrahedra:       {n_tets}",
        f"Total vertices:         {n_verts}",
        f"Interior vertices:      {model.num_int_verts}",
        f"Alive tets (contrib > {args.clone_min_contrib:.4f}): {n_alive}",
        f"Candidates (clone_mask): {n_candidates}",
        f"  Within-var triggered: {n_within}",
        f"  Total-var triggered:  {n_total}",
        f"  Bad ray intersections: {n_bad} ({100*n_bad/max(n_candidates,1):.1f}%)",
        "",
        "--- Within-var (all tets) ---",
        f"  Mean:   {wv.mean().item():.6f}",
        f"  Std:    {wv.std().item():.6f}",
        f"  Median: {wv.median().item():.6f}",
        f"  P90:    {torch.quantile(wv, 0.9).item():.6f}",
        f"  P99:    {torch.quantile(wv, 0.99).item():.6f}",
        f"  Thresh: {args.within_thresh:.6f}",
        "",
        "--- Total-var (all tets) ---",
        f"  Mean:   {tv.mean().item():.6f}",
        f"  Std:    {tv.std().item():.6f}",
        f"  Median: {tv.median().item():.6f}",
        f"  P90:    {torch.quantile(tv, 0.9).item():.6f}",
        f"  P99:    {torch.quantile(tv, 0.99).item():.6f}",
        f"  Thresh: {args.total_thresh:.6f}",
        "",
        "--- Peak contribution ---",
        f"  Mean:   {candidates.peak_contrib.mean().item():.6f}",
        f"  Median: {candidates.peak_contrib.median().item():.6f}",
        f"  P10:    {torch.quantile(candidates.peak_contrib, 0.1).item():.6f}",
        f"  P90:    {torch.quantile(candidates.peak_contrib, 0.9).item():.6f}",
    ]
    if n_candidates > 0:
        lines += [
            "",
            "--- Candidate scores ---",
            f"  Within-var mean: {cand_wv.mean().item():.6f}",
            f"  Total-var mean:  {cand_tv.mean().item():.6f}",
            f"  Combined mean:   {(cand_wv + cand_tv).mean().item():.6f}",
        ]
    table_text = "\n".join(lines)
    print(table_text)
    with open(static_dir / "statistics_table.txt", "w") as f:
        f.write(table_text)

    # --- 2. Error histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    wv_np = wv[wv > 0].cpu().numpy()
    if len(wv_np) > 0:
        axes[0].hist(wv_np, bins=100, alpha=0.7, color='steelblue')
    axes[0].axvline(args.within_thresh, color='red', linestyle='--', label=f'thresh={args.within_thresh:.4f}')
    axes[0].set_xlabel('Within-var score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Within-Image Variance Distribution')
    axes[0].legend()
    axes[0].set_yscale('log')

    tv_np = tv[tv > 0].cpu().numpy()
    if len(tv_np) > 0:
        axes[1].hist(tv_np, bins=100, alpha=0.7, color='coral')
    axes[1].axvline(args.total_thresh, color='red', linestyle='--', label=f'thresh={args.total_thresh:.4f}')
    axes[1].set_xlabel('Total-var score')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Total Variance Distribution')
    axes[1].legend()
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(static_dir / "error_histogram.png", dpi=150)
    plt.close()

    # --- 3. Score distributions (box plots) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cand_idx = candidates.clone_mask.cpu().numpy()
    non_cand_idx = ~candidates.clone_mask.cpu().numpy()

    data_wv = [wv[non_cand_idx].cpu().numpy(), wv[cand_idx].cpu().numpy()]
    axes[0].boxplot(data_wv, labels=['Non-candidates', 'Candidates'], showfliers=False)
    axes[0].set_title('Within-var')

    data_tv = [tv[non_cand_idx].cpu().numpy(), tv[cand_idx].cpu().numpy()]
    axes[1].boxplot(data_tv, labels=['Non-candidates', 'Candidates'], showfliers=False)
    axes[1].set_title('Total-var')

    pc = candidates.peak_contrib
    data_pc = [pc[non_cand_idx].cpu().numpy(), pc[cand_idx].cpu().numpy()]
    axes[2].boxplot(data_pc, labels=['Non-candidates', 'Candidates'], showfliers=False)
    axes[2].set_title('Peak Contribution')

    plt.tight_layout()
    plt.savefig(static_dir / "score_distributions.png", dpi=150)
    plt.close()

    # --- 4. Per-camera visualizations (4 sample cameras) ---
    sample_indices = np.linspace(0, len(train_cameras) - 1, 4, dtype=int)
    cmap = plt.get_cmap("jet")

    for idx in sample_indices:
        cam = train_cameras[idx]
        target = cam.original_image.cuda()
        gt_mask = cam.gt_alpha_mask.cuda()
        render_pkg = render(cam, model, tile_size=args.tile_size, min_t=args.min_t)
        image = render_pkg['render']

        # Error heatmap
        err = ((target - image).abs() * gt_mask).mean(dim=0).cpu().numpy()
        err_norm = err / max(err.max(), 1e-6)
        err_rgb = (cmap(err_norm)[:, :, :3] * 255).astype(np.uint8)
        plt.figure(figsize=(10, 8))
        plt.imshow(err_rgb)
        plt.colorbar(plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, err.max())), ax=plt.gca(), label='L1 Error')
        plt.title(f'Error Heatmap - Camera {idx}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(static_dir / f"error_heatmap_{idx:02d}.png", dpi=150)
        plt.close()

        # Decision map via render_debug
        f_debug = torch.zeros((n_tets, 4), device=device)
        # Red = within (split), Blue = total-only (clone)
        within_only = candidates.within_mask & ~candidates.total_mask
        total_only = candidates.total_mask & ~candidates.within_mask
        both_mask = candidates.within_mask & candidates.total_mask

        f_debug[within_only, 0] = 1.0  # red
        f_debug[within_only, 3] = 2.0
        f_debug[total_only, 2] = 1.0   # blue
        f_debug[total_only, 3] = 2.0
        f_debug[both_mask, 0] = 1.0    # magenta
        f_debug[both_mask, 2] = 1.0
        f_debug[both_mask, 3] = 2.0

        debug_img = render_debug(f_debug, model, cam, density_multi=10, tile_size=args.tile_size)
        debug_np = (debug_img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        plt.figure(figsize=(10, 8))
        plt.imshow(debug_np)
        plt.title(f'Densify Decisions - Camera {idx}\nRed=Split, Blue=Clone, Magenta=Both')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(static_dir / f"densify_decisions_{idx:02d}.png", dpi=150)
        plt.close()

        # Vertex placement overlay
        render_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        px, py, valid = project_points_to_image(candidates.split_points, cam, device)
        px_v = px[valid].cpu().numpy()
        py_v = py[valid].cpu().numpy()

        plt.figure(figsize=(10, 8))
        plt.imshow(render_np)
        if len(px_v) > 0:
            plt.scatter(px_v, py_v, c='lime', s=1, alpha=0.5, marker='.')
        plt.title(f'Proposed New Vertices - Camera {idx} ({len(px_v)} visible)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(static_dir / f"vertex_placement_{idx:02d}.png", dpi=150)
        plt.close()

        del render_pkg, image, target
        torch.cuda.empty_cache()

    print(f"Static diagnostics saved to {static_dir}")


# ---------------------------------------------------------------------------
# Part 2: Batch Ordering Experiment
# ---------------------------------------------------------------------------

def define_ordering_strategies(candidates: CandidateInfo):
    """Define sort orderings over the candidate set.

    Returns dict mapping strategy_name -> indices into the candidate arrays
    (i.e., indices into split_points), sorted best-first.
    """
    n = candidates.clone_mask.sum().item()
    wv = candidates.within_var[candidates.clone_mask]
    tv = candidates.total_var[candidates.clone_mask]
    ssim_score = candidates.top_ssim[candidates.clone_mask].sum(dim=1)

    strategies = {}

    # 1. Current heuristic: combined
    combined = wv + tv
    strategies["current_heuristic"] = combined.argsort(descending=True)

    # 2. Within-var only
    strategies["within_var_only"] = wv.argsort(descending=True)

    # 3. Total-var only
    strategies["total_var_only"] = tv.argsort(descending=True)

    # 4. SSIM only
    strategies["ssim_only"] = ssim_score.argsort(descending=True)

    # 5. Random (fixed seed)
    gen = torch.Generator(device=wv.device)
    gen.manual_seed(42)
    strategies["random"] = torch.randperm(n, generator=gen, device=wv.device)

    return strategies


def retrain_backbone(model, tet_optim, train_cameras, num_iterations, args, device):
    """Retrain backbone only (no vertex moves) for num_iterations steps."""
    model.train()
    cam_indices = list(range(len(train_cameras)))

    for _ in range(num_iterations):
        idx = random.choice(cam_indices)
        camera = train_cameras[idx]
        target = camera.original_image.cuda()
        gt_mask = camera.gt_alpha_mask.cuda()

        ray_jitter = torch.rand(
            (camera.image_height, camera.image_width, 2), device=device)
        render_pkg = render(camera, model, ray_jitter=ray_jitter, **args.as_dict())
        image = render_pkg['render']

        l1_loss = ((target - image).abs() * gt_mask).mean()
        ssim_loss = (1 - fused_ssim(image.unsqueeze(0), target.unsqueeze(0))).clip(min=0, max=1)
        reg = tet_optim.regularizer(render_pkg, **args.as_dict())
        loss = (1 - args.lambda_ssim) * l1_loss + args.lambda_ssim * ssim_loss + reg

        loss.backward()
        tet_optim.main_step()
        tet_optim.main_zero_grad()

    model.eval()


def run_single_strategy(
    strategy_name, sort_indices, candidates, ckpt_path,
    train_cameras, args, num_batches, retrain_iters, device,
    max_eval_cameras=30,
):
    """Run batch ordering experiment for one strategy.

    Returns list of dicts with per-batch results.
    """
    print(f"\n--- Strategy: {strategy_name} ---")

    # Fresh model load
    model, _ = load_model_and_config(ckpt_path, device)
    tet_optim = TetOptimizer(model, **args.as_dict())

    # Baseline
    base_psnr, base_l1, base_ssim = compute_mean_psnr(
        model, train_cameras, tile_size=args.tile_size, min_t=args.min_t,
        max_cameras=max_eval_cameras)
    print(f"  Baseline: PSNR={base_psnr:.2f}, L1={base_l1:.4f}, SSIM={base_ssim:.4f}")

    # Split candidates into batches
    n = sort_indices.shape[0]
    batch_size = max(n // num_batches, 1)
    batches = [sort_indices[i * batch_size: min((i + 1) * batch_size, n)]
               for i in range(num_batches)]
    # Absorb remainder into last batch
    if len(batches) > num_batches:
        batches[-2] = torch.cat([batches[-2], batches[-1]])
        batches = batches[:num_batches]

    results = []
    prev_psnr = base_psnr
    cumulative_points = 0

    all_split_points = candidates.split_points

    for k, batch_idx in enumerate(batches):
        if batch_idx.numel() == 0:
            continue

        batch_points = all_split_points[batch_idx]
        num_added = batch_points.shape[0]
        cumulative_points += num_added

        # Add points
        tet_optim.split(batch_points, **args.as_dict())

        # Retrain backbone only
        retrain_backbone(model, tet_optim, train_cameras, retrain_iters, args, device)

        # Measure
        psnr_val, l1_val, ssim_val = compute_mean_psnr(
            model, train_cameras, tile_size=args.tile_size, min_t=args.min_t,
            max_cameras=max_eval_cameras)
        marginal = psnr_val - prev_psnr

        result = dict(
            batch_idx=k,
            num_points_added=num_added,
            cumulative_points=cumulative_points,
            train_psnr=psnr_val,
            train_l1=l1_val,
            train_ssim=ssim_val,
            marginal_psnr=marginal,
        )
        results.append(result)
        prev_psnr = psnr_val

        print(f"  Batch {k}: +{num_added} pts (cum={cumulative_points}), "
              f"PSNR={psnr_val:.2f} (+{marginal:.3f})")

        torch.cuda.empty_cache()

    # Cleanup
    del model, tet_optim
    gc.collect()
    torch.cuda.empty_cache()

    return {"name": strategy_name, "baseline_psnr": base_psnr, "batches": results}


def plot_ordering_results(all_results, output_dir):
    """Generate comparison plots."""
    order_dir = Path(output_dir) / "ordering"
    order_dir.mkdir(parents=True, exist_ok=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    # --- PSNR vs cumulative points ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, res in enumerate(all_results):
        cum_pts = [0] + [b['cumulative_points'] for b in res['batches']]
        psnrs = [res['baseline_psnr']] + [b['train_psnr'] for b in res['batches']]
        ax.plot(cum_pts, psnrs, color=colors[i % len(colors)],
                marker=markers[i % len(markers)], markersize=5,
                label=res['name'], linewidth=1.5)
    ax.set_xlabel('Cumulative Points Added')
    ax.set_ylabel('Training PSNR (dB)')
    ax.set_title('PSNR vs Number of Points Added')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(order_dir / "psnr_vs_points.png", dpi=150)
    plt.close()

    # --- Marginal PSNR per batch ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, res in enumerate(all_results):
        batch_idx = [b['batch_idx'] for b in res['batches']]
        marginals = [b['marginal_psnr'] for b in res['batches']]
        ax.plot(batch_idx, marginals, color=colors[i % len(colors)],
                marker=markers[i % len(markers)], markersize=5,
                label=res['name'], linewidth=1.5)
    ax.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Batch Index')
    ax.set_ylabel('Marginal PSNR Improvement (dB)')
    ax.set_title('Marginal PSNR Improvement per Batch\n(Should decrease for good ordering)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(order_dir / "marginal_psnr.png", dpi=150)
    plt.close()

    # --- Monotonicity scores ---
    lines = ["Strategy | Inversions | Total Improvement | Final PSNR"]
    lines.append("-" * 65)
    for res in all_results:
        marginals = [b['marginal_psnr'] for b in res['batches']]
        inversions = sum(1 for i in range(len(marginals) - 1) if marginals[i + 1] > marginals[i])
        total_improvement = sum(marginals)
        final_psnr = res['batches'][-1]['train_psnr'] if res['batches'] else res['baseline_psnr']
        lines.append(f"{res['name']:25s} | {inversions:10d} | {total_improvement:17.4f} | {final_psnr:.2f}")
    mono_text = "\n".join(lines)
    print("\n" + mono_text)
    with open(order_dir / "monotonicity_score.txt", "w") as f:
        f.write(mono_text)

    # Save raw results
    with open(order_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Ordering results saved to {order_dir}")


def run_ordering_experiment(
    stats, candidates, train_cameras, args,
    ckpt_path, output_dir, num_batches, retrain_iters, device,
    max_eval_cameras=30,
):
    """Run the full batch ordering experiment across all strategies."""
    strategies = define_ordering_strategies(candidates)

    all_results = []
    for name, sort_idx in strategies.items():
        result = run_single_strategy(
            name, sort_idx, candidates, ckpt_path,
            train_cameras, args, num_batches, retrain_iters, device,
            max_eval_cameras=max_eval_cameras,
        )
        all_results.append(result)

        # Save intermediate
        order_dir = Path(output_dir) / "ordering"
        order_dir.mkdir(parents=True, exist_ok=True)
        with open(order_dir / f"strategy_{name}.json", "w") as f:
            json.dump(result, f, indent=2)

    plot_ordering_results(all_results, output_dir)
    return all_results


# ---------------------------------------------------------------------------
# Part 3: Placement Quality Analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_placement_analysis(model, candidates, args, output_dir, device):
    """Analyze vertex placement quality."""
    place_dir = Path(output_dir) / "placement"
    place_dir.mkdir(parents=True, exist_ok=True)

    n = candidates.split_points.shape[0]
    if n == 0:
        print("No candidates to analyze for placement.")
        return

    ray_pts = candidates.split_points
    centroids = candidates.tet_centroids

    dist_ray_centroid = torch.norm(ray_pts - centroids, dim=1).cpu().numpy()

    bad = candidates.bad_intersections.cpu().numpy()
    good = ~candidates.bad_intersections.cpu().numpy()

    # --- 1. Distance histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(dist_ray_centroid[good], bins=80, alpha=0.7,
            label=f'Good intersections (n={good.sum()})', color='steelblue')
    if bad.sum() > 0:
        ax.hist(dist_ray_centroid[bad], bins=80, alpha=0.7,
                label=f'Bad intersections (n={bad.sum()})', color='coral')
    ax.set_xlabel('Distance: Ray-intersection to Centroid')
    ax.set_ylabel('Count')
    ax.set_title('Distance Between Placement Methods')
    ax.legend()
    plt.tight_layout()
    plt.savefig(place_dir / "placement_distances.png", dpi=150)
    plt.close()

    # --- 2. Score vs distance scatter ---
    cand_wv = candidates.within_var[candidates.clone_mask].cpu().numpy()
    cand_tv = candidates.total_var[candidates.clone_mask].cpu().numpy()
    combined_score = cand_wv + cand_tv

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter_good = ax.scatter(
        dist_ray_centroid[good], combined_score[good],
        c='steelblue', s=2, alpha=0.3, label='Good intersection')
    if bad.sum() > 0:
        scatter_bad = ax.scatter(
            dist_ray_centroid[bad], combined_score[bad],
            c='coral', s=2, alpha=0.3, label='Bad intersection')
    ax.set_xlabel('Distance: Ray-intersection to Centroid')
    ax.set_ylabel('Combined Score (within_var + total_var)')
    ax.set_title('Candidate Score vs Placement Distance')
    ax.legend()
    plt.tight_layout()
    plt.savefig(place_dir / "placement_vs_error.png", dpi=150)
    plt.close()

    # --- 3. Bad intersection analysis ---
    lines = [
        "=== Bad Intersection Analysis ===",
        f"Total candidates:      {n}",
        f"Good intersections:    {good.sum()} ({100*good.sum()/n:.1f}%)",
        f"Bad intersections:     {bad.sum()} ({100*bad.sum()/n:.1f}%)",
        "",
    ]
    if good.sum() > 0 and bad.sum() > 0:
        lines += [
            "--- Score comparison ---",
            f"Good mean combined score: {combined_score[good].mean():.6f}",
            f"Bad mean combined score:  {combined_score[bad].mean():.6f}",
            f"Good mean within_var:     {cand_wv[good].mean():.6f}",
            f"Bad mean within_var:      {cand_wv[bad].mean():.6f}",
            f"Good mean total_var:      {cand_tv[good].mean():.6f}",
            f"Bad mean total_var:       {cand_tv[bad].mean():.6f}",
            "",
            "--- Distance comparison ---",
            f"Good mean dist to centroid: {dist_ray_centroid[good].mean():.6f}",
            f"Bad mean dist to centroid:  {dist_ray_centroid[bad].mean():.6f}",
        ]
    analysis_text = "\n".join(lines)
    print(analysis_text)
    with open(place_dir / "bad_intersection_analysis.txt", "w") as f:
        f.write(analysis_text)

    print(f"Placement analysis saved to {place_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Densification diagnostics")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--dataset", type=str, default="", help="Dataset path (overrides config)")
    parser.add_argument("--image-folder", type=str, default="", help="Image folder (overrides config)")
    parser.add_argument("--output-dir", type=str, default="output/densify_diagnostics",
                        help="Output directory for diagnostics")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches for ordering experiment")
    parser.add_argument("--retrain-iters", type=int, default=200,
                        help="Backbone-only training iterations per batch")
    parser.add_argument("--num-stat-cameras", type=int, default=200,
                        help="Number of cameras for collecting densification stats")
    parser.add_argument("--max-eval-cameras", type=int, default=30,
                        help="Max cameras for PSNR evaluation (default 30, -1 for all)")
    parser.add_argument("--static-only", action="store_true", help="Only run static diagnostics")
    parser.add_argument("--ordering-only", action="store_true", help="Only run ordering experiment")
    parser.add_argument("--placement-only", action="store_true", help="Only run placement analysis")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    cli_args = parse_args()

    # Seed
    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)

    device = torch.device('cuda')
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config from checkpoint
    config = Args.load_from_json(str(Path(cli_args.ckpt) / "config.json"))

    # Override dataset path if provided
    dataset_path = cli_args.dataset if cli_args.dataset else config.dataset_path
    image_folder = cli_args.image_folder if cli_args.image_folder else config.image_folder

    # Determine what to run
    run_all = not (cli_args.static_only or cli_args.ordering_only or cli_args.placement_only)
    run_static = run_all or cli_args.static_only
    run_ordering = run_all or cli_args.ordering_only
    run_placement = run_all or cli_args.placement_only

    # Load dataset
    print(f"Loading dataset from {dataset_path} ({image_folder})...")
    train_cameras, test_cameras, scene_info = loader.load_dataset(
        Path(dataset_path), image_folder, data_device='cpu',
        eval=config.eval if hasattr(config, 'eval') else False,
        resolution=config.resolution if hasattr(config, 'resolution') else 1)
    print(f"  {len(train_cameras)} train cameras, {len(test_cameras)} test cameras")

    # Load model
    print(f"Loading checkpoint from {cli_args.ckpt}...")
    model, _ = load_model_and_config(cli_args.ckpt, device)
    print(f"  {model.vertices.shape[0]} vertices, {model.indices.shape[0]} tets")

    # Collect candidates (needed by all parts)
    print("Collecting densification statistics...")
    stats, candidates = collect_candidates(
        model, train_cameras, config, device,
        num_stat_cameras=cli_args.num_stat_cameras)
    print(f"  {candidates.clone_mask.sum().item()} candidates identified")

    # Part 1: Static diagnostics
    if run_static:
        print("\n=== Part 1: Static Diagnostics ===")
        run_static_diagnostics(model, train_cameras, stats, candidates, config, output_dir, device)

    # Part 3: Placement analysis (before ordering since it doesn't need retraining)
    if run_placement:
        print("\n=== Part 3: Placement Quality Analysis ===")
        run_placement_analysis(model, candidates, config, output_dir, device)

    # Free the initial model before ordering experiment (which loads fresh copies)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Part 2: Ordering experiment
    if run_ordering:
        print("\n=== Part 2: Batch Ordering Experiment ===")
        run_ordering_experiment(
            stats, candidates, train_cameras, config,
            cli_args.ckpt, output_dir, cli_args.num_batches,
            cli_args.retrain_iters, device,
            max_eval_cameras=cli_args.max_eval_cameras)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
