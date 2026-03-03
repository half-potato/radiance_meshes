#!/usr/bin/env python3
"""
Evaluate selection strategies for densification, including MCMC-inspired approaches.

Strategies:
  A. current          - Hard threshold: within_var > thresh OR total_var > thresh
  B. topk_score       - Top-K by combined score (within_var + total_var), no threshold
  C. multinomial      - Sample proportional to score (MCMC-style stochastic selection)
  D. mcmc_relocation  - Fixed budget: remove low-contrib vertices, add at high-error tets
  E. mcmc_noise       - Current selection + SGLD noise on vertex positions during retrain

For each strategy: select candidates -> add points -> retrain backbone -> measure PSNR.

Usage:
  python evaluate_selection_strategies.py --ckpt output/bicycle_base
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
from fused_ssim import fused_ssim

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Utilities (shared with evaluate_placement_strategies.py)
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


def compute_scores(stats, args):
    """Compute within_var and total_var scores from stats."""
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

    return within_var, total_var


def compute_split_points(stats, clone_mask, model, device):
    """Compute split points for selected tets (clamped, no random fallback)."""
    split_point_all, bad_all = get_approx_ray_intersections(stats.within_var_rays)
    split_points = split_point_all[clone_mask]
    return split_points


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def select_current(within_var, total_var, stats, args, n_budget, device):
    """Strategy A: Current hard threshold selection."""
    within_mask = within_var > args.within_thresh
    total_mask = total_var > args.total_thresh
    clone_mask = within_mask | total_mask

    if clone_mask.sum() > n_budget:
        true_indices = clone_mask.nonzero().squeeze(-1)
        perm = torch.randperm(true_indices.size(0))
        selected = true_indices[perm[:n_budget]]
        clone_mask = torch.zeros_like(clone_mask, dtype=torch.bool)
        clone_mask[selected] = True

    return clone_mask


def _compute_score(within_var, total_var, stats, args):
    """Shared score computation for B/C/D strategies."""
    wv_max = within_var.max().clamp(min=1e-8)
    tv_max = total_var.max().clamp(min=1e-8)
    score = torch.maximum(within_var / wv_max, total_var / tv_max)
    score[stats.peak_contrib < args.clone_min_contrib] = 0
    return score


def select_topk_score(within_var, total_var, stats, args, n_budget, device):
    """Strategy B: Top-K by combined score, no threshold."""
    score = _compute_score(within_var, total_var, stats, args)

    n_select = min(n_budget, (score > 0).sum().item())
    if n_select <= 0:
        return torch.zeros_like(within_var, dtype=torch.bool)

    topk_idx = torch.topk(score, n_select).indices
    clone_mask = torch.zeros_like(within_var, dtype=torch.bool)
    clone_mask[topk_idx] = True
    return clone_mask


def select_multinomial(within_var, total_var, stats, args, n_budget, device):
    """Strategy C: MCMC-style multinomial sampling proportional to score."""
    score = _compute_score(within_var, total_var, stats, args)

    temperature = 0.5
    probs = (score / temperature).softmax(dim=0)

    n_valid = (score > 0).sum().item()
    n_select = min(n_budget, n_valid)
    if n_select <= 0:
        return torch.zeros_like(within_var, dtype=torch.bool)

    selected = torch.multinomial(probs, n_select, replacement=False)
    clone_mask = torch.zeros_like(within_var, dtype=torch.bool)
    clone_mask[selected] = True
    return clone_mask


def select_mcmc_relocation(within_var, total_var, stats, model, args,
                           n_budget, device):
    """Strategy D: MCMC relocation - remove low-contrib, add at high-error.

    Returns (clone_mask, remove_mask) where remove_mask is over interior vertices.
    """
    score = _compute_score(within_var, total_var, stats, args)

    # Identify low-contribution VERTICES for removal
    n_verts = model.vertices.shape[0]
    n_interior = model.interior_vertices.shape[0]
    vertex_contrib = torch.zeros(n_verts, device=device)
    indices = model.indices.long()
    for c in range(4):
        vertex_contrib.scatter_reduce_(
            0, indices[:, c], stats.peak_contrib, reduce="amax")

    interior_contrib = vertex_contrib[:n_interior]

    # Remove bottom N vertices by contribution (cap at 10% of interior)
    n_remove = min(n_budget // 3, n_interior // 10)
    if n_remove > 0 and interior_contrib.numel() > n_remove:
        _, remove_idx = torch.topk(interior_contrib, n_remove, largest=False)
        remove_mask = torch.ones(n_interior, device=device, dtype=torch.bool)
        remove_mask[remove_idx] = False
    else:
        remove_mask = None

    # Select top tets for addition (same budget as other strategies)
    n_valid = (score > 0).sum().item()
    n_select = min(n_budget, n_valid)

    clone_mask = torch.zeros_like(within_var, dtype=torch.bool)
    if n_select > 0:
        topk_idx = torch.topk(score, n_select).indices
        clone_mask[topk_idx] = True

    return clone_mask, remove_mask


# ---------------------------------------------------------------------------
# SGLD noise for vertex positions
# ---------------------------------------------------------------------------

def apply_sgld_noise(model, peak_contrib, noise_lr=1e-4, k=100, threshold=0.5):
    """Apply SGLD noise to vertex positions, scaled by inverse contribution.

    Low-contribution vertices get more noise (explore more).
    High-contribution vertices get less noise (exploit).
    peak_contrib: per-tet contribution tensor (on CPU, moved to device as needed).
    """
    n_verts = model.vertices.shape[0]
    n_interior = model.interior_vertices.shape[0]

    n_tets = model.indices.shape[0]
    pc = peak_contrib.to(model.device)
    # Pad with zeros for new tets (unknown contribution → full noise)
    if pc.shape[0] < n_tets:
        pc = torch.cat([pc, torch.zeros(n_tets - pc.shape[0], device=model.device)])
    elif pc.shape[0] > n_tets:
        pc = pc[:n_tets]
    vertex_contrib = torch.zeros(n_verts, device=model.device)
    indices = model.indices.long()
    for c in range(4):
        vertex_contrib.scatter_reduce_(0, indices[:, c], pc, reduce="amax")

    interior_contrib = vertex_contrib[:n_interior]
    gate = torch.sigmoid(-k * (interior_contrib - threshold))

    noise = torch.randn_like(model.interior_vertices) * gate.unsqueeze(1) * noise_lr
    model.interior_vertices.data.add_(noise)


# ---------------------------------------------------------------------------
# Retrain and evaluate
# ---------------------------------------------------------------------------

def retrain_and_evaluate(
    strategy_name, split_points, ckpt_path, train_cameras, args, device,
    retrain_iters=500, eval_interval=100, max_eval_cameras=30,
    remove_mask=None, apply_noise=False, noise_lr=1e-4, peak_contrib=None,
):
    """Load fresh model, optionally remove vertices, add points, retrain, measure PSNR."""
    print(f"\n--- {strategy_name} ({split_points.shape[0]} points) ---")

    model, _ = load_model_and_config(ckpt_path, device)
    tet_optim = TetOptimizer(model, **args.as_dict())

    # Baseline PSNR
    base_psnr = compute_mean_psnr(
        model, train_cameras, tile_size=args.tile_size, min_t=args.min_t,
        max_cameras=max_eval_cameras)
    n_verts_before = model.vertices.shape[0]
    print(f"  Baseline PSNR: {base_psnr:.2f} ({n_verts_before} verts)")

    # Optionally remove low-contrib vertices first (MCMC relocation)
    if remove_mask is not None:
        n_before = model.interior_vertices.shape[0]
        tet_optim.remove_points(remove_mask.to(device))
        n_after = model.interior_vertices.shape[0]
        print(f"  Removed {n_before - n_after} vertices ({n_after} interior remain)")

    # Add new points (with tiny jitter to avoid degenerate Delaunay configs)
    if split_points.shape[0] > 0:
        pts = split_points.to(device)
        jitter = torch.randn_like(pts) * 1e-6
        tet_optim.split(pts + jitter, **args.as_dict())
        del pts, jitter
    n_verts_after = model.vertices.shape[0]
    n_tets_after = model.indices.shape[0]
    print(f"  After changes: {n_verts_after} vertices, {n_tets_after} tets")

    # PSNR right after structural changes
    psnr_after_add = compute_mean_psnr(
        model, train_cameras, tile_size=args.tile_size, min_t=args.min_t,
        max_cameras=max_eval_cameras)
    print(f"  After changes: {psnr_after_add:.2f}")

    # Retrain backbone (with optional SGLD noise)
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

        # Apply SGLD noise every 10 iterations (strategy E)
        if apply_noise and it % 10 == 0 and peak_contrib is not None:
            apply_sgld_noise(model, peak_contrib, noise_lr=noise_lr)

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
    torch.cuda.synchronize()

    return {
        "name": strategy_name,
        "baseline_psnr": base_psnr,
        "psnr_after_add": psnr_after_add,
        "psnr_curve": psnr_curve,
        "final_psnr": psnr_curve[-1][1],
        "num_points_added": int(split_points.shape[0]),
        "num_verts_before": int(n_verts_before),
        "num_verts_after": int(n_verts_after),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    # PSNR convergence
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, res in enumerate(all_results):
        iters = [p[0] for p in res['psnr_curve']]
        psnrs = [p[1] for p in res['psnr_curve']]
        ax.plot(iters, psnrs, color=colors[i % len(colors)],
                marker=markers[i % len(markers)], markersize=6, linewidth=2,
                label=f"{res['name']} (final={res['final_psnr']:.2f})")
        ax.axhline(res['baseline_psnr'], color=colors[i % len(colors)],
                   linestyle=':', alpha=0.3)
    ax.set_xlabel('Retraining Iterations')
    ax.set_ylabel('Training PSNR (dB)')
    ax.set_title('Selection Strategy Comparison:\nPSNR After Adding Candidates + Backbone Retraining')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "selection_psnr_convergence.png", dpi=150)
    plt.close()

    # Improvement bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r['name'] for r in all_results]
    baseline = all_results[0]['baseline_psnr']
    improvements = [r['final_psnr'] - baseline for r in all_results]
    bars = ax.bar(names, improvements, color=colors[:len(names)])
    ax.set_ylabel('PSNR Improvement over Baseline (dB)')
    ax.set_title('Selection Strategy: Total PSNR Improvement')
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{imp:+.3f}', ha='center', va='bottom', fontsize=10)
    ax.axhline(0, color='black', linewidth=0.5)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "selection_improvement.png", dpi=150)
    plt.close()

    # Efficiency: PSNR improvement per vertex
    fig, ax = plt.subplots(figsize=(12, 6))
    efficiency = []
    for r in all_results:
        delta = r['final_psnr'] - r['baseline_psnr']
        n_added = r['num_verts_after'] - r['num_verts_before']
        eff = delta / max(n_added, 1) * 1000  # dB per 1000 vertices
        efficiency.append(eff)
    bars = ax.bar(names, efficiency, color=colors[:len(names)])
    ax.set_ylabel('PSNR Improvement per 1000 Vertices (dB)')
    ax.set_title('Vertex Efficiency')
    for bar, eff in zip(bars, efficiency):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0001,
                f'{eff:.4f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "selection_efficiency.png", dpi=150)
    plt.close()

    # Save JSON
    json_results = []
    for r in all_results:
        rr = dict(r)
        rr['psnr_curve'] = [[it, psnr] for it, psnr in r['psnr_curve']]
        json_results.append(rr)
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_results, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate selection strategies for densification")
    parser.add_argument("--ckpt", type=str, default="output/bicycle_base")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--output-dir", type=str,
                        default="output/selection_eval")
    parser.add_argument("--retrain-iters", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--max-eval-cameras", type=int, default=30)
    parser.add_argument("--noise-lr", type=float, default=1e-4,
                        help="SGLD noise learning rate for strategy E")
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
    n_verts = model.vertices.shape[0]
    n_tets = model.indices.shape[0]
    print(f"  {n_verts} vertices, {n_tets} tets")

    # Collect stats
    print("\nCollecting densification stats...")
    num_stat_cams = min(200, len(train_cameras))
    sampler = SimpleSampler(len(train_cameras), num_stat_cams, device)
    sampled_cams = [train_cameras[i] for i in sampler.nextids()]

    model.eval()
    stats = collect_render_stats(sampled_cams, model, config, device)

    # Compute scores
    within_var, total_var = compute_scores(stats, config)
    target_addition = config.budget - n_verts
    print(f"  Target addition: {target_addition}")

    # --- Compute selections for each strategy ---
    # First, determine budget from strategy A (the baseline)
    torch.manual_seed(args.seed)
    mask_A_raw = select_current(within_var, total_var, stats, config,
                                target_addition, device)
    n_budget = mask_A_raw.sum().item()
    print(f"\nSelection budget (from strategy A threshold): {n_budget}")
    print("All strategies will select exactly this many tets.\n")
    print("Computing selections for each strategy...")
    strategies = {}

    # A: Current (hard threshold) - already computed
    pts_A = compute_split_points(stats, mask_A_raw, model, device)
    strategies["A: current"] = dict(
        clone_mask=mask_A_raw, split_points=pts_A, remove_mask=None,
        apply_noise=False)
    print(f"  A (current):     {mask_A_raw.sum().item()} selected")

    # B: Top-K by score (same budget)
    mask_B = select_topk_score(within_var, total_var, stats, config,
                               n_budget, device)
    pts_B = compute_split_points(stats, mask_B, model, device)
    strategies["B: topk_score"] = dict(
        clone_mask=mask_B, split_points=pts_B, remove_mask=None,
        apply_noise=False)
    print(f"  B (topk_score):  {mask_B.sum().item()} selected")

    # C: Multinomial sampling (same budget)
    torch.manual_seed(args.seed)
    mask_C = select_multinomial(within_var, total_var, stats, config,
                                n_budget, device)
    pts_C = compute_split_points(stats, mask_C, model, device)
    strategies["C: multinomial"] = dict(
        clone_mask=mask_C, split_points=pts_C, remove_mask=None,
        apply_noise=False)
    print(f"  C (multinomial): {mask_C.sum().item()} selected")

    # D: MCMC relocation (same addition budget + remove low-contrib)
    mask_D, remove_mask_D = select_mcmc_relocation(
        within_var, total_var, stats, model, config,
        n_budget, device)
    pts_D = compute_split_points(stats, mask_D, model, device)
    strategies["D: mcmc_reloc"] = dict(
        clone_mask=mask_D, split_points=pts_D, remove_mask=remove_mask_D,
        apply_noise=False)
    n_remove = (~remove_mask_D).sum().item() if remove_mask_D is not None else 0
    print(f"  D (mcmc_reloc):  {mask_D.sum().item()} add, {n_remove} remove")

    # E: Current selection + SGLD noise during retraining (same selection as A)
    strategies["E: sgld_noise"] = dict(
        clone_mask=mask_A_raw, split_points=pts_A.clone(), remove_mask=None,
        apply_noise=True)
    print(f"  E (sgld_noise):  {mask_A_raw.sum().item()} selected (same as A + noise)")

    # --- Overlap analysis ---
    print("\n--- Selection overlap ---")
    for name_i, si in strategies.items():
        for name_j, sj in strategies.items():
            if name_i >= name_j:
                continue
            overlap = (si['clone_mask'] & sj['clone_mask']).sum().item()
            union = (si['clone_mask'] | sj['clone_mask']).sum().item()
            iou = overlap / max(union, 1)
            print(f"  {name_i} vs {name_j}: "
                  f"overlap={overlap}, IoU={iou:.3f}")

    # Save peak_contrib for SGLD noise (strategy E) before freeing stats
    peak_contrib_cpu = stats.peak_contrib.cpu()

    # Move split points to CPU to free GPU memory during sequential evaluation
    for name, info in strategies.items():
        info['split_points'] = info['split_points'].cpu()
        if info['remove_mask'] is not None:
            info['remove_mask'] = info['remove_mask'].cpu()

    # Free model and stats before retraining
    del model, stats, within_var, total_var
    del mask_A_raw, mask_B, mask_C, mask_D
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # --- Evaluate each strategy ---
    print(f"\n{'='*60}")
    print("Starting selection strategy evaluation...")
    print(f"  Retrain iterations: {args.retrain_iters}")
    print(f"  Eval interval:      {args.eval_interval}")
    print(f"  Max eval cameras:   {args.max_eval_cameras}")
    print(f"{'='*60}")

    all_results = []

    for name, info in strategies.items():
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        result = retrain_and_evaluate(
            name, info['split_points'], args.ckpt, train_cameras, config,
            device,
            retrain_iters=args.retrain_iters,
            eval_interval=args.eval_interval,
            max_eval_cameras=args.max_eval_cameras,
            remove_mask=info['remove_mask'],
            apply_noise=info['apply_noise'],
            noise_lr=args.noise_lr,
            peak_contrib=peak_contrib_cpu if info['apply_noise'] else None)
        all_results.append(result)

        # Save intermediate
        with open(output_dir / f"result_{name.split(': ')[1]}.json", "w") as f:
            rr = dict(result)
            rr['psnr_curve'] = [[it, p] for it, p in result['psnr_curve']]
            json.dump(rr, f, indent=2)

    # --- Summary ---
    print(f"\n{'='*95}")
    print(f"{'Strategy':20s} | {'Baseline':>8s} | {'After':>8s} | "
          f"{'Final':>8s} | {'Delta':>8s} | "
          f"{'#Added':>7s} | {'#Final V':>8s} | {'dB/1kV':>7s}")
    print("-" * 95)
    for r in all_results:
        delta = r['final_psnr'] - r['baseline_psnr']
        n_added = r['num_verts_after'] - r['num_verts_before']
        eff = delta / max(n_added, 1) * 1000
        print(f"{r['name']:20s} | {r['baseline_psnr']:8.2f} | "
              f"{r['psnr_after_add']:8.2f} | {r['final_psnr']:8.2f} | "
              f"{delta:+8.3f} | {n_added:7d} | {r['num_verts_after']:8d} | "
              f"{eff:7.4f}")
    print(f"{'='*95}")

    # Plot
    plot_results(all_results, output_dir)
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
