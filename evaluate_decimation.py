#!/usr/bin/env python3
"""
Greedy decimation benchmark: compare edge-collapse ordering heuristics.

For each heuristic:
  1. Rank ALL interior edges by heuristic value (ascending = least important first)
  2. Collapse edges in large batches (e.g., 5000 per step)
  3. After each batch: re-triangulate, render test cameras, measure global PSNR
  4. Plot PSNR-vs-edges-removed curves

Usage:
  uv run evaluate_decimation.py --ckpt output/bicycle_exp1_densify
"""

import gc
import json
import math
import random
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import loader
from models.ingp_color import Model
from utils.args import Args
from utils.train_util import render
from utils.topo_utils import (
    calculate_circumcenters_torch, tet_volumes,
)
from utils.contraction import contract_mean_std

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_config(ckpt_path, device):
    config = Args.load_from_json(str(Path(ckpt_path) / "config.json"))
    model = Model.load_ckpt(Path(ckpt_path), device)
    return model, config


@torch.no_grad()
def compute_global_psnr(model, cameras, config, gt_images, device):
    """Render all test cameras and compute mean PSNR."""
    model.eval()
    psnrs = []
    for i, cam in enumerate(cameras):
        render_pkg = render(cam, model, tile_size=config.tile_size,
                            min_t=config.min_t)
        pred = render_pkg['render']  # [3, H, W] on GPU
        gt = gt_images[i].to(device)
        mse = ((pred - gt) ** 2).mean().item()
        if mse < 1e-10:
            psnrs.append(50.0)
        else:
            psnrs.append(-10 * math.log10(mse))
        del render_pkg, pred, gt
    torch.cuda.empty_cache()
    return np.mean(psnrs)


# ---------------------------------------------------------------------------
# Edge list + topology
# ---------------------------------------------------------------------------

def build_edge_list(indices):
    """Extract unique edges from tet indices. Returns (E, 2) int64 tensor."""
    idx = indices.long()
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    all_edges = []
    for i, j in pairs:
        all_edges.append(torch.stack([idx[:, i], idx[:, j]], dim=1))
    all_edges = torch.cat(all_edges, dim=0)
    all_edges, _ = all_edges.sort(dim=1)
    return torch.unique(all_edges, dim=0)


def build_vertex_tet_adjacency(indices, n_verts):
    """Per-vertex list of tet indices."""
    adjacency = [[] for _ in range(n_verts)]
    indices_np = indices.cpu().numpy()
    for t in range(indices_np.shape[0]):
        for c in range(4):
            adjacency[indices_np[t, c]].append(t)
    return adjacency


# ---------------------------------------------------------------------------
# Per-edge heuristic scoring (vectorized where possible)
# ---------------------------------------------------------------------------

@torch.no_grad()
def query_field(model, tet_verts):
    """Query backbone at circumcenters. Returns (density, rgb, cc, radii)."""
    if tet_verts.shape[0] == 0:
        d = model.device
        return (torch.zeros(0, device=d), torch.zeros(0, 3, device=d),
                torch.zeros(0, 3, device=d), torch.zeros(0, device=d))
    cc, radius = calculate_circumcenters_torch(tet_verts.double())
    cc, radius = cc.float(), radius.float()
    normalized = (cc.detach() - model.center) / model.scene_scaling
    cv, cr = contract_mean_std(normalized, radius / model.scene_scaling)
    x = (cv / 2 + 1) / 2
    density, rgb, grd, sh, attr = model.backbone(x, cr)
    return density.squeeze(-1), rgb, cc, radius


@torch.no_grad()
def compute_edge_heuristics(edges, adjacency, vertices, indices,
                             tet_density, tet_rgb, tet_cc, tet_vol,
                             device):
    """Compute heuristic scores for all edges using edge star (intersection).

    The edge star = tets containing BOTH va AND vb = the ring of tets that
    actually degenerate during edge collapse.

    Returns dict: heuristic_name -> np.array of shape (n_edges,).
    Lower value = less important = collapse first.
    """
    n_edges = edges.shape[0]
    edges_np = edges.cpu().numpy()

    # Initialize heuristic arrays
    h = {
        # Baselines
        "edge_length": np.zeros(n_edges),
        "random": np.random.rand(n_edges),
        # Edge-star field statistics (ascending = uniform/stable first)
        "estar_density_std": np.zeros(n_edges),
        "estar_rgb_std": np.zeros(n_edges),
        "estar_density_range": np.zeros(n_edges),
        "estar_rgb_range": np.zeros(n_edges),
        "estar_cc_shift": np.zeros(n_edges),
        # Legacy union-based (kept for comparison)
        "inv_rgb_std": np.zeros(n_edges),
    }

    # Edge lengths (vectorized)
    va_pos = vertices[edges[:, 0]]
    vb_pos = vertices[edges[:, 1]]
    edge_lengths = torch.linalg.norm(va_pos - vb_pos, dim=1).cpu().numpy()
    h["edge_length"] = edge_lengths

    # Per-edge: edge star heuristics
    for ei in tqdm(range(n_edges), desc="Computing edge-star heuristics",
                   mininterval=2.0):
        va, vb = edges_np[ei]
        # Edge star = intersection (tets containing BOTH endpoints)
        edge_star = sorted(set(adjacency[va]) & set(adjacency[vb]))

        if not edge_star:
            continue

        star_t = torch.tensor(edge_star, dtype=torch.long, device=device)
        n_star = len(edge_star)

        # --- Density stats on edge star ---
        star_d = tet_density[star_t]
        h["estar_density_range"][ei] = (star_d.max() - star_d.min()).item()
        if n_star > 1:
            h["estar_density_std"][ei] = star_d.std().item()

        # --- RGB stats on edge star ---
        star_rgb = tet_rgb[star_t]  # (n_star, 3)
        if n_star > 1:
            h["estar_rgb_std"][ei] = star_rgb.std(dim=0).mean().item()
            rgb_range = (star_rgb.max(dim=0).values -
                         star_rgb.min(dim=0).values).mean().item()
            h["estar_rgb_range"][ei] = rgb_range

        # --- Circumcenter shift ---
        # Replace va/vb with midpoint in each tet, recompute cc, measure shift
        midpoint = (vertices[va] + vertices[vb]) / 2.0
        tet_idx = indices[star_t].long()  # (n_star, 4)
        tet_verts_new = vertices[tet_idx].clone()  # (n_star, 4, 3)
        for c in range(4):
            mask_a = (tet_idx[:, c] == va)
            mask_b = (tet_idx[:, c] == vb)
            if mask_a.any():
                tet_verts_new[mask_a, c] = midpoint
            if mask_b.any():
                tet_verts_new[mask_b, c] = midpoint
        cc_new, _ = calculate_circumcenters_torch(tet_verts_new.double())
        cc_new = cc_new.float()
        shifts = torch.linalg.norm(cc_new - tet_cc[star_t], dim=1)
        h["estar_cc_shift"][ei] = shifts.mean().item()

        # --- Legacy union-based rgb_std (for comparison) ---
        union_star = sorted(set(adjacency[va]) | set(adjacency[vb]))
        if len(union_star) > 1:
            union_t = torch.tensor(union_star, dtype=torch.long, device=device)
            h["inv_rgb_std"][ei] = -tet_rgb[union_t].std(dim=0).mean().item()

    # ---------------------------------------------------------------
    # Ratio scores: edge_length / (field_measure + eps)
    # Short edges in uniform regions → low score → collapse first
    # ---------------------------------------------------------------
    eps_field = 1e-3
    eps_cc = 1e-4
    h["len_over_rgb_std"] = edge_lengths / (h["estar_rgb_std"] + eps_field)
    h["len_over_density_std"] = edge_lengths / (h["estar_density_std"] + eps_field)
    h["len_over_density_range"] = edge_lengths / (h["estar_density_range"] + eps_field)
    h["len_over_cc_shift"] = edge_lengths / (h["estar_cc_shift"] + eps_cc)

    # ---------------------------------------------------------------
    # Combined heuristics (rank-based)
    # ---------------------------------------------------------------
    def to_ranks(arr):
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(arr)) / max(1, len(arr) - 1)
        return ranks

    # Ranks for edge-star heuristics (ascending = collapse first)
    r_len = to_ranks(h["edge_length"])
    r_estar_rgb = to_ranks(h["estar_rgb_std"])
    r_estar_dens = to_ranks(h["estar_density_std"])
    r_estar_cc = to_ranks(h["estar_cc_shift"])
    # Legacy
    r_irgb = to_ranks(h["inv_rgb_std"])

    # Rank-based combinations
    h["rank_len+ergb"] = (r_len + r_estar_rgb) / 2
    h["rank_len+edens"] = (r_len + r_estar_dens) / 2
    h["rank_len+ergb+edens"] = (r_len + r_estar_rgb + r_estar_dens) / 3
    h["rank_3len+ergb+edens"] = (3*r_len + r_estar_rgb + r_estar_dens) / 5
    h["rank_len+eccshift"] = (r_len + r_estar_cc) / 2

    # Legacy combination (for comparison with previous runs)
    r_idgm = to_ranks(-np.abs(h["estar_density_std"]))  # approximate inv density grad
    h["rank_3len+irgb+idgmax"] = (3*r_len + r_irgb + r_idgm) / 5

    return h


# ---------------------------------------------------------------------------
# Edge collapse on model
# ---------------------------------------------------------------------------

@torch.no_grad()
def collapse_edges_inplace(model, edge_va, edge_vb, device):
    """Collapse edges by moving va to midpoint and removing vb.

    edge_va, edge_vb: 1D tensors of vertex indices to collapse.
    Modifies model in-place.
    """
    verts_data = model.interior_vertices.data
    n_int = verts_data.shape[0]

    # Move va to midpoint
    midpoints = (verts_data[edge_va] + verts_data[edge_vb]) / 2
    verts_data[edge_va] = midpoints

    # Build keep mask (remove vb vertices)
    remove_set = set(edge_vb.cpu().tolist())
    keep_mask = torch.ones(n_int, dtype=torch.bool, device=device)
    for idx in remove_set:
        if idx < n_int:
            keep_mask[idx] = False

    kept_verts = verts_data[keep_mask]
    model.interior_vertices = nn.Parameter(kept_verts)
    model.update_triangulation()

    return keep_mask.sum().item()


# ---------------------------------------------------------------------------
# Greedy decimation for one heuristic
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_decimation(heuristic_name, edge_scores, edges, n_interior,
                   ckpt_path, cameras, gt_images, config, device,
                   batch_size=5000, max_steps=10):
    """Collapse edges in order of ascending score, measuring PSNR at each step.

    Returns list of (n_removed, n_remaining_verts, psnr) tuples.
    """
    print(f"\n--- Heuristic: {heuristic_name} ---")

    # Filter to interior-interior edges
    mask = (edges[:, 0] < n_interior) & (edges[:, 1] < n_interior)
    int_edges = edges[mask].cpu().numpy()
    int_scores = edge_scores[mask.cpu().numpy()]

    # Sort by ascending score (least important first)
    order = np.argsort(int_scores)
    sorted_edges = int_edges[order]

    # Load fresh model
    model, _ = load_model_and_config(ckpt_path, device)
    model.eval()

    # Baseline PSNR
    baseline_psnr = compute_global_psnr(model, cameras, config, gt_images, device)
    n_verts_start = model.interior_vertices.shape[0]
    print(f"  Baseline: {n_verts_start} verts, PSNR={baseline_psnr:.4f}")

    curve = [(0, n_verts_start, baseline_psnr)]
    total_removed = 0
    edge_ptr = 0

    for step in range(max_steps):
        if edge_ptr >= len(sorted_edges):
            print(f"  Step {step+1}: no more edges to collapse")
            break

        # Get next batch of edges
        batch_end = min(edge_ptr + batch_size, len(sorted_edges))
        batch_edges = sorted_edges[edge_ptr:batch_end]

        # Resolve conflicts: if a vertex appears as both va and vb
        # in different edges, skip the later edge. Also skip edges
        # where either endpoint was already removed.
        current_n = model.interior_vertices.shape[0]
        used_verts = set()
        valid_va = []
        valid_vb = []

        for va, vb in batch_edges:
            if va >= current_n or vb >= current_n:
                continue
            if va in used_verts or vb in used_verts:
                continue
            if va == vb:
                continue
            used_verts.add(va)
            used_verts.add(vb)
            valid_va.append(va)
            valid_vb.append(vb)

        if not valid_va:
            edge_ptr = batch_end
            continue

        va_t = torch.tensor(valid_va, dtype=torch.long, device=device)
        vb_t = torch.tensor(valid_vb, dtype=torch.long, device=device)

        t0 = time.time()
        n_remaining = collapse_edges_inplace(model, va_t, vb_t, device)
        total_removed += len(valid_va)
        edge_ptr = batch_end

        # Measure PSNR
        psnr = compute_global_psnr(model, cameras, config, gt_images, device)
        elapsed = time.time() - t0

        curve.append((total_removed, n_remaining, psnr))
        print(f"  Step {step+1}: collapsed {len(valid_va)} edges, "
              f"{n_remaining} verts remaining, PSNR={psnr:.4f} "
              f"(delta={psnr - baseline_psnr:+.4f}) [{elapsed:.1f}s]")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return curve


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_decimation_curves(all_curves, output_path):
    """Plot PSNR vs edges removed for all heuristics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_curves)))

    for i, (name, curve) in enumerate(all_curves.items()):
        removed = [c[0] for c in curve]
        psnrs = [c[2] for c in curve]
        ax1.plot(removed, psnrs, 'o-', color=colors[i], label=name, markersize=4)

        # Also plot vs remaining verts
        remaining = [c[1] for c in curve]
        ax2.plot(remaining, psnrs, 'o-', color=colors[i], label=name, markersize=4)

    ax1.set_xlabel('Edges Collapsed')
    ax1.set_ylabel('Mean Test PSNR (dB)')
    ax1.set_title('PSNR vs Edges Removed')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Vertices Remaining')
    ax2.set_ylabel('Mean Test PSNR (dB)')
    ax2.set_title('PSNR vs Remaining Vertices')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {output_path}")


def plot_auc_comparison(all_curves, output_path):
    """Bar chart of area-under-curve (higher = better preservation)."""
    aucs = {}
    for name, curve in all_curves.items():
        removed = np.array([c[0] for c in curve])
        psnrs = np.array([c[2] for c in curve])
        if len(removed) > 1:
            aucs[name] = np.trapz(psnrs, removed)
        else:
            aucs[name] = 0.0

    # Sort by AUC descending (best first)
    sorted_items = sorted(aucs.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    vals = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
    ax.barh(range(len(names)), vals)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('AUC (PSNR × edges_removed)')
    ax.set_title('Decimation Quality: Area Under PSNR Curve (higher = better)')
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f'{v:.0f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Greedy decimation benchmark for edge heuristics")
    parser.add_argument("--ckpt", type=str,
                        default="output/bicycle_exp1_densify")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--output-dir", type=str,
                        default="output/decimation_benchmark")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Edges to collapse per step")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Max decimation steps per heuristic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated list of heuristics to run "
                             "(empty = all)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config & dataset
    config = Args.load_from_json(str(Path(args.ckpt) / "config.json"))
    dataset_path = args.dataset if args.dataset else config.dataset_path
    print(f"Loading dataset from {dataset_path}...")
    train_cameras, test_cameras, _ = loader.load_dataset(
        Path(dataset_path), config.image_folder, data_device='cpu',
        eval=config.eval, resolution=config.resolution)
    print(f"  {len(train_cameras)} train, {len(test_cameras)} test cameras")

    # Ground truth images
    gt_images = [cam.original_image.cpu() for cam in test_cameras]

    # ===================================================================
    # Step 1: Compute all heuristics on the original model
    # ===================================================================
    print(f"\n{'='*60}")
    print("Step 1: Computing edge heuristics")
    print(f"{'='*60}")

    model, _ = load_model_and_config(args.ckpt, device)
    model.eval()
    vertices = model.vertices.detach()
    indices = model.indices.detach()
    n_verts = vertices.shape[0]
    n_interior = model.interior_vertices.shape[0]
    n_tets = indices.shape[0]
    print(f"  {n_verts} vertices ({n_interior} interior), {n_tets} tets")

    # Field values
    print("  Evaluating field at circumcenters...")
    all_tet_verts = vertices[indices]
    chunk = 50000
    all_d, all_rgb, all_cc, all_r = [], [], [], []
    for s in range(0, n_tets, chunk):
        e = min(s + chunk, n_tets)
        d, rgb, cc, r = query_field(model, all_tet_verts[s:e])
        all_d.append(d); all_rgb.append(rgb)
        all_cc.append(cc); all_r.append(r)
    tet_density = torch.cat(all_d)
    tet_rgb = torch.cat(all_rgb)
    tet_cc = torch.cat(all_cc)
    tet_radii = torch.cat(all_r)
    tet_vol = tet_volumes(all_tet_verts).abs()

    # Build topology
    print("  Building topology...")
    adjacency = build_vertex_tet_adjacency(indices, n_verts)
    edges = build_edge_list(indices)
    print(f"  {edges.shape[0]} unique edges")

    # Compute heuristics
    heuristics = compute_edge_heuristics(
        edges, adjacency, vertices, indices,
        tet_density, tet_rgb, tet_cc, tet_vol,
        device)

    # Free memory
    del model, tet_density, tet_rgb, tet_cc, tet_radii, tet_vol
    del all_tet_verts, adjacency
    gc.collect()
    torch.cuda.empty_cache()

    # Save heuristics
    print(f"\n  Heuristic stats:")
    for name, scores in heuristics.items():
        print(f"    {name:25s}: mean={np.mean(scores):.6f}, "
              f"std={np.std(scores):.6f}")

    # ===================================================================
    # Step 2: Run decimation for each heuristic
    # ===================================================================
    print(f"\n{'='*60}")
    print("Step 2: Greedy decimation benchmark")
    print(f"{'='*60}")

    all_curves = {}

    # Load previous results if they exist (to merge with new runs)
    results_path = output_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            prev = json.load(f)
        all_curves = {k: [tuple(x) for x in v]
                      for k, v in prev.get("curves", {}).items()}
        print(f"  Loaded {len(all_curves)} previous curves")

    # Filter heuristics to run
    if args.only:
        run_names = [s.strip() for s in args.only.split(",")]
    else:
        run_names = list(heuristics.keys())

    for heuristic_name in run_names:
        if heuristic_name not in heuristics:
            print(f"\n  WARNING: unknown heuristic '{heuristic_name}', skipping")
            continue
        scores = heuristics[heuristic_name]
        curve = run_decimation(
            heuristic_name, scores, edges, n_interior,
            args.ckpt, test_cameras, gt_images, config, device,
            batch_size=args.batch_size, max_steps=args.max_steps)
        all_curves[heuristic_name] = curve

        # Save incrementally
        results = {
            "batch_size": args.batch_size,
            "curves": {k: [(int(r), int(v), float(p)) for r, v, p in c]
                       for k, c in all_curves.items()},
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # ===================================================================
    # Step 3: Plot results
    # ===================================================================
    print(f"\n{'='*60}")
    print("Step 3: Results")
    print(f"{'='*60}")

    plot_decimation_curves(all_curves, output_dir / "decimation_curves.png")
    plot_auc_comparison(all_curves, output_dir / "auc_comparison.png")

    # Summary table
    print(f"\n{'Heuristic':25s} | {'Final PSNR':>10s} | {'PSNR Drop':>10s}")
    print("-" * 50)
    for name, curve in sorted(all_curves.items(),
                               key=lambda x: x[1][-1][2], reverse=True):
        final_psnr = curve[-1][2]
        baseline_psnr = curve[0][2]
        drop = final_psnr - baseline_psnr
        print(f"{name:25s} | {final_psnr:10.4f} | {drop:+10.4f}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
