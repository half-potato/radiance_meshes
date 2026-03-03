#!/usr/bin/env python3
"""
Investigate edge collapse importance via local PSNR measurement.

For length-controlled, spatially-spread edges:
  1. Batch-collapse edges (move endpoints to midpoint, re-triangulate)
  2. Render full images before/after
  3. Measure local PSNR delta in a window around each projected edge midpoint
  4. Correlate with cheap heuristics

Usage:
  python investigate_vertex_importance.py --ckpt output/bicycle_exp1_densify
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
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm

from data import loader
from models.ingp_color import Model
from utils.args import Args
from utils.train_util import render, SimpleSampler
from utils.densification import collect_render_stats
from utils.topo_utils import (
    calculate_circumcenters_torch, tet_volumes, build_tv_struct,
    max_density_contrast
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


def project_point(point_3d, camera, device):
    """Project a 3D world point to pixel coordinates.

    Returns (px, py, depth) or None if behind camera.
    """
    wvt = camera.world_view_transform.to(device)  # (4, 4), column-major
    p_hom = torch.cat([point_3d.to(device), torch.ones(1, device=device)])
    p_cam = wvt @ p_hom  # camera space
    depth = p_cam[2]
    if depth <= 0.01:
        return None

    cx = camera.cx if camera.cx != -1 else camera.image_width / 2
    cy = camera.cy if camera.cy != -1 else camera.image_height / 2
    px = camera.fx * p_cam[0] / depth + cx
    py = camera.fy * p_cam[1] / depth + cy
    return float(px), float(py), float(depth)


def get_local_window(px, py, window_size, img_h, img_w):
    """Get pixel bounds for a window centered at (px, py)."""
    half = window_size // 2
    x_min = max(0, int(px) - half)
    y_min = max(0, int(py) - half)
    x_max = min(img_w, x_min + window_size)
    y_max = min(img_h, y_min + window_size)
    # Adjust if window was clamped at boundary
    if x_max - x_min < window_size // 2 or y_max - y_min < window_size // 2:
        return None  # Too close to edge
    return y_min, y_max, x_min, x_max


def local_psnr(pred, gt):
    """Compute PSNR between two image crops [C, H, W]."""
    mse = ((pred - gt) ** 2).mean().item()
    if mse < 1e-10:
        return 50.0
    return -10 * math.log10(mse)


# ---------------------------------------------------------------------------
# Edge sampling
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


def count_edge_visibility(edges, vertices, cameras, device, min_views=3):
    """Count how many test cameras each edge midpoint is visible in.

    Returns visibility counts per edge (len = edges.shape[0]).
    """
    n = edges.shape[0]
    vis_counts = np.zeros(n, dtype=np.int32)
    midpoints = (vertices[edges[:, 0]] + vertices[edges[:, 1]]) / 2

    for cam in cameras:
        wvt = cam.world_view_transform.to(device)
        cx = cam.cx if cam.cx != -1 else cam.image_width / 2
        cy = cam.cy if cam.cy != -1 else cam.image_height / 2

        # Batch project all midpoints
        p_hom = torch.cat([midpoints.to(device),
                           torch.ones(n, 1, device=device)], dim=1)  # (N, 4)
        p_cam = (wvt @ p_hom.T).T  # (N, 4)
        depth = p_cam[:, 2]
        px = cam.fx * p_cam[:, 0] / depth + cx
        py = cam.fy * p_cam[:, 1] / depth + cy

        # Check visibility: in front of camera and within image bounds (with margin)
        margin = 40  # pixels margin from edge
        visible = ((depth > 0.01) &
                   (px >= margin) & (px < cam.image_width - margin) &
                   (py >= margin) & (py < cam.image_height - margin))
        vis_counts += visible.cpu().numpy().astype(np.int32)

    return vis_counts


def sample_length_controlled_edges(edges, vertices, n_interior, n_target,
                                    cameras, device, length_quantile=0.5,
                                    length_band_width=0.1, min_views=3):
    """Sample edges from a narrow length band, spatially spread out.

    Pre-filters to edges visible in >= min_views test cameras.
    """
    # Filter to interior-interior
    mask = (edges[:, 0] < n_interior) & (edges[:, 1] < n_interior)
    interior_edges = edges[mask]

    if interior_edges.shape[0] == 0:
        return []

    # Compute lengths
    va_pos = vertices[interior_edges[:, 0]]
    vb_pos = vertices[interior_edges[:, 1]]
    lengths = torch.linalg.norm(va_pos - vb_pos, dim=1)

    # Select edges in the median length band
    lo_q = max(0, length_quantile - length_band_width / 2)
    hi_q = min(1, length_quantile + length_band_width / 2)
    lo_len = torch.quantile(lengths, lo_q).item()
    hi_len = torch.quantile(lengths, hi_q).item()
    band_mask = (lengths >= lo_len) & (lengths <= hi_len)
    band_edges = interior_edges[band_mask]
    band_lengths = lengths[band_mask]

    if band_edges.shape[0] == 0:
        return []

    print(f"  Length band: [{lo_len:.6f}, {hi_len:.6f}], {band_edges.shape[0]} edges")

    # Pre-filter by visibility
    print(f"  Computing visibility for {band_edges.shape[0]} band edges...")
    vis_counts = count_edge_visibility(band_edges, vertices, cameras, device)
    vis_mask = vis_counts >= min_views
    n_visible = vis_mask.sum()
    print(f"  {n_visible}/{band_edges.shape[0]} edges visible in >= {min_views} views")

    if n_visible == 0:
        return []

    band_edges = band_edges[vis_mask]
    band_lengths = band_lengths[vis_mask]
    vis_counts = vis_counts[vis_mask]

    # Farthest-point sampling on midpoints for spatial spread
    midpoints = (vertices[band_edges[:, 0]] + vertices[band_edges[:, 1]]) / 2
    midpoints_np = midpoints.cpu().numpy()

    n_select = min(n_target, band_edges.shape[0])
    selected = [random.randint(0, len(midpoints_np) - 1)]
    min_dists = np.full(len(midpoints_np), np.inf)

    for _ in range(n_select - 1):
        last = midpoints_np[selected[-1]]
        dists = np.linalg.norm(midpoints_np - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1  # exclude already selected
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)

    result = []
    for idx in selected:
        va, vb = band_edges[idx].tolist()
        result.append((va, vb, band_lengths[idx].item()))
    return result


def assign_batches(sampled_edges, vertices, batch_size, device):
    """Group edges into batches where edges within a batch are far apart."""
    # Already spatially spread from FPS, so simple sequential batching works
    batches = []
    for i in range(0, len(sampled_edges), batch_size):
        batches.append(sampled_edges[i:i+batch_size])
    return batches


# ---------------------------------------------------------------------------
# Edge collapse + rendering
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_all_cameras(model, cameras, config, device):
    """Render all cameras, return list of [3, H, W] tensors on CPU."""
    model.eval()
    images = []
    for cam in cameras:
        render_pkg = render(cam, model, tile_size=config.tile_size,
                            min_t=config.min_t)
        images.append(render_pkg['render'].cpu())
        del render_pkg
        torch.cuda.empty_cache()
    return images


@torch.no_grad()
def collapse_edges_and_render(model, edge_batch, cameras, config, device):
    """Collapse a batch of edges in the model and render all cameras.

    Modifies model in-place. Caller must reload model afterward.

    For each edge (va, vb): move va to midpoint, mark vb for removal.
    Then remove all vb vertices at once + re-triangulate.
    """
    # Sort by descending index so removals don't shift earlier indices
    # We need to be careful: after removing higher-index vertices,
    # the lower indices are still valid.
    edges_sorted = sorted(edge_batch, key=lambda e: -e[1])

    verts_data = model.interior_vertices.data
    remove_indices = set()

    for va, vb, length in edges_sorted:
        midpoint = (verts_data[va] + verts_data[vb]) / 2
        verts_data[va] = midpoint
        remove_indices.add(vb)

    # Build keep mask
    n_int = model.interior_vertices.shape[0]
    keep_mask = torch.ones(n_int, dtype=torch.bool, device=device)
    for idx in remove_indices:
        keep_mask[idx] = False

    # Manually shrink interior_vertices and re-triangulate
    kept_verts = verts_data[keep_mask]
    model.interior_vertices = nn.Parameter(kept_verts)
    model.update_triangulation()

    # Render
    images = render_all_cameras(model, cameras, config, device)
    return images


@torch.no_grad()
def measure_batch_psnr_deltas(baseline_images, collapsed_images, gt_images,
                               edge_batch, midpoints_3d, cameras,
                               window_size, device):
    """Measure per-edge local PSNR delta across all cameras.

    Returns dict: edge_idx -> mean_psnr_delta (negative = quality loss)
    """
    results = {}

    for edge_idx, (va, vb, length) in enumerate(edge_batch):
        mid = midpoints_3d[edge_idx]
        psnr_deltas = []

        for cam_idx, cam in enumerate(cameras):
            proj = project_point(mid, cam, device)
            if proj is None:
                continue
            px, py, depth = proj

            window = get_local_window(
                px, py, window_size, cam.image_height, cam.image_width)
            if window is None:
                continue
            y_min, y_max, x_min, x_max = window

            gt_crop = gt_images[cam_idx][:, y_min:y_max, x_min:x_max]
            base_crop = baseline_images[cam_idx][:, y_min:y_max, x_min:x_max]
            coll_crop = collapsed_images[cam_idx][:, y_min:y_max, x_min:x_max]

            psnr_before = local_psnr(base_crop, gt_crop)
            psnr_after = local_psnr(coll_crop, gt_crop)
            psnr_deltas.append(psnr_after - psnr_before)

        results[edge_idx] = {
            'mean_psnr_delta': np.mean(psnr_deltas) if psnr_deltas else 0.0,
            'n_views': len(psnr_deltas),
            'psnr_deltas': psnr_deltas,
        }

    return results


# ---------------------------------------------------------------------------
# Heuristics
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
def compute_all_heuristics(sampled_edges, adjacency, model, stats,
                            vertices, indices, tet_density, tet_rgb,
                            tet_cc, tet_radii, tet_vol,
                            tet_neighbor_pairs, cameras, device):
    """Compute all heuristics for sampled edges."""
    # Pre-compute density contrast
    per_tet_contrast = max_density_contrast(
        vertices, indices.long(), tet_density, mode="diff")

    # Pair-wise gradients
    pair_cc_dist = torch.linalg.norm(
        tet_cc[tet_neighbor_pairs[:, 0]] - tet_cc[tet_neighbor_pairs[:, 1]],
        dim=1).clamp(min=1e-8)
    pair_density_diff = (tet_density[tet_neighbor_pairs[:, 0]] -
                         tet_density[tet_neighbor_pairs[:, 1]]).abs()
    pair_rgb_diff = torch.linalg.norm(
        tet_rgb[tet_neighbor_pairs[:, 0]] - tet_rgb[tet_neighbor_pairs[:, 1]],
        dim=1)
    pair_density_grad = pair_density_diff / pair_cc_dist
    pair_rgb_grad = pair_rgb_diff / pair_cc_dist

    # tet -> pair index
    tet_to_pairs = defaultdict(list)
    pairs_np = tet_neighbor_pairs.cpu().numpy()
    for pi in range(pairs_np.shape[0]):
        tet_to_pairs[pairs_np[pi, 0]].append(pi)
        tet_to_pairs[pairs_np[pi, 1]].append(pi)

    heuristics = defaultdict(list)

    for va, vb, length in tqdm(sampled_edges, desc="Computing heuristics"):
        star_a = set(adjacency[va])
        star_b = set(adjacency[vb])
        star = sorted(star_a | star_b)

        if not star:
            for k in ["edge_length", "max_peak_contrib", "mean_peak_contrib",
                       "density_max", "density_range", "density_std",
                       "rgb_std", "density_contrast", "circumcenter_shift",
                       "star_volume", "valence",
                       "density_grad_max", "density_grad_mean",
                       "rgb_grad_max", "n_views_visible"]:
                heuristics[k].append(0.0)
            continue

        star_t = torch.tensor(star, dtype=torch.long, device=device)

        heuristics["edge_length"].append(length)

        # Peak contrib
        star_pc = stats.peak_contrib[star_t]
        heuristics["max_peak_contrib"].append(star_pc.max().item())
        heuristics["mean_peak_contrib"].append(star_pc.mean().item())

        # Density
        star_d = tet_density[star_t]
        heuristics["density_max"].append(star_d.max().item())
        heuristics["density_range"].append(
            (star_d.max() - star_d.min()).item())
        heuristics["density_std"].append(
            star_d.std().item() if len(star) > 1 else 0.0)

        # RGB
        star_rgb = tet_rgb[star_t]
        heuristics["rgb_std"].append(
            star_rgb.std(dim=0).mean().item() if len(star) > 1 else 0.0)

        # Density contrast
        heuristics["density_contrast"].append(
            per_tet_contrast[star_t].max().item())

        # Circumcenter shift estimate
        midpoint = (vertices[va] + vertices[vb]) / 2
        shared = [t for t in star
                  if va in indices[t].tolist() and vb in indices[t].tolist()]
        if shared:
            shifts = []
            for t in shared:
                tv = vertices[indices[t]].clone()
                tidx = indices[t].tolist()
                for c in range(4):
                    if tidx[c] == va or tidx[c] == vb:
                        tv[c] = midpoint
                new_cc, _ = calculate_circumcenters_torch(
                    tv.unsqueeze(0).double())
                old_cc = tet_cc[t]
                shifts.append(
                    torch.linalg.norm(new_cc.float().squeeze() - old_cc).item())
            heuristics["circumcenter_shift"].append(np.mean(shifts))
        else:
            heuristics["circumcenter_shift"].append(0.0)

        # Volume, valence
        star_v = tet_vol[star_t]
        heuristics["star_volume"].append(star_v.sum().item())
        heuristics["valence"].append(float(len(star)))

        # Gradients
        relevant_pairs = set()
        for t in star:
            for pi in tet_to_pairs.get(t, []):
                relevant_pairs.add(pi)
        if relevant_pairs:
            rp = torch.tensor(list(relevant_pairs), dtype=torch.long,
                              device=device)
            heuristics["density_grad_max"].append(
                pair_density_grad[rp].max().item())
            heuristics["density_grad_mean"].append(
                pair_density_grad[rp].mean().item())
            heuristics["rgb_grad_max"].append(
                pair_rgb_grad[rp].max().item())
        else:
            heuristics["density_grad_max"].append(0.0)
            heuristics["density_grad_mean"].append(0.0)
            heuristics["rgb_grad_max"].append(0.0)

        # Visibility
        n_visible = 0
        for cam in cameras:
            proj = project_point(midpoint, cam, device)
            if proj is not None:
                px, py, depth = proj
                if 0 <= px < cam.image_width and 0 <= py < cam.image_height:
                    n_visible += 1
        heuristics["n_views_visible"].append(float(n_visible))

    return dict(heuristics)


# ---------------------------------------------------------------------------
# Correlation & plotting
# ---------------------------------------------------------------------------

def compute_correlations(ground_truth, heuristics):
    gt = np.array(ground_truth)
    results = {}
    for name, values in heuristics.items():
        vals = np.array(values)
        if np.std(gt) < 1e-12 or np.std(vals) < 1e-12:
            results[name] = 0.0
            continue
        corr, pval = spearmanr(gt, vals)
        results[name] = corr if not np.isnan(corr) else 0.0
    return results


def plot_correlations(correlations, title, output_path):
    names = list(correlations.keys())
    vals = [correlations[n] for n in names]
    order = np.argsort(np.abs(vals))[::-1]
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.4)))
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in vals]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Spearman Correlation')
    ax.set_title(title)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(-1, 1)
    for i, v in enumerate(vals):
        ax.text(v + 0.02 * np.sign(v), i, f'{v:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_scatter_grid(ground_truth, heuristics, title, output_path, n_cols=4):
    names = list(heuristics.keys())
    n = len(names)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    gt = np.array(ground_truth)
    for i, name in enumerate(names):
        ax = axes[i]
        vals = np.array(heuristics[name])
        ax.scatter(vals, gt, alpha=0.3, s=8)
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel('PSNR Delta', fontsize=8)
        corr, _ = spearmanr(gt, vals) if np.std(vals) > 1e-12 else (0, 1)
        ax.set_title(f'r={corr:.3f}', fontsize=9)
        ax.tick_params(labelsize=7)
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Edge collapse importance via local PSNR")
    parser.add_argument("--ckpt", type=str,
                        default="output/bicycle_exp1_densify")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--output-dir", type=str,
                        default="output/vertex_importance")
    parser.add_argument("--n-edges", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--min-views", type=int, default=3,
                        help="Min test views an edge must be visible in")
    parser.add_argument("--seed", type=int, default=42)
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

    # Load model
    print("Loading checkpoint...")
    model, _ = load_model_and_config(args.ckpt, device)
    model.eval()
    vertices = model.vertices.detach()
    indices = model.indices.detach()
    n_verts = vertices.shape[0]
    n_interior = model.interior_vertices.shape[0]
    n_tets = indices.shape[0]
    print(f"  {n_verts} vertices ({n_interior} interior), {n_tets} tets")

    # Collect render stats
    print("\nCollecting render stats...")
    num_stat_cams = min(200, len(train_cameras))
    sampler = SimpleSampler(len(train_cameras), num_stat_cams, device)
    sampled_cams = [train_cameras[i] for i in sampler.nextids()]
    stats = collect_render_stats(sampled_cams, model, config, device)

    # Pre-compute field values
    print("\nEvaluating field at all circumcenters...")
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
    print("\nBuilding topology...")
    adjacency = build_vertex_tet_adjacency(indices, n_verts)
    tet_neighbor_pairs, _ = build_tv_struct(vertices, indices.long(),
                                             device=device)
    edges = build_edge_list(indices)
    print(f"  {edges.shape[0]} unique edges")

    # ===================================================================
    # Step 1: Sample length-controlled, spatially-spread edges
    # ===================================================================
    print(f"\n{'='*60}")
    print("Step 1: Sampling edges")
    print(f"{'='*60}")

    sampled_edges = sample_length_controlled_edges(
        edges, vertices, n_interior, args.n_edges,
        test_cameras, device, min_views=args.min_views)
    print(f"  Selected {len(sampled_edges)} edges")

    if not sampled_edges:
        print("ERROR: No edges selected. Exiting.")
        return

    lengths = [e[2] for e in sampled_edges]
    print(f"  Length: mean={np.mean(lengths):.6f}, "
          f"std={np.std(lengths):.6f}, "
          f"range=[{np.min(lengths):.6f}, {np.max(lengths):.6f}]")

    # Precompute midpoints
    midpoints_all = []
    for va, vb, length in sampled_edges:
        mid = (vertices[va] + vertices[vb]) / 2
        midpoints_all.append(mid)

    # ===================================================================
    # Step 2: Compute heuristics (before any modification)
    # ===================================================================
    print(f"\n{'='*60}")
    print("Step 2: Computing heuristics")
    print(f"{'='*60}")

    heuristics = compute_all_heuristics(
        sampled_edges, adjacency, model, stats, vertices, indices,
        tet_density, tet_rgb, tet_cc, tet_radii, tet_vol,
        tet_neighbor_pairs, test_cameras, device)

    # Free field tensors
    del tet_density, tet_rgb, tet_cc, tet_radii, tet_vol
    del all_tet_verts, stats, adjacency, tet_neighbor_pairs
    gc.collect()
    torch.cuda.empty_cache()

    # ===================================================================
    # Step 3: Render baseline
    # ===================================================================
    print(f"\n{'='*60}")
    print("Step 3: Rendering baseline")
    print(f"{'='*60}")

    # Reload model fresh for rendering
    del model
    gc.collect()
    torch.cuda.empty_cache()

    model, _ = load_model_and_config(args.ckpt, device)
    model.eval()

    baseline_images = render_all_cameras(model, test_cameras, config, device)
    gt_images = [cam.original_image.cpu() for cam in test_cameras]
    print(f"  Rendered {len(baseline_images)} baseline images")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ===================================================================
    # Step 4: Batch collapse + render
    # ===================================================================
    print(f"\n{'='*60}")
    print("Step 4: Batch edge collapse + local PSNR measurement")
    print(f"{'='*60}")

    batches = assign_batches(sampled_edges, vertices, args.batch_size, device)
    print(f"  {len(batches)} batches of up to {args.batch_size} edges")

    all_psnr_results = {}
    edge_offset = 0

    for bi, batch in enumerate(batches):
        print(f"\n  Batch {bi+1}/{len(batches)} ({len(batch)} edges)...")
        t0 = time.time()

        # Load fresh model
        model, _ = load_model_and_config(args.ckpt, device)
        model.eval()

        # Compute midpoints for this batch (from fresh model vertices)
        batch_midpoints = []
        for va, vb, length in batch:
            mid = (model.vertices[va] + model.vertices[vb]) / 2
            batch_midpoints.append(mid.cpu())

        # Collapse and render
        collapsed_images = collapse_edges_and_render(
            model, batch, test_cameras, config, device)

        # Measure local PSNR
        batch_results = measure_batch_psnr_deltas(
            baseline_images, collapsed_images, gt_images,
            batch, batch_midpoints, test_cameras,
            args.window_size, device)

        for edge_idx, result in batch_results.items():
            global_idx = edge_offset + edge_idx
            all_psnr_results[global_idx] = result

        edge_offset += len(batch)

        del model, collapsed_images
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

    # ===================================================================
    # Step 5: Correlation analysis
    # ===================================================================
    print(f"\n{'='*60}")
    print("Step 5: Correlation Analysis")
    print(f"{'='*60}")

    # Collect ground truth
    psnr_deltas = []
    valid_mask = []
    for i in range(len(sampled_edges)):
        if i in all_psnr_results and all_psnr_results[i]['n_views'] > 0:
            psnr_deltas.append(all_psnr_results[i]['mean_psnr_delta'])
            valid_mask.append(True)
        else:
            psnr_deltas.append(0.0)
            valid_mask.append(False)

    n_valid = sum(valid_mask)
    print(f"  {n_valid}/{len(sampled_edges)} edges with valid PSNR measurements")

    if n_valid < 10:
        print("ERROR: Too few valid edges. Exiting.")
        return

    # Filter heuristics and ground truth to valid edges
    valid_psnr = [psnr_deltas[i] for i in range(len(sampled_edges))
                  if valid_mask[i]]
    valid_heuristics = {}
    for name, values in heuristics.items():
        valid_heuristics[name] = [values[i] for i in range(len(sampled_edges))
                                   if valid_mask[i]]

    print(f"\n  PSNR delta stats: mean={np.mean(valid_psnr):.4f}, "
          f"std={np.std(valid_psnr):.4f}, "
          f"range=[{np.min(valid_psnr):.4f}, {np.max(valid_psnr):.4f}]")

    # Correlations
    correlations = compute_correlations(valid_psnr, valid_heuristics)
    sorted_corrs = sorted(correlations.items(),
                          key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Heuristic':30s} | {'Correlation':>12s}")
    print("-" * 45)
    for name, corr in sorted_corrs:
        print(f"{name:30s} | {corr:+12.4f}")

    # Save
    results = {
        "n_edges": len(sampled_edges),
        "n_valid": n_valid,
        "window_size": args.window_size,
        "length_stats": {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
        },
        "psnr_delta_stats": {
            "mean": float(np.mean(valid_psnr)),
            "std": float(np.std(valid_psnr)),
            "min": float(np.min(valid_psnr)),
            "max": float(np.max(valid_psnr)),
        },
        "correlations": correlations,
        "per_edge": [
            {
                "va": sampled_edges[i][0],
                "vb": sampled_edges[i][1],
                "length": sampled_edges[i][2],
                "psnr_delta": psnr_deltas[i],
                "n_views": all_psnr_results.get(i, {}).get('n_views', 0),
            }
            for i in range(len(sampled_edges))
        ],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plots
    plot_correlations(correlations,
                      "Edge Collapse: Heuristic Correlations with Local PSNR Delta",
                      output_dir / "correlations.png")
    plot_scatter_grid(valid_psnr, valid_heuristics,
                      "Edge Collapse: Heuristics vs Local PSNR Delta",
                      output_dir / "scatter.png")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
