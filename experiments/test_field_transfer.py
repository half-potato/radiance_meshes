"""
Per-Tet Attribute Transfer Through Retriangulation

Uses vertex adjacency for O(1) candidate lookup — no spatial hash or tet walk.
Since vertices persist through retriangulation, for each new tet we gather
candidate old tets via the old vertex→tet adjacency table, then pick the best.

Usage:
    # Single config:
    uv run python experiments/test_field_transfer.py --n_points 2000 --perturbation 0.1

    # Full sweep:
    uv run python experiments/test_field_transfer.py --sweep --output_dir output/field_transfer/
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.topo_utils import calculate_circumcenters_torch, tet_volumes
from dtlookup.dtlookup import TetrahedraLookup


# ---------------------------------------------------------------------------
# Synthetic field
# ---------------------------------------------------------------------------

def sum_of_gaussians(points: torch.Tensor, n_gaussians: int = 8, seed: int = 42) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    centers = torch.tensor(rng.uniform(-0.8, 0.8, (n_gaussians, 3)), device=points.device, dtype=points.dtype)
    widths = torch.tensor(rng.uniform(0.15, 0.4, n_gaussians), device=points.device, dtype=points.dtype)
    amplitudes = torch.tensor(rng.uniform(0.5, 2.0, n_gaussians), device=points.device, dtype=points.dtype)
    diff = points[:, None, :] - centers[None, :, :]
    sq_dist = (diff * diff).sum(dim=-1)
    return (amplitudes[None, :] * torch.exp(-sq_dist / (2 * widths[None, :] ** 2))).sum(dim=-1)


def sum_of_gaussians_with_grad(points: torch.Tensor, n_gaussians: int = 8, seed: int = 42):
    rng = np.random.RandomState(seed)
    centers = torch.tensor(rng.uniform(-0.8, 0.8, (n_gaussians, 3)), device=points.device, dtype=points.dtype)
    widths = torch.tensor(rng.uniform(0.15, 0.4, n_gaussians), device=points.device, dtype=points.dtype)
    amplitudes = torch.tensor(rng.uniform(0.5, 2.0, n_gaussians), device=points.device, dtype=points.dtype)
    diff = points[:, None, :] - centers[None, :, :]
    sq_dist = (diff * diff).sum(dim=-1)
    inv_var = 1.0 / (2 * widths[None, :] ** 2)
    gauss = amplitudes[None, :] * torch.exp(-sq_dist * inv_var)
    values = gauss.sum(dim=-1)
    grad = (gauss[:, :, None] * (-2 * inv_var[:, :, None]) * diff).sum(dim=1)
    return values, grad


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def build_delaunay(points_np: np.ndarray, device: torch.device):
    tri = Delaunay(points_np)
    indices = torch.tensor(tri.simplices, dtype=torch.int32, device=device)
    vertices = torch.tensor(tri.points, dtype=torch.float32, device=device)
    vols = tet_volumes(vertices[indices.long()])
    neg = vols < 0
    if neg.any():
        idx = indices.clone()
        idx[neg] = idx[neg][:, [1, 0, 2, 3]]
        indices = idx
    return indices, vertices


def compute_centroids(indices: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    return vertices[indices.long()].mean(dim=1)


def compute_mean_edge_length(indices: torch.Tensor, vertices: torch.Tensor) -> float:
    tv = vertices[indices.long()]
    edges = torch.stack([
        tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0], tv[:, 3] - tv[:, 0],
        tv[:, 2] - tv[:, 1], tv[:, 3] - tv[:, 1], tv[:, 3] - tv[:, 2],
    ], dim=1)
    return edges.norm(dim=-1).mean().item()


# ---------------------------------------------------------------------------
# Vertex-to-tet adjacency (CSR-style, built once per old mesh)
# ---------------------------------------------------------------------------

def build_v2t(indices: torch.Tensor, n_verts: int):
    """Build padded vertex→tet adjacency table.

    Returns:
        v2t: (V, max_valence) int64, padded with -1
        valence: (V,) int64, actual count per vertex
    """
    T = indices.shape[0]
    device = indices.device
    flat_vidx = indices.long().reshape(-1)                          # (T*4,)
    flat_tidx = torch.arange(T, device=device).repeat_interleave(4) # (T*4,)

    # Sort by vertex index
    sort_order = torch.argsort(flat_vidx)
    sorted_vidx = flat_vidx[sort_order]
    sorted_tidx = flat_tidx[sort_order]

    valence = torch.bincount(flat_vidx, minlength=n_verts)
    offsets = torch.zeros(n_verts + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(valence, dim=0)
    max_val = valence.max().item()

    # Compute local position within each vertex's group
    group_starts = offsets[sorted_vidx]
    local_pos = torch.arange(len(sorted_vidx), device=device) - group_starts

    v2t = torch.full((n_verts, max_val), -1, dtype=torch.long, device=device)
    v2t[sorted_vidx, local_pos] = sorted_tidx.long()

    return v2t, valence


def gather_candidates(v2t: torch.Tensor, new_indices: torch.Tensor):
    """Gather candidate old tets for each new tet via vertex adjacency.

    Returns:
        candidates: (T_new, 4 * max_valence) old tet indices, -1 for padding
    """
    cands = v2t[new_indices.long()]       # (T_new, 4, max_val)
    T_new = new_indices.shape[0]
    return cands.reshape(T_new, -1)       # (T_new, C)


# ---------------------------------------------------------------------------
# Transfer methods
# ---------------------------------------------------------------------------

@dataclass
class TransferResult:
    name: str
    new_values: torch.Tensor
    wall_time: float


def transfer_adj_nearest_cc(
    candidates: torch.Tensor,
    old_values: torch.Tensor,
    old_cc: torch.Tensor,
    new_cc: torch.Tensor,
) -> TransferResult:
    """Pick candidate with nearest circumcenter, copy its value."""
    t0 = time.perf_counter()
    valid = candidates >= 0
    cands_safe = candidates.clamp(min=0)

    dists = (old_cc[cands_safe] - new_cc.unsqueeze(1)).pow(2).sum(dim=-1)
    dists[~valid] = float("inf")

    best = dists.argmin(dim=1)
    best_tet = cands_safe.gather(1, best.unsqueeze(1)).squeeze(1)
    new_values = old_values[best_tet]
    return TransferResult("adj_near_cc", new_values, time.perf_counter() - t0)


def transfer_adj_nearest_centroid(
    candidates: torch.Tensor,
    old_values: torch.Tensor,
    old_centroids: torch.Tensor,
    new_centroids: torch.Tensor,
) -> TransferResult:
    """Pick candidate with nearest centroid, copy its value."""
    t0 = time.perf_counter()
    valid = candidates >= 0
    cands_safe = candidates.clamp(min=0)

    dists = (old_centroids[cands_safe] - new_centroids.unsqueeze(1)).pow(2).sum(dim=-1)
    dists[~valid] = float("inf")

    best = dists.argmin(dim=1)
    best_tet = cands_safe.gather(1, best.unsqueeze(1)).squeeze(1)
    new_values = old_values[best_tet]
    return TransferResult("adj_near_ctr", new_values, time.perf_counter() - t0)


def transfer_adj_max_shared(
    candidates: torch.Tensor,
    old_values: torch.Tensor,
    old_centroids: torch.Tensor,
    new_centroids: torch.Tensor,
) -> TransferResult:
    """Pick the candidate sharing the most vertices with the new tet.

    A candidate reached through k different new-tet vertices appears k times
    in the candidate list (once per vertex adjacency). So the duplicate count
    in the sorted candidate row IS the shared vertex count.
    Tie-break: nearest centroid.
    """
    t0 = time.perf_counter()
    T_new, C = candidates.shape
    device = candidates.device

    sorted_cands, sort_perm = candidates.sort(dim=1)

    # Compute streak length: consecutive equal values after sorting
    # streak[i,j] = how many preceding equal values at position j
    streak = torch.zeros(T_new, C, device=device, dtype=torch.long)
    for i in range(1, C):
        same = sorted_cands[:, i] == sorted_cands[:, i - 1]
        streak[:, i] = torch.where(same, streak[:, i - 1] + 1, torch.zeros(T_new, device=device, dtype=torch.long))

    # Mask out invalid candidates (-1)
    streak[sorted_cands < 0] = -1

    # For tie-breaking, subtract a small centroid distance penalty
    # Scale distances to [0, 0.5) so they never override a higher streak
    cands_safe = sorted_cands.clamp(min=0)
    dists = (old_centroids[cands_safe] - new_centroids.unsqueeze(1)).pow(2).sum(dim=-1)
    max_dist = dists[sorted_cands >= 0].max().clamp(min=1e-12)
    dist_penalty = 0.49 * dists / max_dist  # in [0, 0.49)

    score = streak.float() - dist_penalty
    score[sorted_cands < 0] = float("-inf")

    best = score.argmax(dim=1)
    best_tet = cands_safe.gather(1, best.unsqueeze(1)).squeeze(1)
    new_values = old_values[best_tet]
    return TransferResult("adj_shared", new_values, time.perf_counter() - t0)


def transfer_adj_idw(
    candidates: torch.Tensor,
    old_values: torch.Tensor,
    old_cc: torch.Tensor,
    new_cc: torch.Tensor,
) -> TransferResult:
    """Inverse-distance-weighted average of all candidate values."""
    t0 = time.perf_counter()
    valid = candidates >= 0
    cands_safe = candidates.clamp(min=0)

    dists = (old_cc[cands_safe] - new_cc.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()
    weights = 1.0 / dists.clamp(min=1e-12)
    weights[~valid] = 0.0

    vals = old_values[cands_safe]
    vals[~valid] = 0.0

    w_sum = weights.sum(dim=1).clamp(min=1e-30)
    new_values = (vals * weights).sum(dim=1) / w_sum
    return TransferResult("adj_idw", new_values, time.perf_counter() - t0)


def transfer_adj_grad(
    candidates: torch.Tensor,
    old_values: torch.Tensor,
    old_gradients: torch.Tensor,
    old_cc: torch.Tensor,
    new_cc: torch.Tensor,
) -> TransferResult:
    """Pick nearest candidate by CC distance, apply gradient correction."""
    t0 = time.perf_counter()
    valid = candidates >= 0
    cands_safe = candidates.clamp(min=0)

    diff = old_cc[cands_safe] - new_cc.unsqueeze(1)
    dists = diff.pow(2).sum(dim=-1)
    dists[~valid] = float("inf")

    best = dists.argmin(dim=1)
    best_tet = cands_safe.gather(1, best.unsqueeze(1)).squeeze(1)

    v = old_values[best_tet]
    g = old_gradients[best_tet]                          # (T_new, 3)
    cc_old = old_cc[best_tet]                             # (T_new, 3)
    correction = (g * (new_cc - cc_old)).sum(dim=-1)      # (T_new,)
    new_values = v + correction
    return TransferResult("adj_grad", new_values, time.perf_counter() - t0)


def transfer_adj_idw_grad(
    candidates: torch.Tensor,
    old_values: torch.Tensor,
    old_gradients: torch.Tensor,
    old_cc: torch.Tensor,
    new_cc: torch.Tensor,
    mean_edge: float,
) -> TransferResult:
    """Gaussian-weighted average of gradient-corrected candidate values.

    Uses exp(-dist^2 / (2*h^2)) with h = mean_edge_length so that
    distant candidates (where gradient extrapolation is unreliable)
    contribute almost nothing.
    """
    t0 = time.perf_counter()
    valid = candidates >= 0
    cands_safe = candidates.clamp(min=0)

    diff = new_cc.unsqueeze(1) - old_cc[cands_safe]       # (T_new, C, 3)
    sq_dist = diff.pow(2).sum(dim=-1)                      # (T_new, C)

    # Gradient-corrected values at the new circumcenter
    v = old_values[cands_safe]                             # (T_new, C)
    g = old_gradients[cands_safe]                          # (T_new, C, 3)
    correction = (g * diff).sum(dim=-1)                    # (T_new, C)
    corrected = v + correction

    # Gaussian kernel — suppresses distant (unreliable) extrapolations
    h_sq = 2.0 * mean_edge * mean_edge
    weights = torch.exp(-sq_dist / h_sq)
    weights[~valid] = 0.0
    corrected[~valid] = 0.0

    w_sum = weights.sum(dim=1).clamp(min=1e-30)
    new_values = (corrected * weights).sum(dim=1) / w_sum
    return TransferResult("adj_gauss_grad", new_values, time.perf_counter() - t0)


def transfer_vertex_scatter(
    old_indices: torch.Tensor,
    old_values: torch.Tensor,
    new_indices: torch.Tensor,
    n_vertices: int,
) -> TransferResult:
    """Scatter old tet values to vertices, gather mean per new tet."""
    t0 = time.perf_counter()
    device = old_values.device
    flat_idx = old_indices.long().reshape(-1)
    flat_vals = old_values.unsqueeze(1).expand(-1, 4).reshape(-1)
    vertex_sum = torch.zeros(n_vertices, device=device)
    vertex_count = torch.zeros(n_vertices, device=device)
    vertex_sum.scatter_add_(0, flat_idx, flat_vals)
    vertex_count.scatter_add_(0, flat_idx, torch.ones_like(flat_vals))
    vertex_vals = vertex_sum / vertex_count.clamp(min=1)
    new_values = vertex_vals[new_indices.long()].mean(dim=1)
    return TransferResult("vertex", new_values, time.perf_counter() - t0)


def transfer_baseline_lookup(
    old_indices: torch.Tensor,
    old_vertices: torch.Tensor,
    old_values: torch.Tensor,
    old_cc: torch.Tensor,
    new_indices: torch.Tensor,
    new_vertices: torch.Tensor,
) -> TransferResult:
    """Reference baseline: TetrahedraLookup centroid containment."""
    t0 = time.perf_counter()
    lookup = TetrahedraLookup(old_indices, old_vertices)
    new_centroids = compute_centroids(new_indices, new_vertices)
    hit_ids = lookup.lookup(new_centroids)

    new_values = torch.zeros(new_indices.shape[0], device=old_values.device)
    valid = hit_ids >= 0
    new_values[valid] = old_values[hit_ids[valid].long()]
    if (~valid).any():
        dists = torch.cdist(new_centroids[~valid], old_cc)
        new_values[~valid] = old_values[dists.argmin(dim=1)]

    return TransferResult("baseline", new_values, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    name: str
    l2: float
    l2_vol_weighted: float
    linf: float
    wall_time: float


def evaluate(result: TransferResult, gt: torch.Tensor, new_indices: torch.Tensor, new_vertices: torch.Tensor) -> Metrics:
    err = (result.new_values - gt).abs()
    l2 = err.pow(2).mean().sqrt().item()
    linf = err.max().item()
    vols = tet_volumes(new_vertices[new_indices.long()]).abs()
    l2_vw = (err.pow(2) * vols / vols.sum().clamp(min=1e-30)).sum().sqrt().item()
    return Metrics(result.name, l2, l2_vw, linf, result.wall_time)


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single(n_points: int, perturbation: float, device: torch.device, seed: int = 42) -> list[Metrics]:
    rng = np.random.RandomState(seed)
    points_np = rng.uniform(-1, 1, (n_points, 3)).astype(np.float64)

    # Old mesh
    old_indices, old_vertices = build_delaunay(points_np, device)
    old_cc, _ = calculate_circumcenters_torch(old_vertices[old_indices.long()])
    old_centroids = compute_centroids(old_indices, old_vertices)
    old_values, old_gradients = sum_of_gaussians_with_grad(old_cc)

    # Vertex adjacency (one-time build)
    t_build = time.perf_counter()
    v2t, valence = build_v2t(old_indices, old_vertices.shape[0])
    t_build = time.perf_counter() - t_build

    # Perturb & retriangulate
    mean_edge = compute_mean_edge_length(old_indices, old_vertices)
    noise_std = perturbation * mean_edge
    noise = torch.tensor(rng.normal(0, noise_std, points_np.shape), device=device, dtype=torch.float32)
    new_points_np = points_np + noise.cpu().numpy()

    new_indices, new_vertices = build_delaunay(new_points_np, device)
    new_cc, _ = calculate_circumcenters_torch(new_vertices[new_indices.long()])
    new_centroids = compute_centroids(new_indices, new_vertices)
    gt_values = sum_of_gaussians(new_cc)

    # Gather candidates (shared across all adj methods)
    t_gather = time.perf_counter()
    candidates = gather_candidates(v2t, new_indices)
    t_gather = time.perf_counter() - t_gather

    T_old = old_indices.shape[0]
    T_new = new_indices.shape[0]
    max_val = v2t.shape[1]
    print(f"  old_tets={T_old}  new_tets={T_new}  max_valence={max_val}  "
          f"candidates/tet={4*max_val}  v2t_build={t_build:.4f}s  gather={t_gather:.6f}s")

    results = []

    r = transfer_adj_nearest_cc(candidates, old_values, old_cc, new_cc)
    results.append(evaluate(r, gt_values, new_indices, new_vertices))

    r = transfer_adj_nearest_centroid(candidates, old_values, old_centroids, new_centroids)
    results.append(evaluate(r, gt_values, new_indices, new_vertices))

    r = transfer_adj_max_shared(candidates, old_values, old_centroids, new_centroids)
    results.append(evaluate(r, gt_values, new_indices, new_vertices))

    r = transfer_adj_idw(candidates, old_values, old_cc, new_cc)
    results.append(evaluate(r, gt_values, new_indices, new_vertices))

    r = transfer_adj_grad(candidates, old_values, old_gradients, old_cc, new_cc)
    results.append(evaluate(r, gt_values, new_indices, new_vertices))

    r = transfer_adj_idw_grad(candidates, old_values, old_gradients, old_cc, new_cc, mean_edge)
    results.append(evaluate(r, gt_values, new_indices, new_vertices))

    r = transfer_vertex_scatter(old_indices, old_values, new_indices, old_vertices.shape[0])
    results.append(evaluate(r, gt_values, new_indices, new_vertices))

    r = transfer_baseline_lookup(old_indices, old_vertices, old_values, old_cc, new_indices, new_vertices)
    results.append(evaluate(r, gt_values, new_indices, new_vertices))

    return results


# ---------------------------------------------------------------------------
# Sweep & plotting
# ---------------------------------------------------------------------------

def run_sweep(device: torch.device, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    perturbations = [0.01, 0.05, 0.1, 0.2, 0.5]
    mesh_sizes = [1000, 5000]

    all_data = []

    for n_pts in mesh_sizes:
        for pert in perturbations:
            print(f"  n_points={n_pts}, perturbation={pert}")
            metrics = run_single(n_pts, pert, device)
            all_data.append((n_pts, pert, metrics))
            for m in metrics:
                print(f"    {m.name:20s}  L2={m.l2:.6f}  L2vw={m.l2_vol_weighted:.6f}  Linf={m.linf:.6f}  t={m.wall_time:.4f}s")

    # --- Plot: Error vs perturbation ---
    for n_pts in mesh_sizes:
        fig, ax = plt.subplots(figsize=(9, 5))
        method_names = [m.name for m in all_data[0][2]]
        for midx, mname in enumerate(method_names):
            xs = [p for n, p, ms in all_data if n == n_pts]
            ys = [ms[midx].l2 for n, p, ms in all_data if n == n_pts]
            ax.plot(xs, ys, "o-", label=mname)
        ax.set_xlabel("Perturbation (fraction of mean edge length)")
        ax.set_ylabel("L2 Error")
        ax.set_title(f"Transfer Error vs Perturbation (n={n_pts})")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"error_vs_pert_n{n_pts}.png"), dpi=150)
        plt.close(fig)

    # --- Plot: Timing comparison ---
    fig, ax = plt.subplots(figsize=(9, 5))
    method_names = [m.name for m in all_data[0][2]]
    # Average time across all configs
    avg_times = []
    for midx, mname in enumerate(method_names):
        times = [ms[midx].wall_time for _, _, ms in all_data]
        avg_times.append(np.mean(times))
    bars = ax.barh(method_names, avg_times)
    ax.set_xlabel("Wall time (seconds)")
    ax.set_title("Average Transfer Time per Method")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "timing.png"), dpi=150)
    plt.close(fig)

    # --- Plot: L2 vs Time (pareto) ---
    for n_pts in mesh_sizes:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Use pert=0.1 data
        for n, p, ms in all_data:
            if n != n_pts or abs(p - 0.1) > 0.01:
                continue
            for m in ms:
                ax.scatter(m.wall_time, m.l2, s=60, zorder=5)
                ax.annotate(m.name, (m.wall_time, m.l2), fontsize=7,
                            xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("Wall time (seconds)")
        ax.set_ylabel("L2 Error")
        ax.set_title(f"Error vs Speed (n={n_pts}, pert=0.1)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"pareto_n{n_pts}.png"), dpi=150)
        plt.close(fig)

    print(f"\nPlots saved to {output_dir}")


def print_summary(metrics_list: list[Metrics]):
    header = f"{'Method':20s} {'L2':>10s} {'L2(vol)':>10s} {'Linf':>10s} {'Time(s)':>10s}"
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        print(f"{m.name:20s} {m.l2:10.6f} {m.l2_vol_weighted:10.6f} {m.linf:10.6f} {m.wall_time:10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Per-tet attribute transfer experiment")
    parser.add_argument("--n_points", type=int, default=2000)
    parser.add_argument("--perturbation", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output/field_transfer/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.sweep:
        print("Running full sweep...")
        run_sweep(device, args.output_dir)
    else:
        print(f"Config: n_points={args.n_points}, perturbation={args.perturbation}")
        metrics = run_single(args.n_points, args.perturbation, device, seed=args.seed)
        print()
        print_summary(metrics)


if __name__ == "__main__":
    main()
