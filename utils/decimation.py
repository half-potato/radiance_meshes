"""
Vectorized edge-star decimation for Delaunay meshes.

Implements the `len_over_rgb_std` heuristic:
    score = edge_length / (edge_star_rgb_std + eps)

Low-score edges (short, in uniform RGB regions) are collapsed first.
"""

import torch
import torch.nn as nn
import time
from utils.topo_utils import calculate_circumcenters_torch
from utils.contraction import contract_mean_std


def build_edge_list(indices):
    """Extract unique sorted edges from tet indices.

    Args:
        indices: (T, 4) int tensor of tet vertex indices.

    Returns:
        (E, 2) int64 tensor of unique edges, sorted per-edge (va < vb).
    """
    idx = indices.long()
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    all_edges = []
    for i, j in pairs:
        all_edges.append(torch.stack([idx[:, i], idx[:, j]], dim=1))
    all_edges = torch.cat(all_edges, dim=0)
    all_edges, _ = all_edges.sort(dim=1)
    return torch.unique(all_edges, dim=0)


@torch.no_grad()
def query_tet_rgb(model):
    """Query backbone at circumcenters to get per-tet RGB.

    Returns:
        (T, 3) float tensor of RGB values.
    """
    vertices = model.vertices
    indices = model.indices
    T = indices.shape[0]
    chunk = 50000
    rgbs = []
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        _, density, rgb, grd, sh, *rest = model.compute_batch_features(
            vertices, indices, start, end)
        rgbs.append(rgb.float())
        del density, grd, sh
    return torch.cat(rgbs, dim=0)


@torch.no_grad()
def compute_edge_scores(edges, indices, vertices, tet_rgb, n_interior):
    """Vectorized len_over_rgb_std scoring.

    Args:
        edges: (E, 2) int64 unique sorted edges
        indices: (T, 4) int tet indices
        vertices: (V, 3) float vertex positions
        tet_rgb: (T, 3) float per-tet RGB
        n_interior: int, number of interior vertices

    Returns:
        (E,) float scores. Lower = collapse first. Exterior edges = inf.
    """
    device = edges.device
    E = edges.shape[0]
    T = indices.shape[0]
    V_max = int(vertices.shape[0])

    # --- 1. Build edge→tet mapping ---
    # Generate all 6 edges per tet
    idx = indices.long()
    pair_offsets = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    all_va = []
    all_vb = []
    all_tet_idx = []
    tet_arange = torch.arange(T, device=device)
    for i, j in pair_offsets:
        ei, ej = idx[:, i], idx[:, j]
        # Sort so va < vb
        va = torch.min(ei, ej)
        vb = torch.max(ei, ej)
        all_va.append(va)
        all_vb.append(vb)
        all_tet_idx.append(tet_arange)

    all_va = torch.cat(all_va)       # (6T,)
    all_vb = torch.cat(all_vb)       # (6T,)
    all_tet_idx = torch.cat(all_tet_idx)  # (6T,)

    # Pack edge keys as int64
    all_keys = all_va * V_max + all_vb          # (6T,)
    edge_keys = edges[:, 0].long() * V_max + edges[:, 1].long()  # (E,)

    # Map each (6T) pair to its unique edge index via searchsorted
    edge_idx = torch.searchsorted(edge_keys, all_keys)  # (6T,)
    # Clamp to valid range (shouldn't be needed but safety)
    edge_idx = edge_idx.clamp(0, E - 1)

    # --- 2. Per-edge RGB stats via scatter ---
    # Gather RGB for each tet in the pairs
    pair_rgb = tet_rgb[all_tet_idx]  # (6T, 3)

    count = torch.zeros(E, device=device)
    sum_rgb = torch.zeros(E, 3, device=device)
    sum_rgb2 = torch.zeros(E, 3, device=device)

    count.scatter_add_(0, edge_idx, torch.ones(6 * T, device=device))
    sum_rgb.scatter_add_(0, edge_idx.unsqueeze(1).expand_as(pair_rgb), pair_rgb)
    sum_rgb2.scatter_add_(0, edge_idx.unsqueeze(1).expand_as(pair_rgb), pair_rgb ** 2)

    # Variance = E[X^2] - E[X]^2
    safe_count = count.clamp(min=1).unsqueeze(1)
    mean_rgb = sum_rgb / safe_count
    var_rgb = (sum_rgb2 / safe_count - mean_rgb ** 2).clamp(min=0)
    rgb_std = var_rgb.sqrt().mean(dim=1)  # mean std across R,G,B channels

    # --- 3. Edge lengths ---
    edge_lengths = torch.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], dim=1)

    # --- 4. Score ---
    eps = 1e-3
    scores = edge_lengths / (rgb_std + eps)

    # Mark exterior edges as inf (never collapse)
    exterior = (edges[:, 0] >= n_interior) | (edges[:, 1] >= n_interior)
    scores[exterior] = float('inf')

    return scores


@torch.no_grad()
def apply_decimation(model, tet_optim, args, device):
    """Run one round of edge-star decimation.

    Args:
        model: iNGPDW model
        tet_optim: TetOptimizer
        args: training args (needs decimate_count)
        device: torch device

    Returns:
        Number of vertices removed.
    """
    st = time.time()
    was_training = model.training
    model.eval()

    # 1. Query field
    tet_rgb = query_tet_rgb(model)

    # 2. Build edges and score
    edges = build_edge_list(model.indices)
    n_interior = model.num_int_verts
    scores = compute_edge_scores(
        edges, model.indices, model.vertices, tet_rgb, n_interior)

    # 3. Select candidates by threshold or top-k
    threshold = getattr(args, 'decimate_threshold', 0.0)
    if threshold > 0:
        # Threshold mode: collapse all edges with score < threshold
        mask = scores < threshold
        candidate_indices = mask.nonzero(as_tuple=True)[0]
        # Still sort by score so lowest-score edges get priority in conflict resolution
        sub_scores = scores[candidate_indices]
        sub_order = torch.argsort(sub_scores)
        candidates = candidate_indices[sub_order]
    else:
        # Top-k mode (legacy)
        order = torch.argsort(scores)
        max_candidates = min(args.decimate_count * 3, edges.shape[0])
        candidates = order[:max_candidates]

    # 4. Conflict resolution — greedy, no vertex reused
    max_collapses = args.decimate_count if threshold <= 0 else candidates.shape[0]
    # Move to CPU for fast Python loop
    cand_cpu = candidates.cpu()
    scores_cpu = scores.cpu()
    edges_cpu = edges.cpu()
    claimed = set()
    collapse_va = []
    collapse_vb = []
    for i in range(cand_cpu.shape[0]):
        ci = cand_cpu[i].item()
        if scores_cpu[ci] == float('inf'):
            break
        va = edges_cpu[ci, 0].item()
        vb = edges_cpu[ci, 1].item()
        if va in claimed or vb in claimed:
            continue
        # Both must be interior
        if va >= n_interior or vb >= n_interior:
            continue
        claimed.add(va)
        claimed.add(vb)
        collapse_va.append(va)
        collapse_vb.append(vb)
        if len(collapse_va) >= max_collapses:
            break

    n_collapse = len(collapse_va)
    if n_collapse == 0:
        return 0

    collapse_va = torch.tensor(collapse_va, device=device, dtype=torch.long)
    collapse_vb = torch.tensor(collapse_vb, device=device, dtype=torch.long)

    # 5. Move va to midpoint (in-place, preserves Adam momentum)
    verts_data = model.interior_vertices.data
    midpoints = (verts_data[collapse_va] + verts_data[collapse_vb]) / 2
    verts_data[collapse_va] = midpoints

    # 6. Build keep mask and remove vb
    keep_mask = torch.ones(n_interior, dtype=torch.bool, device=device)
    keep_mask[collapse_vb] = False

    tet_optim.remove_points(keep_mask)

    if was_training:
        model.train()

    elapsed = time.time() - st
    threshold = getattr(args, 'decimate_threshold', 0.0)
    mode_str = f"threshold={threshold:.4f}" if threshold > 0 else f"top-k={args.decimate_count}"
    print(f"Decimation ({mode_str}): removed {n_collapse} vertices in {elapsed:.2f}s "
          f"(#V: {model.vertices.shape[0]}, #T: {model.indices.shape[0]})")

    return n_collapse
