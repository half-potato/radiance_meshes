"""Compare shell kernel coverage at different neighborhood sizes and scales.

Tests:
  1. Shell (8 verts: 4 own + 4 flap) vs full star (~35 verts)
  2. Shell-2 (shell + 2nd-ring flaps, ~14-16 verts) vs full star
  3. Both with mean edge scale and 25th-percentile edge scale
  4. Both raw and clipped circumcenters
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))

import torch
import numpy as np
from scipy.spatial import Delaunay
from utils.topo_utils import calculate_circumcenters_torch, build_adj
from utils.safe_math import safe_exp


def project_points_to_tetrahedra(points, tets):
    """Project points onto tetrahedra by clamping negative barycentrics."""
    v0 = tets[:, 0, :]
    T = (tets[:, 1:, :] - v0.unsqueeze(1)).permute(0, 2, 1)
    x = torch.linalg.solve(T, (points - v0).unsqueeze(2)).squeeze(2)
    w0 = 1 - x.sum(dim=1, keepdim=True)
    bary = torch.cat([w0, x], dim=1)
    bary = bary.clamp(min=0)
    norm = bary.sum(dim=1, keepdim=True).clamp(min=1e-8)
    mask = (norm > 1).reshape(-1)
    bary[mask] = bary[mask] / norm[mask]
    p_proj = (T * bary[:, 1:].unsqueeze(1)).sum(dim=2) + v0
    return p_proj


def build_tet_to_verts_star(indices, n_verts):
    """For each tet, collect all vertices in its 1-ring star."""
    T = indices.shape[0]
    device = indices.device
    vert_to_tets: dict[int, list[int]] = {}
    idx_cpu = indices.cpu().numpy()
    for t in range(T):
        for v in idx_cpu[t]:
            vert_to_tets.setdefault(int(v), []).append(t)
    star_verts = []
    for t in range(T):
        vset = set()
        for v in idx_cpu[t]:
            for neigh_t in vert_to_tets[int(v)]:
                for nv in idx_cpu[neigh_t]:
                    vset.add(int(nv))
        star_verts.append(torch.tensor(sorted(vset), dtype=torch.long, device=device))
    return star_verts


def compute_flap_indices(indices, tet_adj):
    """Replicate _compute_flap_indices logic standalone."""
    T = indices.shape[0]
    device = indices.device
    flap = torch.full((T, 4), -1, dtype=torch.long, device=device)
    for j in range(4):
        neigh = tet_adj[:, j]
        has_neigh = neigh >= 0
        if not has_neigh.any():
            continue
        own_verts = indices[has_neigh]
        neigh_verts = indices[neigh[has_neigh]]
        not_shared = (neigh_verts.unsqueeze(2) != own_verts.unsqueeze(1)).all(dim=2)
        flap_local = not_shared.float().argmax(dim=1)
        flap_vertex = neigh_verts[torch.arange(neigh_verts.shape[0], device=device), flap_local]
        flap[has_neigh, j] = flap_vertex.long()
    return flap


def build_shell2_verts(indices, tet_adj, flap_indices):
    """Build 2nd-ring shell: own verts + flap verts + flap-of-flap verts.

    For each face neighbor, grab its flap vertices (the vertices opposite to
    its other 3 faces, excluding the shared face back to the original tet).
    """
    T = indices.shape[0]
    device = indices.device
    idx_cpu = indices.cpu().numpy()
    adj_cpu = tet_adj.cpu().numpy()
    flap_cpu = flap_indices.cpu().numpy()

    # Also need flap indices for each neighbor tet
    shell2 = []
    for t in range(T):
        vset = set(int(v) for v in idx_cpu[t])
        # Add 1st ring flaps
        for j in range(4):
            fv = int(flap_cpu[t, j])
            if fv >= 0:
                vset.add(fv)
        # Add 2nd ring: for each face neighbor, grab its flap vertices
        for j in range(4):
            neigh_t = int(adj_cpu[t, j])
            if neigh_t < 0:
                continue
            for k in range(4):
                fv2 = int(flap_cpu[neigh_t, k])
                if fv2 >= 0:
                    vset.add(fv2)
        shell2.append(torch.tensor(sorted(vset), dtype=torch.long, device=device))
    return shell2


def compute_vertex_edge_scale(vertices, indices, mode="mean"):
    """Compute per-vertex edge scale. mode: 'mean' or 'p25' (25th percentile)."""
    n_verts = vertices.shape[0]
    device = vertices.device
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    if mode == "mean":
        edge_sum = torch.zeros(n_verts, device=device)
        edge_count = torch.zeros(n_verts, device=device)
        ones = torch.ones(indices.shape[0], device=device)
        for i, j in pairs:
            elen = torch.linalg.norm(vertices[indices[:, i]] - vertices[indices[:, j]], dim=-1)
            edge_sum.scatter_add_(0, indices[:, i], elen)
            edge_sum.scatter_add_(0, indices[:, j], elen)
            edge_count.scatter_add_(0, indices[:, i], ones)
            edge_count.scatter_add_(0, indices[:, j], ones)
        return (edge_sum / edge_count.clamp(min=1)).clamp(min=1e-8)

    elif mode == "p25":
        # Collect all edge lengths per vertex, then take 25th percentile
        vert_edges: dict[int, list[float]] = {v: [] for v in range(n_verts)}
        idx_cpu = indices.cpu().numpy()
        verts_cpu = vertices.cpu().numpy()
        for i, j in pairs:
            vi = idx_cpu[:, i]
            vj = idx_cpu[:, j]
            elens = np.linalg.norm(verts_cpu[vi] - verts_cpu[vj], axis=-1)
            for k in range(len(elens)):
                vert_edges[int(vi[k])].append(float(elens[k]))
                vert_edges[int(vj[k])].append(float(elens[k]))
        scale = torch.zeros(n_verts, device=device)
        for v in range(n_verts):
            edges = vert_edges[v]
            if edges:
                scale[v] = float(np.percentile(edges, 25))
            else:
                scale[v] = 1e-8
        return scale.clamp(min=1e-8)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def eval_kernel(eval_pt, vert_ids, vertices, vertex_edge_scale, kernel_sigma):
    """Evaluate Gaussian kernel weights for vertices at a given point."""
    pos = vertices[vert_ids]
    dist = torch.linalg.norm(pos - eval_pt.unsqueeze(0), dim=-1)
    scale = vertex_edge_scale[vert_ids]
    inv_2sigma2 = 1.0 / (2.0 * kernel_sigma * kernel_sigma)
    return safe_exp(-(dist / scale) ** 2 * inv_2sigma2)


def run_comparison(label, eval_pts, indices, flap_indices, neighborhoods,
                   pts, vertex_edge_scale, kernel_sigmas, test_ids):
    """Run comparison for multiple neighborhood sizes.

    neighborhoods: dict of {name: list_of_vert_tensors_per_tet}
    """
    device = pts.device
    print(f"\n{'#'*60}")
    print(f"  {label}")
    print(f"{'#'*60}")

    # Compute mean neighborhood size for each
    for name, verts_list in neighborhoods.items():
        mean_sz = np.mean([len(v) for v in verts_list])
        print(f"  {name}: mean {mean_sz:.1f} verts/tet")

    for sigma in kernel_sigmas:
        print(f"\n  sigma = {sigma}")

        # For each neighborhood, compute stats
        for name, verts_list in neighborhoods.items():
            total_w = []
            for t_idx in test_ids:
                t = t_idx.item()
                pt = eval_pts[t]
                ids = verts_list[t]
                w = eval_kernel(pt, ids, pts, vertex_edge_scale, sigma)
                total_w.append(w.sum().item())
            total_w = torch.tensor(total_w)

            # Compare to star (last entry in neighborhoods)
            star_name = list(neighborhoods.keys())[-1]
            if name == star_name:
                print(f"    {name:20s}: w_mean={total_w.mean():.4f}, "
                      f"w_std={total_w.std():.4f}")
            else:
                star_verts_list = neighborhoods[star_name]
                missed = []
                for i, t_idx in enumerate(test_ids):
                    t = t_idx.item()
                    pt = eval_pts[t]
                    star_ids = star_verts_list[t]
                    w_star = eval_kernel(pt, star_ids, pts, vertex_edge_scale, sigma)
                    s_star = w_star.sum().item()
                    if s_star > 1e-12:
                        missed.append(1.0 - total_w[i].item() / s_star)
                    else:
                        missed.append(0.0)
                missed = torch.tensor(missed)
                print(f"    {name:20s}: w_mean={total_w.mean():.4f}, "
                      f"missed: mean={missed.mean():.4f}, "
                      f"p95={missed.quantile(0.95):.4f}, "
                      f"max={missed.max():.4f}")


def build_shell1_verts(indices, flap_indices):
    """Build 1st-ring shell: own + flap."""
    T = indices.shape[0]
    device = indices.device
    idx_cpu = indices.cpu().numpy()
    flap_cpu = flap_indices.cpu().numpy()
    shell1 = []
    for t in range(T):
        vset = set(int(v) for v in idx_cpu[t])
        for j in range(4):
            fv = int(flap_cpu[t, j])
            if fv >= 0:
                vset.add(fv)
        shell1.append(torch.tensor(sorted(vset), dtype=torch.long, device=device))
    return shell1


def run_test(n_points=500, kernel_sigmas=[0.5, 1.0, 1.5, 2.0, 3.0], seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    # Generate a random point cloud and Delaunay triangulate
    pts = torch.randn(n_points, 3, device=device) * 2.0
    d = Delaunay(pts.cpu().numpy())
    indices = torch.tensor(d.simplices, dtype=torch.long, device=device)
    T = indices.shape[0]

    # Ensure positive volume
    v0 = pts[indices[:, 0]]
    v1 = pts[indices[:, 1]]
    v2 = pts[indices[:, 2]]
    v3 = pts[indices[:, 3]]
    vols = torch.einsum("bi,bi->b", v3 - v0, torch.cross(v1 - v0, v2 - v0, dim=1))
    flip = vols < 0
    if flip.any():
        indices[flip] = indices[flip][:, [1, 0, 2, 3]]

    # Build topology
    tet_adj = build_adj(pts, indices, device=device)
    flap_indices = compute_flap_indices(indices, tet_adj)

    # Build neighborhoods
    star_verts = build_tet_to_verts_star(indices, pts.shape[0])
    shell1_verts = build_shell1_verts(indices, flap_indices)
    shell2_verts = build_shell2_verts(indices, tet_adj, flap_indices)

    # Compute edge scales
    scale_mean = compute_vertex_edge_scale(pts, indices, mode="mean")
    scale_p25 = compute_vertex_edge_scale(pts, indices, mode="p25")

    # Compute circumcenters
    own_pos = pts[indices]  # (T, 4, 3)
    cc, R = calculate_circumcenters_torch(own_pos.double())
    cc = cc.float()
    cc_clipped = project_points_to_tetrahedra(cc, own_pos)

    # Stats
    displacement = torch.linalg.norm(cc - cc_clipped, dim=-1)
    n_outside = (displacement > 1e-6).sum().item()
    print(f"Mesh: {pts.shape[0]} vertices, {T} tets")
    print(f"Circumcenters outside tet: {n_outside}/{T} ({100*n_outside/T:.1f}%)")
    print(f"Displacement when clipped: mean={displacement.mean():.4f}, "
          f"median={displacement.median():.4f}, max={displacement.max():.4f}")

    # Scale stats
    ratio = scale_mean / scale_p25.clamp(min=1e-8)
    print(f"\nScale stats:")
    print(f"  Mean edge:  mean={scale_mean.mean():.4f}, std={scale_mean.std():.4f}")
    print(f"  P25 edge:   mean={scale_p25.mean():.4f}, std={scale_p25.std():.4f}")
    print(f"  Ratio (mean/p25): mean={ratio.mean():.4f}, std={ratio.std():.4f}")

    n_test = min(T, 2000)
    test_ids = torch.randperm(T)[:n_test]

    neighborhoods = {
        "shell-1 (8)": shell1_verts,
        "shell-2 (~14)": shell2_verts,
        "star (~35)": star_verts,
    }

    # Test all 4 combos: {raw, clipped} x {mean, p25}
    for cc_label, eval_pts in [("RAW CC", cc), ("CLIPPED CC", cc_clipped)]:
        for scale_label, scale in [("mean-edge scale", scale_mean), ("p25-edge scale", scale_p25)]:
            run_comparison(
                f"{cc_label} + {scale_label}",
                eval_pts, indices, flap_indices, neighborhoods,
                pts, scale, kernel_sigmas, test_ids,
            )


if __name__ == "__main__":
    run_test()
