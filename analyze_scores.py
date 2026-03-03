"""Quick analysis of edge score distribution for threshold-based decimation."""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')
from utils.decimation import build_edge_list, query_tet_rgb, compute_edge_scores
from models.ingp_color import Model

ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "output/bicycle_ifimages_4_i4000_lel0.01_dc20000_ds3000_di1000_di200_nl0.0_v7"

device = torch.device("cuda")
model = Model.load_ckpt(Path(ckpt_dir), device)
print(f"Loaded model: {model.vertices.shape[0]} vertices, {model.indices.shape[0]} tets")

model.eval()
tet_rgb = query_tet_rgb(model)
edges = build_edge_list(model.indices)
n_interior = model.num_int_verts
scores = compute_edge_scores(edges, model.indices, model.vertices, tet_rgb, n_interior)

# Filter out inf (exterior)
finite_mask = scores != float('inf')
finite_scores = scores[finite_mask]

print(f"\nEdge score distribution ({finite_scores.shape[0]} interior edges):")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
vals = np.percentile(finite_scores.cpu().numpy(), percentiles)
for p, v in zip(percentiles, vals):
    print(f"  P{p:2d}: {v:.6f}")

print(f"\n  min: {finite_scores.min().item():.6f}")
print(f"  max: {finite_scores.max().item():.6f}")
print(f"  mean: {finite_scores.mean().item():.6f}")
print(f"  std: {finite_scores.std().item():.6f}")

# Show how many edges would be collapsed at various thresholds
print("\nThreshold analysis:")
for t in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
    n = (finite_scores < t).sum().item()
    pct = 100 * n / finite_scores.shape[0]
    print(f"  threshold={t:.3f}: {n:>8d} edges ({pct:5.1f}%)")
