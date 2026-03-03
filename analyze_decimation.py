#!/usr/bin/env python3
"""Analyze decimation benchmark results with normalized comparison."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('output/decimation_benchmark/results.json') as f:
    data = json.load(f)

curves = data['curves']

# Interpolate to common vertex counts
target_verts = np.array([99000, 98000, 97000, 96000, 94000, 92000, 90000, 88000, 86000])

interp_data = {}
for name, curve in curves.items():
    remaining = np.array([c[1] for c in curve])
    psnrs = np.array([c[2] for c in curve])
    remaining_flip = remaining[::-1]
    psnrs_flip = psnrs[::-1]
    interp_psnr = np.interp(target_verts, remaining_flip, psnrs_flip,
                             left=np.nan, right=np.nan)
    interp_data[name] = interp_psnr

# Print normalized table
print(f"{'Heuristic':30s}", end='')
for v in target_verts:
    print(f" | {v//1000:3d}K", end='')
print()
print('-' * 100)

for name in sorted(interp_data.keys()):
    row = f"{name:30s}"
    for p in interp_data[name]:
        if np.isnan(p):
            row += " |   N/A"
        else:
            row += f" | {p:5.2f}"
    print(row)

# Rankings at key vertex counts
for target, label in [(98000, "2%"), (96000, "4%"), (94000, "6%"),
                       (92000, "8%"), (90000, "10%")]:
    idx = np.where(target_verts == target)[0][0]
    vals = {name: interp_data[name][idx] for name in interp_data
            if not np.isnan(interp_data[name][idx])}
    if not vals:
        continue
    print(f"\n\nRanking at {target//1000}K vertices ({label} decimation):")
    for rank, (name, psnr) in enumerate(sorted(vals.items(), key=lambda x: -x[1]), 1):
        marker = " ***" if rank <= 3 else ""
        print(f"  {rank:2d}. {name:30s} {psnr:.4f} dB{marker}")

# --- Plots ---

def color_for(name):
    if name.startswith('rank_'):
        return '#9467bd'  # purple for combinations
    if name.startswith('len_over_'):
        return '#e377c2'  # pink for ratio scores
    if name.startswith('estar_'):
        return '#d62728'  # red for edge-star field stats
    if 'inv_' in name:
        return '#2ca02c'  # green for inverted (legacy)
    if name == 'edge_length':
        return '#1f77b4'  # blue for geometric
    if name == 'random':
        return '#ff7f0e'  # orange
    return '#7f7f7f'  # gray for other

# Plot 1: PSNR vs Remaining Vertices (highlighting best)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

ax = axes[0]
# Determine top performers at 94K for highlighting
idx_94 = np.where(target_verts == 94000)[0][0]
vals_94 = {name: interp_data[name][idx_94] for name in interp_data
           if not np.isnan(interp_data[name][idx_94])}
top5 = [n for n, _ in sorted(vals_94.items(), key=lambda x: -x[1])[:5]]
top5 += ['random']  # always show baseline

for name, curve in curves.items():
    remaining = [c[1] for c in curve]
    psnrs = [c[2] for c in curve]
    if name in top5:
        ax.plot(remaining, psnrs, 'o-', label=name, markersize=4,
                linewidth=2.5, color=color_for(name))
    else:
        ax.plot(remaining, psnrs, '-', label=name, markersize=2,
                linewidth=0.6, alpha=0.2, color=color_for(name))
ax.set_xlabel('Vertices Remaining', fontsize=11)
ax.set_ylabel('Mean Test PSNR (dB)', fontsize=11)
ax.set_title('Decimation: PSNR vs Remaining Vertices (top 5 highlighted)')
ax.legend(fontsize=6, loc='lower left', ncol=2)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()
ax.set_ylim(16.5, 21.2)

# Plot 2: Bar chart at 94K (most heuristics have data here)
ax = axes[1]
sorted_items = sorted(vals_94.items(), key=lambda x: x[1], reverse=True)
names = [x[0] for x in sorted_items]
psnrs_bar = [x[1] for x in sorted_items]
colors = [color_for(n) for n in names]
ax.barh(range(len(names)), psnrs_bar, color=colors)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('PSNR at 94K vertices (dB)')
ax.set_title('Ranking at 6% decimation (94K / 100K verts)\n'
             'purple=rank combo, pink=ratio, red=edge-star, blue=geometric, orange=random')
for i, v in enumerate(psnrs_bar):
    ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=7)

plt.tight_layout()
plt.savefig('output/decimation_benchmark/normalized_comparison.png', dpi=150)
print('\nSaved: output/decimation_benchmark/normalized_comparison.png')
