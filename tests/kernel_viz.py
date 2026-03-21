"""
Shell Kernel Visualization — Full Shell Heatmap

Renders a piecewise-constant field over the full Delaunay triangulation, with a
selected "shell" (focus triangle + up to 3 edge-neighbors) shown in detail.

Layout:
  +------------------+----------+----------+
  |                  | Shell T0 | Shell T1 |
  | Full             |  heatmap |  heatmap |
  | Triangulation    +----------+----------+
  | Heatmap          | Shell T2 | Shell T3 |
  |                  |  heatmap |  heatmap |
  +------------------+----------+----------+
  | Time-series graph (full width)         |
  +----------------------------------------+

Output: tests/kernel_viz.mp4
"""

import numpy as np
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import gridspec
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Point configuration — two parallel lines, 10 vertices
# ──────────────────────────────────────────────────────────────────────

LINE_A = np.column_stack([np.linspace(0.3, 3.7, 5), np.full(5, 0.4)])
LINE_B = np.column_stack([np.linspace(0.6, 3.4, 5), np.full(5, 0.6)])
REST_POINTS = np.vstack([LINE_A, LINE_B]).astype(np.float64)
N_PTS = REST_POINTS.shape[0]

VALUES = np.sin(2.0 * REST_POINTS[:, 0]) + np.cos(3.0 * REST_POINTS[:, 1])

RNG = np.random.RandomState(42)
PHASES = RNG.uniform(0, 2 * np.pi, size=N_PTS)

KERNEL_SIGMA = 1.0
N_FRAMES = 120
GRID_RES = 300

# Orbit radius: fraction of mean edge length at rest
_tri0 = Delaunay(REST_POINTS)
_edges = set()
for _s in _tri0.simplices:
    for _i in range(3):
        for _j in range(_i + 1, 3):
            _edges.add((min(_s[_i], _s[_j]), max(_s[_i], _s[_j])))
_mean_edge = np.mean(
    [np.linalg.norm(REST_POINTS[a] - REST_POINTS[b]) for a, b in _edges]
)
ORBIT_R = 0.25 * _mean_edge

# ──────────────────────────────────────────────────────────────────────
# Perturbation
# ──────────────────────────────────────────────────────────────────────

def perturbed_points(t):
    angle = 2 * np.pi * t + PHASES
    dx = ORBIT_R * np.cos(angle)
    dy = ORBIT_R * np.sin(angle)
    return REST_POINTS + np.column_stack([dx, dy])

# ──────────────────────────────────────────────────────────────────────
# 2D geometry helpers
# ──────────────────────────────────────────────────────────────────────

def circumcenter_2d(tri_pts):
    ax, ay = tri_pts[0]
    bx, by = tri_pts[1]
    cx, cy = tri_pts[2]
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-12:
        cc = tri_pts.mean(axis=0)
        return cc, np.linalg.norm(cc - tri_pts[0])
    a2 = ax**2 + ay**2
    b2 = bx**2 + by**2
    c2 = cx**2 + cy**2
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / D
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / D
    cc = np.array([ux, uy])
    return cc, np.linalg.norm(cc - tri_pts[0])


def compute_vertex_edge_scale(points, simplices):
    n = points.shape[0]
    edge_sum = np.zeros(n)
    edge_count = np.zeros(n)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        a_idx = simplices[:, i]
        b_idx = simplices[:, j]
        elen = np.linalg.norm(points[a_idx] - points[b_idx], axis=1)
        np.add.at(edge_sum, a_idx, elen)
        np.add.at(edge_sum, b_idx, elen)
        np.add.at(edge_count, a_idx, 1)
        np.add.at(edge_count, b_idx, 1)
    return np.maximum(edge_sum / np.maximum(edge_count, 1), 1e-8)


def compute_flap_indices(simplices):
    T = simplices.shape[0]
    edge_to_tri = {}
    for t in range(T):
        for k in range(3):
            edge = tuple(sorted([simplices[t][(k + 1) % 3], simplices[t][(k + 2) % 3]]))
            edge_to_tri.setdefault(edge, []).append((t, k))
    flap = np.full((T, 3), -1, dtype=np.int64)
    for edge, entries in edge_to_tri.items():
        if len(entries) == 2:
            (t1, k1), (t2, k2) = entries
            flap[t1, k1] = simplices[t2][k2]
            flap[t2, k2] = simplices[t1][k1]
    return flap


def find_neighbor_tri_indices(simplices, tri_idx):
    T = simplices.shape[0]
    edge_to_tri = {}
    for t in range(T):
        for k in range(3):
            edge = tuple(sorted([simplices[t][(k + 1) % 3], simplices[t][(k + 2) % 3]]))
            edge_to_tri.setdefault(edge, []).append(t)
    neighbors = []
    s = simplices[tri_idx]
    for k in range(3):
        edge = tuple(sorted([s[(k + 1) % 3], s[(k + 2) % 3]]))
        for t in edge_to_tri.get(edge, []):
            if t != tri_idx:
                neighbors.append(t)
                break
    return neighbors

# ──────────────────────────────────────────────────────────────────────
# Per-triangle shell kernel evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_triangle(points, simplices, tri_idx, flap_indices, ves, sigma):
    own_idx = simplices[tri_idx]
    # Fix: sort own_idx consistently (ascending vertex index)
    sort_order = np.argsort(own_idx)
    own_idx = own_idx[sort_order]
    tri_id = tuple(int(v) for v in own_idx)

    own_pos = points[own_idx]
    own_val = VALUES[own_idx]

    cc, R = circumcenter_2d(own_pos)

    own_dist = np.linalg.norm(own_pos - cc[np.newaxis, :], axis=1)
    own_scale = ves[own_idx]
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    own_w = np.exp(-(own_dist / np.maximum(own_scale, 1e-8)) ** 2 * inv_2s2)

    flap_idx_list = []
    flap_pos_list = []
    flap_w_list = []
    flap_val_list = []
    # Reorder flap_indices to match sorted vertex order
    reordered_flap = flap_indices[tri_idx][sort_order]
    for k in range(3):
        fi = reordered_flap[k]
        if fi >= 0:
            fp = points[fi]
            fd = np.linalg.norm(fp - cc)
            fs = ves[fi]
            fw = np.exp(-(fd / max(fs, 1e-8)) ** 2 * inv_2s2)
            flap_idx_list.append(int(fi))
            flap_pos_list.append(fp)
            flap_w_list.append(fw)
            flap_val_list.append(VALUES[fi])

    all_w = np.array(list(own_w) + flap_w_list)
    all_v = np.array(list(own_val) + flap_val_list)
    output = (all_w * all_v).sum() / max(all_w.sum(), 1e-12)

    return dict(
        tri_id=tri_id,
        tri_idx=tri_idx,
        own_idx=[int(v) for v in own_idx],
        own_pos=own_pos,
        own_weights=own_w,
        own_values=own_val,
        flap_idx=flap_idx_list,
        flap_pos=flap_pos_list,
        flap_weights=flap_w_list,
        flap_values=flap_val_list,
        cc=cc,
        R=R,
        output=output,
    )

# ──────────────────────────────────────────────────────────────────────
# Heatmap pixel grid
# ──────────────────────────────────────────────────────────────────────

def build_pixel_grid(points, res=GRID_RES, margin=0.15):
    xmin, xmax = points[:, 0].min() - margin, points[:, 0].max() + margin
    ymin, ymax = points[:, 1].min() - margin, points[:, 1].max() + margin
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    gx, gy = np.meshgrid(xs, ys)
    query = np.column_stack([gx.ravel(), gy.ravel()])
    extent = [xmin, xmax, ymin, ymax]
    return query, gx.shape, extent


def build_field_image(tri, tri_outputs, query, grid_shape):
    simplex_idx = tri.find_simplex(query)
    field = np.full(query.shape[0], np.nan)
    for i, out in enumerate(tri_outputs):
        mask = simplex_idx == i
        field[mask] = out
    return field.reshape(grid_shape)


def build_masked_field(field_flat, simplex_idx, tri_idx, grid_shape):
    masked = np.full(field_flat.shape, np.nan)
    mask = simplex_idx == tri_idx
    masked[mask] = field_flat[mask]
    return masked.reshape(grid_shape)


def evaluate_gaussian_field_on_grid(query, shell_td, ves, sigma):
    """Evaluate the shell's Gaussian kernel at every query point.

    For each pixel, compute distances to the shell's own + flap vertices,
    apply Gaussian weights (using per-vertex edge scale), and return the
    weighted average — showing how the field varies spatially.
    """
    own_idx = shell_td["own_idx"]
    own_pos = shell_td["own_pos"]       # (3, 2)
    own_val = shell_td["own_values"]    # (3,)
    flap_pos_list = shell_td["flap_pos"]
    flap_val_list = shell_td["flap_values"]
    flap_idx_list = shell_td["flap_idx"]

    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    N = query.shape[0]

    # Own vertices: distances from each query point to each own vertex
    # query: (N, 2), own_pos: (3, 2) -> dists: (N, 3)
    own_dists = np.linalg.norm(query[:, None, :] - own_pos[None, :, :], axis=2)  # (N, 3)
    own_scales = np.array([ves[i] for i in own_idx])  # (3,)
    own_w = np.exp(-(own_dists / own_scales[None, :]) ** 2 * inv_2s2)  # (N, 3)

    all_w = own_w
    all_v = own_val[None, :].repeat(N, axis=0) if isinstance(own_val, np.ndarray) else np.tile(own_val, (N, 1))
    all_v = np.broadcast_to(own_val[None, :], (N, 3))

    # Flap vertices
    if flap_pos_list:
        flap_pos = np.array(flap_pos_list)  # (F, 2)
        flap_val = np.array(flap_val_list)  # (F,)
        flap_scales = np.array([ves[i] for i in flap_idx_list])  # (F,)
        flap_dists = np.linalg.norm(query[:, None, :] - flap_pos[None, :, :], axis=2)  # (N, F)
        flap_w = np.exp(-(flap_dists / flap_scales[None, :]) ** 2 * inv_2s2)  # (N, F)
        all_w = np.concatenate([all_w, flap_w], axis=1)
        all_v = np.concatenate([np.broadcast_to(own_val[None, :], (N, 3)),
                                np.broadcast_to(flap_val[None, :], (N, len(flap_val)))], axis=1)
    else:
        all_v = np.broadcast_to(own_val[None, :], (N, 3))

    w_sum = all_w.sum(axis=1, keepdims=True)
    w_sum = np.maximum(w_sum, 1e-12)
    field = (all_w * all_v).sum(axis=1) / w_sum.ravel()
    return field

# ──────────────────────────────────────────────────────────────────────
# Shell selection — most interior triangle
# ──────────────────────────────────────────────────────────────────────

def select_focus_triangle(points, simplices, all_tri_data, prev_tri_id, prev_centroid):
    """Select the focus triangle: most interior (centroid closest to CoM).
    If prev_tri_id still exists, keep it. Otherwise pick closest replacement."""
    com = points.mean(axis=0)

    # Check if previous focus triangle still exists
    if prev_tri_id is not None:
        for td in all_tri_data:
            if td["tri_id"] == prev_tri_id:
                return td

    # Previous focus disappeared — find closest replacement by centroid distance
    if prev_centroid is not None:
        best_td = None
        best_dist = np.inf
        for td in all_tri_data:
            centroid = td["own_pos"].mean(axis=0)
            d = np.linalg.norm(centroid - prev_centroid)
            if d < best_dist:
                best_dist = d
                best_td = td
        if best_td is not None:
            return best_td

    # First frame — pick most interior
    best_td = None
    best_dist = np.inf
    for td in all_tri_data:
        centroid = td["own_pos"].mean(axis=0)
        d = np.linalg.norm(centroid - com)
        if d < best_dist:
            best_dist = d
            best_td = td
    return best_td

# ──────────────────────────────────────────────────────────────────────
# Precompute all frames
# ──────────────────────────────────────────────────────────────────────

def compute_all_frames(sigma):
    frames = []
    prev_topo = None
    flip_frames = set()
    prev_focus_id = None
    prev_focus_centroid = None

    # Build pixel grid from rest positions (stable across frames)
    query, grid_shape, extent = build_pixel_grid(REST_POINTS)

    for fi in range(N_FRAMES):
        t_norm = fi / N_FRAMES
        pts = perturbed_points(t_norm)
        tri = Delaunay(pts)
        simplices = tri.simplices
        ves = compute_vertex_edge_scale(pts, simplices)
        flap = compute_flap_indices(simplices)

        # Evaluate ALL triangles
        all_tri_data = []
        tri_outputs = np.zeros(simplices.shape[0])
        for ti in range(simplices.shape[0]):
            td = evaluate_triangle(pts, simplices, ti, flap, ves, sigma)
            all_tri_data.append(td)
            tri_outputs[ti] = td["output"]

        # Build full field image
        simplex_idx = tri.find_simplex(query)
        field_flat = np.full(query.shape[0], np.nan)
        for i, out in enumerate(tri_outputs):
            mask = simplex_idx == i
            field_flat[mask] = out
        field_image = field_flat.reshape(grid_shape)

        # Detect topology changes
        cur_topo = frozenset(td["tri_id"] for td in all_tri_data)
        if prev_topo is not None and cur_topo != prev_topo:
            flip_frames.add(fi)
        prev_topo = cur_topo

        # Select focus triangle
        focus_td = select_focus_triangle(
            pts, simplices, all_tri_data, prev_focus_id, prev_focus_centroid
        )
        prev_focus_id = focus_td["tri_id"]
        prev_focus_centroid = focus_td["own_pos"].mean(axis=0)

        # Find shell: focus + up to 3 neighbors
        neighbor_idxs = find_neighbor_tri_indices(simplices, focus_td["tri_idx"])
        neighbor_tds = [all_tri_data[ni] for ni in neighbor_idxs]
        shell_tds = [focus_td] + neighbor_tds

        # Shell bounds: centered on focus circumcenter with fixed half-width
        cc = focus_td["cc"]
        shell_half = 0.7  # fixed half-width around circumcenter
        shell_bounds = [
            cc[0] - shell_half,
            cc[0] + shell_half,
            cc[1] - shell_half,
            cc[1] + shell_half,
        ]

        # Build per-pixel Gaussian field for each shell triangle
        shell_res = 150
        sx = np.linspace(shell_bounds[0], shell_bounds[1], shell_res)
        sy = np.linspace(shell_bounds[2], shell_bounds[3], shell_res)
        sgx, sgy = np.meshgrid(sx, sy)
        shell_query = np.column_stack([sgx.ravel(), sgy.ravel()])
        shell_grid_shape = sgx.shape

        shell_fields = []
        for std in shell_tds:
            gf = evaluate_gaussian_field_on_grid(shell_query, std, ves, sigma)
            shell_fields.append(gf.reshape(shell_grid_shape))
        # Pad to 4 panels
        while len(shell_fields) < 4:
            shell_fields.append(np.full(shell_grid_shape, np.nan))
        while len(shell_tds) < 4:
            shell_tds.append(None)

        frames.append(dict(
            points=pts,
            simplices=simplices,
            tri=tri,
            all_tri_data=all_tri_data,
            field_image=field_image,
            extent=extent,
            focus_td=focus_td,
            shell_tds=shell_tds,
            shell_fields=shell_fields,
            shell_bounds=shell_bounds,
            simplex_idx=simplex_idx,
            field_flat=field_flat,
        ))

    return frames, flip_frames, extent



# ──────────────────────────────────────────────────────────────────────
# Edge collection from triangulation
# ──────────────────────────────────────────────────────────────────────

def get_edge_segments(points, simplices):
    """Return list of line segments [(x0,y0),(x1,y1)] for all triangulation edges."""
    edges = set()
    for s in simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(s[i]), int(s[j])
                edges.add((min(a, b), max(a, b)))
    segments = []
    for a, b in edges:
        segments.append([points[a], points[b]])
    return segments

# ──────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────

FIELD_CMAP = "viridis"
WEIGHT_CMAP = plt.cm.YlOrRd

# Precompute global value range for consistent colormap
_all_vals_approx = np.sin(2.0 * np.linspace(-0.5, 4.5, 200)) + np.cos(3.0 * np.linspace(-0.1, 1.1, 200)[:, None])
VMIN = VALUES.min() - 0.3
VMAX = VALUES.max() + 0.3


def draw_full_heatmap(ax, frame):
    ax.clear()
    ext = frame["extent"]
    ax.imshow(frame["field_image"], origin="lower", extent=ext,
              cmap=FIELD_CMAP, vmin=VMIN, vmax=VMAX, aspect="auto",
              interpolation="nearest")

    # Triangulation edges
    segments = get_edge_segments(frame["points"], frame["simplices"])
    lc = LineCollection(segments, colors="white", linewidths=0.5, alpha=0.6)
    ax.add_collection(lc)

    # Highlight focus triangle edges (bold)
    focus_td = frame["focus_td"]
    focus_pts = focus_td["own_pos"]
    focus_segs = []
    for i in range(3):
        for j in range(i + 1, 3):
            focus_segs.append([focus_pts[i], focus_pts[j]])
    lc_focus = LineCollection(focus_segs, colors="red", linewidths=2.5, alpha=0.9)
    ax.add_collection(lc_focus)

    # Highlight neighbor edges (dashed)
    for std in frame["shell_tds"][1:]:
        if std is not None:
            n_pts = std["own_pos"]
            n_segs = []
            for i in range(3):
                for j in range(i + 1, 3):
                    n_segs.append([n_pts[i], n_pts[j]])
            lc_n = LineCollection(n_segs, colors="orange", linewidths=1.5,
                                  linestyles="dashed", alpha=0.8)
            ax.add_collection(lc_n)

    # Circumcenter of focus
    ax.plot(focus_td["cc"][0], focus_td["cc"][1], "*", color="red",
            markersize=10, markeredgecolor="black", markeredgewidth=0.5, zorder=10)

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_title("Full Triangulation Heatmap", fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_shell_panel(ax, shell_field, shell_td, shell_bounds, panel_idx):
    ax.clear()
    if shell_td is None:
        ax.set_visible(False)
        return
    ax.set_visible(True)

    sb = shell_bounds  # [xmin, xmax, ymin, ymax]
    ax.imshow(shell_field, origin="lower", extent=sb,
              cmap=FIELD_CMAP, vmin=VMIN, vmax=VMAX, aspect="auto",
              interpolation="bilinear")

    # Triangle edges (bold)
    own_pts = shell_td["own_pos"]
    for i in range(3):
        j = (i + 1) % 3
        ax.plot([own_pts[i, 0], own_pts[j, 0]], [own_pts[i, 1], own_pts[j, 1]],
                color="white", linewidth=2.0, alpha=0.9)

    # Collect weights for colormap
    all_w = list(shell_td["own_weights"]) + shell_td["flap_weights"]
    w_max = max(all_w) if all_w else 1.0
    w_min = min(all_w) if all_w else 0.0
    def w_color(w):
        if w_max - w_min < 1e-12:
            return WEIGHT_CMAP(0.5)
        return WEIGHT_CMAP((w - w_min) / (w_max - w_min))

    # Own vertices (circles)
    for k in range(3):
        pos = own_pts[k]
        w = shell_td["own_weights"][k]
        ax.plot(pos[0], pos[1], "o", color=w_color(w), markersize=10,
                markeredgecolor="black", markeredgewidth=1.0, zorder=5)
        ax.annotate(f"w={w:.2f}", (pos[0], pos[1]),
                    textcoords="offset points", xytext=(8, -6), fontsize=5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.8, ec="none"))

    # Flap vertices (squares)
    for k in range(len(shell_td["flap_idx"])):
        pos = shell_td["flap_pos"][k]
        w = shell_td["flap_weights"][k]
        ax.plot(pos[0], pos[1], "s", color=w_color(w), markersize=8,
                markeredgecolor="black", markeredgewidth=1.0, zorder=5)
        ax.annotate(f"w={w:.2f}", (pos[0], pos[1]),
                    textcoords="offset points", xytext=(8, -6), fontsize=5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.8, ec="none"))

    # Circumcenter (star) + circumcircle (dashed)
    ax.plot(shell_td["cc"][0], shell_td["cc"][1], "*", color="royalblue",
            markersize=10, markeredgecolor="black", markeredgewidth=0.5, zorder=10)
    circ = Circle(shell_td["cc"], shell_td["R"], fill=False, edgecolor="gray",
                  linestyle="--", linewidth=0.6, alpha=0.5)
    ax.add_patch(circ)

    # Use shared shell bounds
    ax.set_xlim(sb[0], sb[1])
    ax.set_ylim(sb[2], sb[3])
    ax.set_aspect("equal")

    tri_label = "-".join(str(v) for v in shell_td["own_idx"])
    role = "Focus" if panel_idx == 0 else f"Neighbor {panel_idx}"
    n_flap = len(shell_td["flap_idx"])
    ax.set_title(f"{role} [{tri_label}] out={shell_td['output']:.3f} (3+{n_flap})",
                 fontsize=7, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


SHELL_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


def draw_timeseries(ax, frames, flip_frames, frame_idx):
    ax.clear()

    # Get current shell tri_ids for coloring
    cur_shell_tds = frames[frame_idx]["shell_tds"]
    cur_shell_ids = []
    for std in cur_shell_tds:
        if std is not None:
            cur_shell_ids.append(std["tri_id"])
        else:
            cur_shell_ids.append(None)

    # For each shell slot, trace its output over time
    for slot in range(4):
        if slot >= len(cur_shell_ids) or cur_shell_ids[slot] is None:
            continue
        tid = cur_shell_ids[slot]
        fs_list = []
        vs_list = []
        for fi, frame in enumerate(frames):
            for std in frame["shell_tds"]:
                if std is not None and std["tri_id"] == tid:
                    fs_list.append(fi)
                    vs_list.append(std["output"])
                    break
        if fs_list:
            label_parts = "-".join(str(v) for v in tid)
            role = "Focus" if slot == 0 else f"N{slot}"
            ax.plot(fs_list, vs_list, "-", color=SHELL_COLORS[slot],
                    linewidth=1.5, label=f"{role} [{label_parts}]", alpha=0.8)

    for ff in flip_frames:
        ax.axvspan(ff - 0.5, ff + 0.5, color="red", alpha=0.15)

    ax.axvline(frame_idx, color="black", linewidth=1.5, linestyle="--", alpha=0.8)

    ax.set_xlim(-1, N_FRAMES)
    ax.set_xlabel("Frame", fontsize=8)
    ax.set_ylabel("Kernel output", fontsize=8)
    ax.legend(fontsize=7, loc="upper right", ncol=4)
    ax.set_title("Shell output over time (red = flip frame)", fontsize=9)
    ax.grid(True, alpha=0.3)

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def generate_video(sigma):
    output_path = Path(__file__).parent / f"kernel_viz_sigma{sigma}.mp4"
    print(f"\n{'='*60}")
    print(f"sigma={sigma}: Precomputing {N_FRAMES} frames ...")
    frames, flip_frames, extent = compute_all_frames(sigma)

    print(f"Flip frames: {sorted(flip_frames)}")
    print(f"Generating video -> {output_path}")

    fig = plt.figure(figsize=(14, 9), constrained_layout=False)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.06,
                        wspace=0.08, hspace=0.25)

    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.7])

    ax_full = fig.add_subplot(gs[0:2, 0:2])
    ax_s0 = fig.add_subplot(gs[0, 2])
    ax_s1 = fig.add_subplot(gs[0, 3])
    ax_s2 = fig.add_subplot(gs[1, 2])
    ax_s3 = fig.add_subplot(gs[1, 3])
    ax_ts = fig.add_subplot(gs[2, :])

    shell_axes = [ax_s0, ax_s1, ax_s2, ax_s3]
    title = fig.suptitle("", fontsize=12)

    def animate(frame_idx):
        if frame_idx % 10 == 0:
            print(f"  sigma={sigma} Frame {frame_idx}/{N_FRAMES}")

        frame = frames[frame_idx]

        draw_full_heatmap(ax_full, frame)

        for i, ax in enumerate(shell_axes):
            draw_shell_panel(ax, frame["shell_fields"][i], frame["shell_tds"][i],
                             frame["shell_bounds"], i)

        draw_timeseries(ax_ts, frames, flip_frames, frame_idx)

        flip_marker = "  *** FLIP ***" if frame_idx in flip_frames else ""
        title.set_text(
            f"Shell Kernel Heatmap — Frame {frame_idx}/{N_FRAMES}"
            f"  (sigma={sigma}){flip_marker}"
        )

    anim = FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000 // 24)
    writer = FFMpegWriter(fps=24, bitrate=3000)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)

    print(f"Done: {output_path}")
    print(f"Flip frames: {sorted(flip_frames)}")


def main():
    for sigma in [0.5, 1.0, 1.5, 2.0]:
        generate_video(sigma)


if __name__ == "__main__":
    main()
