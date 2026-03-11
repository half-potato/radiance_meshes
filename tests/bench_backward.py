"""Benchmark backward pass at various tet counts and resolutions.

Usage:
    uv run python tests/bench_backward.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import numpy as np
from gdel3d import Del

from rmesh_wgpu import RMeshRenderer
from utils.topo_utils import calculate_circumcenters_torch
import torch


def make_scene(n_verts, sh_degree=0, width=512, height=512, seed=42):
    np.random.seed(seed)
    pts = np.random.randn(n_verts, 3).astype(np.float32) * 0.5
    # tri = Delaunay(pts)
    v = Del(pts.shape[0])
    indices_np, prev = v.compute(torch.as_tensor(pts).double())
    valid_mask = (indices_np >= 0) & (indices_np < pts.shape[0])
    indices_np = indices_np[valid_mask.all(axis=1)]
    # indices = tri.simplices.astype(np.int32)
    indices = indices_np.clone().numpy()
    n_tets = indices.shape[0]
    nc = (sh_degree + 1) ** 2

    vertices_np = pts.ravel().astype(np.float32)
    indices_np = indices.ravel().astype(np.uint32)
    sh_coeffs_np = np.random.randn(n_tets * nc * 3).astype(np.float32) * 0.5
    densities_np = (np.random.rand(n_tets).astype(np.float32) * 5 + 1.0)
    color_grads_np = np.random.randn(n_tets * 3).astype(np.float32) * 0.1

    tets_t = torch.from_numpy(pts[indices])
    cc, r = calculate_circumcenters_torch(tets_t.double())
    circumdata_np = torch.cat([cc.float(), (r.float() ** 2).unsqueeze(-1)], dim=-1).numpy().ravel().astype(np.float32)

    cam_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)

    fov = 1.0
    aspect = width / height
    znear, zfar = 0.1, 100.0
    f = 1.0 / np.tan(fov / 2.0)

    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = 1.0
    proj[3, 2] = -(zfar * znear) / (zfar - znear)

    view = np.eye(4, dtype=np.float32)
    view[2, 2] = -1.0
    view[3, 2] = cam_pos[2]

    vp = (view @ proj).astype(np.float32)
    inv_vp = np.linalg.inv(vp).astype(np.float32)

    renderer = RMeshRenderer(
        vertices_np, indices_np, sh_coeffs_np, densities_np,
        color_grads_np, circumdata_np, sh_degree, width, height,
    )

    return renderer, cam_pos, vp, inv_vp, vertices_np, sh_coeffs_np, densities_np, color_grads_np, n_tets


def bench(n_verts, width=512, height=512, warmup=3, iters=10):
    renderer, cam_pos, vp, inv_vp, vertices, sh_coeffs, densities, color_grads, n_tets = \
        make_scene(n_verts, width=width, height=height)

    # Forward (also sets up internal state for backward)
    image = renderer.forward(cam_pos, vp.ravel(), inv_vp.ravel())
    dl_d_image = np.random.randn(*image.shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        renderer.forward(cam_pos, vp.ravel(), inv_vp.ravel())
    for _ in range(warmup):
        renderer.backward(dl_d_image)

    # Benchmark forward
    fwd_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        renderer.forward(cam_pos, vp.ravel(), inv_vp.ravel())
        t1 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000)

    # Benchmark backward
    bwd_times = []
    for _ in range(iters):
        renderer.forward(cam_pos, vp.ravel(), inv_vp.ravel())
        t0 = time.perf_counter()
        renderer.backward(dl_d_image)
        t1 = time.perf_counter()
        bwd_times.append((t1 - t0) * 1000)

    fwd_med = np.median(fwd_times)
    bwd_med = np.median(bwd_times)
    ratio = bwd_med / fwd_med if fwd_med > 0 else float('inf')

    print(f"  verts={n_verts:>7d}  tets={n_tets:>7d}  "
          f"fwd={fwd_med:7.2f}ms  bwd={bwd_med:7.2f}ms  ratio={ratio:.2f}x")
    return fwd_med, bwd_med, ratio


if __name__ == "__main__":
    print(f"Backward pass benchmark @ 512x512")
    print("-" * 75)

    vert_counts = [20000, 50000, 100000, 200000, 500000, 1000000]
    for nv in vert_counts:
        try:
            bench(nv)
        except Exception as e:
            print(f"  verts={nv}: FAILED — {e}")
