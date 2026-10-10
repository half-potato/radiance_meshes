"""Benchmark: interval renderer vs interp renderer.

Synthetic scene of many small tets in the frustum; measures forward and
forward+backward wall time for both rasterizer paths at a realistic
resolution. Run with:  uv run python analysis/bench_interval.py
"""

import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from tests.interval_parity_test import FakeModel, make_camera, make_scene, AUX_DIM  # noqa: E402
from utils.train_util import render  # noqa: E402


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - st) / iters


def main(n_tets=200_000, hw=(1080, 1920), tile_size=8):
    torch.manual_seed(0)
    camera = make_camera(height=hw[0], width=hw[1])
    vertices, indices, cell_values, vertex_normals = make_scene(
        n_tets=n_tets, spread=2.0, z_range=(2.0, 8.0), tet_size=0.035)

    target = torch.rand((3, hw[0], hw[1]), device="cuda")

    for use_interval in (False, True):
        name = "interval" if use_interval else "interp  "

        v = vertices.detach().clone().requires_grad_(True)
        cv = cell_values.detach().clone().requires_grad_(True)
        vn = vertex_normals.detach().clone().requires_grad_(True)
        model = FakeModel(v, indices, vn)

        def fwd():
            with torch.no_grad():
                render(camera, model, cell_values=cv, tile_size=tile_size,
                       min_t=0.2, use_interval=use_interval)

        def fwd_bwd():
            for g in (v, cv, vn):
                g.grad = None
            pkg = render(camera, model, cell_values=cv, tile_size=tile_size,
                         min_t=0.2, use_interval=use_interval)
            loss = (pkg["render"] - target).abs().mean() + 0.05 * pkg["distortion_loss"]
            loss.backward()

        t_f = bench(fwd)
        t_fb = bench(fwd_bwd)
        print(f"{name}  forward: {t_f*1e3:8.2f} ms   forward+backward: {t_fb*1e3:8.2f} ms")

        pkg = render(camera, model, cell_values=cv, tile_size=tile_size,
                     min_t=0.2, use_interval=use_interval)
        print(f"{name}  mean alpha: {pkg['alpha'].mean().item():.4f}")


if __name__ == "__main__":
    main()
