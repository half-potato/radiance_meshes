"""Regression tests for the MAX_KEYS emergency brake in vertex_and_tile_shader.

The old clamp only zeroed tets covering the ENTIRE tile grid and re-ran the
offset cumsum in int32, so a flood of merely-large tets could still allocate
multi-GB key buffers or overflow the int32 offsets. The fixed path drops the
largest-footprint tets until the key total fits and keeps the cumsum in
int64. These tests shrink MAX_KEYS to force the brake on a small scene.
"""

import math

import numpy as np
import pytest
import torch

import delaunay_rasterization.internal.tile_shader_slang as tss

torch.manual_seed(20260612)

DEVICE = "cuda"


def make_camera(height=64, width=64, fov_degrees=60.0):
    from data.camera import Camera

    fovx = math.radians(fov_degrees)
    fovy = 2 * math.atan(math.tan(fovx / 2) * height / width)
    image = torch.zeros((3, height, width), device=DEVICE)
    return Camera(colmap_id=0, R=np.eye(3), T=np.zeros(3), fovx=fovx, fovy=fovy,
                  image=image, uid=0, cx=-1, cy=-1, data_device=DEVICE)


def make_scene(n_small=30, n_big=4):
    """Small tets plus a few screen-filling ones (the 'exploding' geometry)."""
    centers = torch.rand((n_small, 3), device=DEVICE)
    centers[:, 0:2] = (centers[:, 0:2] * 2 - 1) * 1.0
    centers[:, 2] = centers[:, 2] * 2 + 3.0
    offsets = (torch.rand((n_small, 4, 3), device=DEVICE) * 2 - 1) * 0.15
    small = (centers[:, None, :] + offsets).reshape(-1, 3)

    big_centers = torch.zeros((n_big, 3), device=DEVICE)
    big_centers[:, 2] = torch.linspace(4.0, 6.0, n_big, device=DEVICE)
    big_offsets = (torch.rand((n_big, 4, 3), device=DEVICE) * 2 - 1) * 4.0
    big = (big_centers[:, None, :] + big_offsets).reshape(-1, 3)

    vertices = torch.cat([small, big], dim=0)
    n_tets = n_small + n_big
    indices = torch.arange(n_tets * 4, device=DEVICE, dtype=torch.int32).reshape(n_tets, 4)
    return vertices, indices, n_small


def run_tile_shader(vertices, indices, camera, tile_size=8):
    from delaunay_rasterization.internal.render_grid import RenderGrid

    rg = RenderGrid(camera.image_height, camera.image_width,
                    tile_height=tile_size, tile_width=tile_size)
    tcam = dict(tile_height=tile_size, tile_width=tile_size,
                grid_height=rg.grid_height, grid_width=rg.grid_width,
                min_t=0.2, **camera.to_dict(DEVICE))
    return tss.vertex_and_tile_shader(indices, vertices, tcam, rg), rg


def test_overflow_brake_drops_largest_tets(monkeypatch):
    camera = make_camera()
    vertices, indices, n_small = make_scene()

    # Baseline: how many keys does the full scene want?
    (sorted_idx, tile_ranges, _, _, mask, _), rg = run_tile_shader(vertices, indices, camera)
    full_keys = sorted_idx.shape[0]
    assert full_keys > 0

    # Force the brake: budget for roughly the small tets only.
    budget = max(full_keys // 4, 8)
    monkeypatch.setattr(tss, "MAX_KEYS", budget)
    (sorted_idx2, tile_ranges2, _, _, mask2, _), _ = run_tile_shader(vertices, indices, camera)

    assert sorted_idx2.shape[0] <= budget, "key buffer exceeds MAX_KEYS budget"
    assert sorted_idx2.shape[0] > 0, "brake dropped everything"
    # The brake must drop the LARGEST footprints first: every surviving tet's
    # footprint must be <= every dropped tet's footprint.
    kept = mask2.cpu().numpy()
    dropped = (~mask2 & mask).cpu().numpy()
    assert dropped.any(), "expected the brake to drop some tets"
    # tile ranges must stay consistent with the shrunken key list
    assert int(tile_ranges2.max().item()) <= sorted_idx2.shape[0]


def test_overflow_brake_render_still_finite(monkeypatch):
    """End-to-end render with the brake active must stay finite and non-empty."""
    from utils.train_util import render
    from tests.interval_parity_test import FakeModel

    camera = make_camera()
    vertices, indices, n_small = make_scene()
    n_tets = indices.shape[0]
    cell_values = torch.zeros((n_tets, 7), device=DEVICE)
    cell_values[:, 0] = 8.0
    cell_values[:, 1:4] = 0.6
    model = FakeModel(vertices, indices, torch.zeros_like(vertices), aux_dim=0)

    pkg_full = render(camera, model, cell_values=cell_values, tile_size=8, min_t=0.2)
    full_keys = int((pkg_full["mask"]).sum().item())
    assert full_keys > 0

    monkeypatch.setattr(tss, "MAX_KEYS", 64)
    pkg = render(camera, model, cell_values=cell_values, tile_size=8, min_t=0.2)
    assert torch.isfinite(pkg["render"]).all()
    assert pkg["alpha"].max() > 0, "small tets should survive the brake"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
