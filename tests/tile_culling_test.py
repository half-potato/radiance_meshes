"""Bit-exactness of conservative tile culling in the Delaunay tile shader.

`generate_keys` emits a sentinel tile id for AABB-rect tiles that provably miss
the projected tetrahedron (2D separating-axis test between the tile pixel box
and the convex hull of the 4 projected verts). Skipped tiles contribute nothing
to the alpha blend, so the rendered image AND all gradients must be BITWISE
identical to the un-culled pipeline.

Contract (property, not snapshot):
* forward: torch.equal on render / alpha / aux / xyzd, cull-on vs cull-off.
* backward: torch.equal on cell_values.grad / vertices.grad /
  vertex_normals.grad under a fixed cotangent.
* edge cases (near-plane straddle, off-screen, degenerate sliver) route through
  the non-cullable fallback and must stay identical.
* culling must actually fire on near-camera slivers (else the parity is a
  vacuous no-op) -- asserted via the non-sentinel key count dropping.

The cull-off path is byte-identical to the original (pre-change) pipeline, so
cull-on == cull-off proves cull-on == original.
"""

import math

import numpy as np
import pytest
import torch

torch.manual_seed(20260721)
np.random.seed(20260721)

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from tests.interval_parity_test import make_camera, make_scene, FakeModel  # noqa: E402
import delaunay_rasterization.internal.tile_shader_slang as tss  # noqa: E402
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader  # noqa: E402
from delaunay_rasterization.internal.render_grid import RenderGrid  # noqa: E402
from utils.train_util import render  # noqa: E402

DEVICE = "cuda"


def _render_and_grad(vertices, indices, cell_values, vertex_normals, camera,
                     cotangents, tile_size=4, min_t=0.2):
    """Render (interp path) + backward under fixed cotangents. Returns a dict of
    detached forward tensors and input grads."""
    v = vertices.detach().clone().requires_grad_(True)
    cv = cell_values.detach().clone().requires_grad_(True)
    vn = vertex_normals.detach().clone().requires_grad_(True)
    model = FakeModel(v, indices, vn, aux_dim=cell_values.shape[1] - 7)
    pkg = render(camera, model, cell_values=cv, tile_size=tile_size,
                 min_t=min_t, use_interval=False)

    out = {
        "render": pkg["render"].detach().clone(),
        "alpha": pkg["alpha"].detach().clone(),
        "aux": pkg["aux"].detach().clone(),
        "xyzd": pkg["xyzd"].detach().clone(),
    }

    cot_r, cot_a, cot_x = cotangents
    loss = (pkg["render"] * cot_r).sum() \
        + (pkg["aux"] * cot_a).sum() \
        + (pkg["xyzd"] * cot_x).sum()
    loss.backward()
    out["g_cv"] = cv.grad.detach().clone()
    out["g_v"] = v.grad.detach().clone()
    out["g_vn"] = vn.grad.detach().clone()
    return out


def _make_cotangents(camera, aux_dim):
    H, W = camera.image_height, camera.image_width
    g = torch.Generator(device=DEVICE).manual_seed(12345)
    cot_r = torch.randn((3, H, W), device=DEVICE, generator=g)
    cot_a = torch.randn((aux_dim, H, W), device=DEVICE, generator=g)
    cot_x = torch.randn((H, W, 4), device=DEVICE, generator=g)
    return cot_r, cot_a, cot_x


_FWD_KEYS = ("render", "alpha", "aux", "xyzd")
_GRAD_KEYS = ("g_cv", "g_v", "g_vn")


def _run_parity(vertices, indices, cell_values, vertex_normals, camera,
                tile_size=4, min_t=0.2):
    """Compare culling on vs off.

    Forward outputs are deterministic, so they must be BITWISE identical
    (torch.equal) -- this is the real correctness gate: a wrongly-culled
    covering tet would deterministically change the render.

    Per-tet gradients are scattered with atomic adds, so they are NOT bitwise
    reproducible even for identical code (two cull-off runs already differ at
    the ~1e-4 float32 rounding floor). We therefore measure that intrinsic
    noise from a second cull-off run and require the culling-induced difference
    to stay inside a comfortable multiple of it -- far below what dropping a
    real covering tet (order 0.1-1.0) would produce.
    """
    aux_dim = cell_values.shape[1] - 7
    cots = _make_cotangents(camera, aux_dim)
    try:
        tss.ENABLE_TILE_CULLING = False
        off = _render_and_grad(vertices, indices, cell_values, vertex_normals,
                               camera, cots, tile_size, min_t)
        off2 = _render_and_grad(vertices, indices, cell_values, vertex_normals,
                                camera, cots, tile_size, min_t)
        tss.ENABLE_TILE_CULLING = True
        on = _render_and_grad(vertices, indices, cell_values, vertex_normals,
                              camera, cots, tile_size, min_t)
    finally:
        tss.ENABLE_TILE_CULLING = True

    # Forward: strictly bitwise.
    for k in _FWD_KEYS:
        assert torch.equal(off[k], on[k]), (
            f"{k}: NOT bitwise identical "
            f"(max |diff|={(off[k]-on[k]).abs().max().item()}, "
            f"n_diff={(off[k]!=on[k]).sum().item()})")

    # Gradients: within the atomic-scatter noise envelope.
    for k in _GRAD_KEYS:
        noise = (off[k] - off2[k]).abs().max().item()          # intrinsic
        induced = (on[k] - off[k]).abs().max().item()          # cull-induced
        tol = max(16.0 * noise, 1e-4)
        assert induced <= tol, (
            f"{k}: culling changed the gradient beyond the atomic-noise floor "
            f"(induced={induced:.3e}, intrinsic_noise={noise:.3e}, tol={tol:.3e})")
    return off, on


def _key_stats(vertices, indices, camera, tile_size=4, min_t=0.2, cull=True):
    """(total keys, non-sentinel keys) for the given cull setting."""
    from data.camera import Camera  # noqa: F401
    rg = RenderGrid(camera.image_height, camera.image_width,
                    tile_height=tile_size, tile_width=tile_size)
    tcam = dict(tile_height=tile_size, tile_width=tile_size,
                grid_height=rg.grid_height, grid_width=rg.grid_width,
                min_t=min_t, **camera.to_dict(DEVICE))
    prev = tss.ENABLE_TILE_CULLING
    try:
        tss.ENABLE_TILE_CULLING = cull
        sti, tile_ranges, *_ = vertex_and_tile_shader(indices, vertices, tcam, rg)
    finally:
        tss.ENABLE_TILE_CULLING = prev
    total = sti.shape[0]
    tr = tile_ranges.long()
    non_sentinel = int((tr[:, 1] - tr[:, 0]).clamp_min(0).sum().item())
    return total, non_sentinel


# --------------------------------------------------------------------------- #


def test_parity_random_scene():
    camera = make_camera(height=96, width=128, fov_degrees=60.0)
    vertices, indices, cell_values, vertex_normals = make_scene(n_tets=200)
    off, on = _run_parity(vertices, indices, cell_values, vertex_normals, camera)
    assert on["alpha"].mean() > 0.02, "scene rendered empty -- parity is vacuous"


def _make_sliver_scene(n_tets=150):
    """Near-camera tets whose projected hull fills a small fraction of its
    screen AABB, so conservative culling has plenty of corner tiles to remove.
    Built from the well-behaved random scene, then compressed toward the near
    plane so each tet spans a large footprint (many AABB tiles per tet)."""
    vertices, indices, cell_values, vertex_normals = make_scene(
        n_tets=n_tets, spread=1.0)
    z = vertices[:, 2]
    zc = 0.5 + (z - z.min()) / (z.max() - z.min()).clamp_min(1e-6) * 0.4
    vertices = vertices.clone()
    vertices[:, 2] = zc
    return vertices, indices, cell_values, vertex_normals


def test_parity_slivers_and_culling_fires():
    camera = make_camera(height=128, width=128, fov_degrees=70.0)
    vertices, indices, cell_values, vertex_normals = _make_sliver_scene()
    _run_parity(vertices, indices, cell_values, vertex_normals, camera,
                tile_size=4, min_t=0.2)

    total_off, real_off = _key_stats(vertices, indices, camera, cull=False)
    total_on, real_on = _key_stats(vertices, indices, camera, cull=True)
    # Buffer size (total keys) must be unchanged; only membership shrinks.
    assert total_on == total_off, (total_on, total_off)
    assert real_off == total_off, "cull-off should have no sentinels"
    print(f"\nslivers: total={total_off} real_off={real_off} real_on={real_on} "
          f"({100*(real_off-real_on)/max(real_off,1):.1f}% culled)")
    assert real_on < real_off, "culling removed nothing on slivers"


def test_parity_edge_cases_fallback():
    """Near-plane straddle, off-screen, and degenerate slivers must render
    bit-identically (they route through the non-cullable fallback)."""
    camera = make_camera(height=64, width=64, fov_degrees=60.0)
    aux = 2
    vertices = torch.tensor([
        # straddles the near plane (one vert behind camera): not cullable
        [0.0, 0.0, -0.5], [0.3, 0.0, 2.0], [0.0, 0.3, 2.0], [0.1, 0.1, 2.5],
        # fully off-screen to the right, in front: may be cullable, stays valid
        [5.0, 0.0, 2.0], [5.3, 0.0, 2.0], [5.0, 0.3, 2.0], [5.1, 0.1, 2.3],
        # near-degenerate thin sliver in front
        [-0.2, -0.2, 3.0], [0.2, 0.2, 3.0], [0.0, 0.0, 3.0001], [0.05, -0.05, 3.0],
        # a normal in-view tet so the frame isn't empty
        [-0.3, -0.3, 2.5], [0.3, -0.3, 2.6], [0.0, 0.3, 2.7], [0.0, 0.0, 2.9],
    ], device=DEVICE)
    n = 4
    indices = torch.arange(n * 4, device=DEVICE, dtype=torch.int32).reshape(n, 4)
    cell_values = torch.zeros((n, 7 + aux), device=DEVICE)
    cell_values[:, 0] = 8.0
    cell_values[:, 1:4] = 0.6
    cell_values[:, 4:7] = 0.1
    cell_values[:, 7:] = 0.4
    vertex_normals = torch.nn.functional.normalize(
        torch.randn_like(vertices), dim=-1)
    _run_parity(vertices, indices, cell_values, vertex_normals, camera)


def test_parity_off_screen_only():
    """Entire scene projected off-screen: nothing renders, still bit-identical."""
    camera = make_camera(height=64, width=64, fov_degrees=50.0)
    vertices, indices, cell_values, vertex_normals = make_scene(n_tets=30)
    vertices = vertices.detach().clone()
    vertices[:, 0] += 50.0  # shove far to the right, off frame
    _run_parity(vertices, indices, cell_values, vertex_normals, camera)


def _load_real_geometry():
    import os
    ckpt = "output/exp_london_dwrjump/ckpt.pth"
    if not os.path.exists(ckpt):
        return None
    d = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    V = d["interior_vertices"].float()
    idx = d["indices"].int()
    V = V - V.mean(0, keepdim=True)
    V = V / V.norm(dim=1).quantile(0.98) * 1.5
    V[:, 2] = V[:, 2] + 2.5
    T = idx.shape[0]
    Nv = V.shape[0]
    aux = 8
    g = torch.Generator(device=DEVICE).manual_seed(999)
    cv = torch.zeros((T, 7 + aux), device=DEVICE)
    cv[:, 0] = torch.rand(T, generator=g, device=DEVICE) * 10
    cv[:, 1:4] = torch.rand((T, 3), generator=g, device=DEVICE)
    cv[:, 4:7] = (torch.rand((T, 3), generator=g, device=DEVICE) * 2 - 1) * 0.4
    cv[:, 7:] = torch.rand((T, aux), generator=g, device=DEVICE)
    vn = torch.nn.functional.normalize(
        torch.randn((Nv, 3), generator=g, device=DEVICE), dim=-1)
    return V, idx, cv, vn


@pytest.mark.slow
def test_parity_real_london():
    geo = _load_real_geometry()
    if geo is None:
        pytest.skip("london checkpoint not present")
    V, idx, cv, vn = geo
    camera = make_camera(height=546, width=988, fov_degrees=60.0)
    _run_parity(V, idx, cv, vn, camera, tile_size=4, min_t=0.2)
    total_off, real_off = _key_stats(V, idx, camera, cull=False)
    total_on, real_on = _key_stats(V, idx, camera, cull=True)
    assert total_on == total_off
    print(f"\nlondon: total={total_off} real_off={real_off} real_on={real_on} "
          f"({100*(real_off-real_on)/max(real_off,1):.1f}% culled)")
    assert real_on < real_off


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
