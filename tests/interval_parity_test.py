"""Parity tests: interval renderer vs interp renderer.

The interval renderer (`use_interval=True`) replaces per-pixel ray-tet
intersection with a screen-space fan-triangle interpolation of baked
(z_front, z_back, normal, color) intervals. For tets that are small relative
to their distance from the camera, the screen-space linear interpolation of
camera-space z is a close approximation of the exact intersection depths, so
the two renderers should agree closely on color/alpha/normals and produce
matching gradients.

These tests build small synthetic scenes directly (no dataset, no JAX).
"""

import math

import numpy as np
import pytest
import torch

torch.manual_seed(20260611)
np.random.seed(20260611)

DEVICE = "cuda"
AUX_DIM = 2


def make_camera(height=64, width=64, fov_degrees=60.0):
    from data.camera import Camera

    fovx = math.radians(fov_degrees)
    fovy = 2 * math.atan(math.tan(fovx / 2) * height / width)
    image = torch.zeros((3, height, width), device=DEVICE)
    return Camera(
        colmap_id=0,
        R=np.eye(3),
        T=np.zeros(3),
        fovx=fovx,
        fovy=fovy,
        image=image,
        gt_alpha_mask=None,
        uid=0,
        cx=-1,
        cy=-1,
        data_device=DEVICE,
    )


def make_scene(n_tets=40, spread=1.2, z_range=(2.5, 4.5), tet_size=0.25,
               density_scale=12.0, grd_scale=0.4, requires_grad=False):
    """Random small tets fully in front of the camera (+z)."""
    centers = torch.rand((n_tets, 3), device=DEVICE)
    centers[:, 0] = (centers[:, 0] * 2 - 1) * spread
    centers[:, 1] = (centers[:, 1] * 2 - 1) * spread
    centers[:, 2] = centers[:, 2] * (z_range[1] - z_range[0]) + z_range[0]

    offsets = (torch.rand((n_tets, 4, 3), device=DEVICE) * 2 - 1) * tet_size
    vertices = (centers[:, None, :] + offsets).reshape(-1, 3)
    indices = torch.arange(n_tets * 4, device=DEVICE, dtype=torch.int32).reshape(n_tets, 4)

    cell_values = torch.zeros((n_tets, 7 + AUX_DIM), device=DEVICE)
    cell_values[:, 0] = torch.rand(n_tets, device=DEVICE) * density_scale
    cell_values[:, 1:4] = torch.rand((n_tets, 3), device=DEVICE)
    cell_values[:, 4:7] = (torch.rand((n_tets, 3), device=DEVICE) * 2 - 1) * grd_scale
    cell_values[:, 7:] = torch.rand((n_tets, AUX_DIM), device=DEVICE)

    vertex_normals = torch.nn.functional.normalize(
        torch.randn_like(vertices), dim=-1)

    if requires_grad:
        vertices.requires_grad_(True)
        cell_values.requires_grad_(True)
        vertex_normals.requires_grad_(True)

    return vertices, indices, cell_values, vertex_normals


class FakeModel:
    """Duck-typed stand-in for models.ingp_color.Model used by render()."""

    def __init__(self, vertices, indices, vertex_normals, aux_dim=AUX_DIM):
        self.vertices = vertices
        self.indices = indices
        self.vertex_normals = vertex_normals
        self.additional_attr = aux_dim
        self.device = DEVICE
        self.mask_values = False
        self.feature_dim = 7 + aux_dim


def render_both(vertices, indices, cell_values, vertex_normals, camera,
                tile_size=8, min_t=0.2):
    from utils.train_util import render

    model = FakeModel(vertices, indices, vertex_normals)
    pkg_interp = render(camera, model, cell_values=cell_values,
                        tile_size=tile_size, min_t=min_t, use_interval=False)
    pkg_interval = render(camera, model, cell_values=cell_values,
                          tile_size=tile_size, min_t=min_t, use_interval=True)
    return pkg_interp, pkg_interval


def _report(name, a, b):
    diff = (a - b).abs()
    return (f"{name}: max={diff.max().item():.4f} mean={diff.mean().item():.5f} "
            f"ref_mean={a.abs().mean().item():.4f}")


def test_forward_parity():
    """Interval forward render should closely match the interp renderer."""
    camera = make_camera()
    vertices, indices, cell_values, vertex_normals = make_scene()
    pkg_interp, pkg_interval = render_both(vertices, indices, cell_values, vertex_normals, camera)

    rgb_a, rgb_b = pkg_interp["render"], pkg_interval["render"]
    alpha_a, alpha_b = pkg_interp["alpha"], pkg_interval["alpha"]
    aux_a, aux_b = pkg_interp["aux"], pkg_interval["aux"]
    xyzd_a, xyzd_b = pkg_interp["xyzd"], pkg_interval["xyzd"]

    msgs = [
        _report("rgb", rgb_a, rgb_b),
        _report("alpha", alpha_a, alpha_b),
        _report("aux", aux_a, aux_b),
        _report("normals", xyzd_a[..., :3], xyzd_b[..., :3]),
    ]
    print("\n".join(msgs))

    assert torch.isfinite(rgb_b).all()
    # Mean error small; max error is permitted to be larger at silhouette
    # boundaries where screen-space interpolation differs from exact
    # intersection by design.
    assert (rgb_a - rgb_b).abs().mean() < 5e-3, msgs[0]
    assert (alpha_a - alpha_b).abs().mean() < 5e-3, msgs[1]
    assert (aux_a - aux_b).abs().mean() < 5e-3, msgs[2]
    assert (xyzd_a[..., :3] - xyzd_b[..., :3]).abs().mean() < 5e-3, msgs[3]
    # Distortion loss uses the ray-distance parameterization; matching values
    # confirm the interval path's z->t conversion (ray_scale, min_t offset).
    da = pkg_interp["distortion_loss"].item()
    db = pkg_interval["distortion_loss"].item()
    assert abs(da - db) < 0.1 * max(abs(da), 1e-6), f"distortion: {da} vs {db}"
    # The scene must actually contain content for this test to mean anything.
    assert alpha_a.mean() > 0.05
    assert alpha_b.mean() > 0.01


def test_forward_parity_single_tet_both_cases():
    """One tet viewed from angles that exercise both silhouette cases."""
    camera = make_camera(height=48, width=48)
    base = torch.tensor([
        [0.0, 0.0, 3.0],
        [0.4, 0.0, 3.1],
        [0.0, 0.4, 3.2],
        [0.1, 0.1, 3.5],
    ], device=DEVICE)
    indices = torch.tensor([[0, 1, 2, 3]], device=DEVICE, dtype=torch.int32)
    cell_values = torch.zeros((1, 7 + AUX_DIM), device=DEVICE)
    cell_values[0, 0] = 8.0
    cell_values[0, 1:4] = torch.tensor([0.7, 0.3, 0.2], device=DEVICE)
    cell_values[0, 4:7] = torch.tensor([0.2, -0.3, 0.25], device=DEVICE)
    cell_values[0, 7:] = 0.5

    for trial in range(6):
        rot = torch.linalg.qr(torch.randn(3, 3, device=DEVICE))[0]
        if torch.det(rot) < 0:
            rot[:, 0] = -rot[:, 0]
        center = base.mean(dim=0, keepdim=True)
        vertices = ((base - center) @ rot.T + center).contiguous()
        vertex_normals = torch.nn.functional.normalize(torch.randn_like(vertices), dim=-1)

        pkg_interp, pkg_interval = render_both(
            vertices, indices, cell_values, vertex_normals, camera)
        a, b = pkg_interp["render"], pkg_interval["render"]
        msg = _report(f"trial{trial} rgb", a, b)
        print(msg)
        assert (a - b).abs().mean() < 5e-3, msg


def test_gradient_parity():
    """Gradients through the interval path should match interp gradients."""
    camera = make_camera()
    vertices, indices, cell_values, vertex_normals = make_scene()
    target = torch.rand((3, camera.image_height, camera.image_width), device=DEVICE)

    grads = {}
    for use_interval in (False, True):
        from utils.train_util import render

        v = vertices.detach().clone().requires_grad_(True)
        cv = cell_values.detach().clone().requires_grad_(True)
        vn = vertex_normals.detach().clone().requires_grad_(True)
        model = FakeModel(v, indices, vn)
        pkg = render(camera, model, cell_values=cv, tile_size=8, min_t=0.2,
                     use_interval=use_interval)
        loss = (pkg["render"] - target).abs().mean() \
            + 0.1 * pkg["aux"].mean() \
            + 0.1 * pkg["xyzd"][..., :3].abs().mean()
        loss.backward()
        grads[use_interval] = (v.grad.clone(), cv.grad.clone(), vn.grad.clone())

    names = ["vertices", "cell_values", "vertex_normals"]
    for name, g_interp, g_interval in zip(names, grads[False], grads[True]):
        assert torch.isfinite(g_interval).all(), f"{name}: non-finite interval grad"
        assert g_interval.abs().sum() > 0, f"{name}: interval grad is identically zero"
        cos = torch.nn.functional.cosine_similarity(
            g_interp.reshape(1, -1), g_interval.reshape(1, -1)).item()
        rel_mag = (g_interval.norm() / g_interp.norm().clamp(min=1e-12)).item()
        print(f"{name}: cos={cos:.4f} |interval|/|interp|={rel_mag:.3f}")
        # Vertices gradients differ more (screen-space approximation moves
        # silhouettes differently than exact intersection), so use a looser
        # bound there.
        min_cos = 0.7 if name == "vertices" else 0.9
        assert cos > min_cos, f"{name}: gradient direction mismatch (cos={cos:.4f})"
        assert 0.3 < rel_mag < 3.0, f"{name}: gradient magnitude mismatch ({rel_mag:.3f})"


def test_gradient_channel_isolation():
    """A loss on rgb only must not produce aux grads, and vice versa."""
    from utils.train_util import render

    camera = make_camera(height=48, width=48)
    vertices, indices, cell_values, vertex_normals = make_scene(n_tets=20)

    cv = cell_values.detach().clone().requires_grad_(True)
    model = FakeModel(vertices, indices, vertex_normals)
    pkg = render(camera, model, cell_values=cv, tile_size=8, min_t=0.2,
                 use_interval=True)
    pkg["render"].sum().backward()
    # rgb loss: gradient lands in density/color/grd, not aux channels
    assert cv.grad[:, 1:4].abs().sum() > 0, "no color gradient"
    assert cv.grad[:, 7:].abs().sum() == 0, "rgb loss leaked into aux channels"

    cv2 = cell_values.detach().clone().requires_grad_(True)
    pkg2 = render(camera, model, cell_values=cv2, tile_size=8, min_t=0.2,
                  use_interval=True)
    pkg2["aux"].sum().backward()
    assert cv2.grad[:, 7:].abs().sum() > 0, "no aux gradient"
    assert cv2.grad[:, 1:4].abs().sum() == 0, "aux loss leaked into color channels"


def test_degenerate_inputs_no_crash():
    """Behind-camera and zero-density tets should render to empty, not crash."""
    from utils.train_util import render

    camera = make_camera(height=32, width=32)
    # One tet behind the camera, one degenerate (all verts identical), one OK.
    vertices = torch.tensor([
        [0.0, 0.0, -3.0], [0.3, 0.0, -3.0], [0.0, 0.3, -3.0], [0.1, 0.1, -3.3],
        [0.5, 0.5, 2.0], [0.5, 0.5, 2.0], [0.5, 0.5, 2.0], [0.5, 0.5, 2.0],
        [0.0, 0.0, 3.0], [0.3, 0.0, 3.0], [0.0, 0.3, 3.0], [0.1, 0.1, 3.3],
    ], device=DEVICE)
    indices = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        device=DEVICE, dtype=torch.int32)
    cell_values = torch.zeros((3, 7 + AUX_DIM), device=DEVICE)
    cell_values[:, 0] = 10.0
    cell_values[:, 1:4] = 0.5
    vertex_normals = torch.zeros_like(vertices)

    model = FakeModel(vertices, indices, vertex_normals)
    pkg = render(camera, model, cell_values=cell_values, tile_size=8,
                 min_t=0.2, use_interval=True)
    assert torch.isfinite(pkg["render"]).all()
    assert pkg["alpha"].max() > 0  # the valid tet still renders


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
