"""Shared helpers for both the radiance_meshes subproject tests and the top-level
dsplat tests/ suite. Both conftests put this directory on sys.path so the helpers
import via plain ``from _helpers import ...``.

Kept minimal: just a tiny camera builder and a tiny FrozenTetModel factory.
Bigger synthetic scenes (Lambert sphere, recoverable_scene) live with the tests
that consume them so the dependencies stay near the assertions.
"""

from __future__ import annotations

import numpy as np
import torch

from data.camera import Camera
from models.frozen import FrozenTetModel
from utils.compare_quad import setup_camera


def make_test_camera(viewmat, height=32, width=32, fov=90):
    """Construct a Camera looking through ``viewmat`` (4x4 world->camera).

    Mirrors the camera construction inside ``utils.compare_quad`` and the
    original ``_make_camera`` in ``freeze_gradient_test.py``.
    """
    viewmat, _, _, fovy, fovx, fx, fy = setup_camera(height, width, fov, viewmat)
    R = viewmat[:3, :3].T
    T = viewmat[:3, 3]
    blank = torch.zeros((3, height, width), device="cuda")
    return Camera(
        colmap_id=0,
        R=R.cpu().numpy(),
        T=T.cpu().numpy(),
        fovx=fovx,
        fovy=fovy,
        image=blank,
        gt_alpha_mask=None,
        uid=0,
        cx=-1,
        cy=-1,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    )


def make_frozen_model(seed=0, max_sh_deg=2):
    """A single dense tetrahedron with camera-visible density.

    Reused for the freeze-gradient diagnostic and any test that just needs a
    tiny renderable model. Deterministic in ``seed``.
    """
    torch.manual_seed(seed)
    verts = ((torch.rand((4, 3)) * 2 - 1) * 0.2).cuda()
    ext_vertices = torch.zeros((0, 3), device="cuda")
    indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device="cuda")
    empty_indices = torch.zeros((0, 4), dtype=torch.int32, device="cuda")

    sh_dim = (max_sh_deg + 1) ** 2 - 1
    density = torch.full((1, 1), 10.0, device="cuda")
    rgb = torch.tensor([[0.25, 0.5, 0.1]], device="cuda")
    gradient = torch.tensor([[-0.15, 0.25, 0.1]], device="cuda")
    sh = 0.1 * torch.randn(1, sh_dim, 3, device="cuda")
    center = verts.mean(0, keepdim=True)

    model = FrozenTetModel(
        int_vertices=verts,
        ext_vertices=ext_vertices,
        indices=indices,
        empty_indices=empty_indices,
        density=density,
        rgb=rgb,
        gradient=gradient,
        sh=sh,
        center=center,
        scene_scaling=1.0,
        max_sh_deg=max_sh_deg,
    )
    model.min_t = 0.01
    return model
