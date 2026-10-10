"""Round-trip test for ``FrozenTetModel`` save → load.

Verifies the contract that a model written via ``state_dict()`` plus an
``Args``-shaped config (the format produced by ``train.py``) can be reloaded
via ``FrozenTetModel.load_ckpt`` and produces a bitwise-equivalent render.

This is round-trip-based, not snapshot-based: we render with the original
model, save, reload, render again, and assert the two renders match — the
test does not pin what those renders contain.

Requires a CUDA device (gated by tests/conftest.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from utils.args import Args
from utils.train_util import render

from _helpers import make_frozen_model, make_test_camera


def _looking_at(model):
    """Camera placed outside the tet, looking at its centroid (well-defined SH)."""
    center = model.interior_vertices.detach().mean(0)
    cam_pos = center + torch.tensor([0.0, 0.0, 0.8], device="cuda")
    forward = center - cam_pos
    forward = forward / forward.norm()
    right = torch.linalg.cross(forward, torch.tensor([0.0, 1.0, 0.0], device="cuda"))
    right = right / right.norm()
    up = torch.linalg.cross(right, forward)
    viewmat = torch.eye(4, device="cuda")
    viewmat[:3, 0] = right
    viewmat[:3, 1] = up
    viewmat[:3, 2] = forward
    viewmat[:3, 3] = cam_pos
    viewmat = torch.linalg.inv(viewmat)
    return make_test_camera(viewmat)


def _save_frozen(model, out_dir: Path, *, max_sh_deg=2, min_t=0.01):
    """Mirror train.py's save protocol: ``ckpt.pth`` + ``config.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = model.state_dict()
    sd["indices"] = model.indices
    sd["empty_indices"] = model.empty_indices
    torch.save(sd, out_dir / "ckpt.pth")

    cfg = Args()
    cfg.max_sh_deg = max_sh_deg
    cfg.chunk_size = model.chunk_size
    cfg.min_t = min_t
    with (out_dir / "config.json").open("w") as f:
        json.dump(cfg.as_dict(), f)


def test_frozen_model_save_load_render_roundtrip(tmp_path):
    """Render(original) ≡ Render(reloaded). The exact bytes don't matter."""
    model = make_frozen_model(seed=0)
    camera = _looking_at(model)

    with torch.no_grad():
        ref_pkg = render(camera, model, tile_size=16, min_t=0.01)
        ref_rgb = ref_pkg["render"].detach().clone()

    _save_frozen(model, tmp_path)

    from models.frozen import FrozenTetModel
    reloaded = FrozenTetModel.load_ckpt(tmp_path, device="cuda")

    with torch.no_grad():
        rel_pkg = render(camera, reloaded, tile_size=16, min_t=0.01)
        rel_rgb = rel_pkg["render"].detach()

    # Same tensor in, same tensor out -- no claim on what the render contains,
    # only that load_ckpt is invertible against the save protocol used at train.py:453.
    assert torch.allclose(ref_rgb, rel_rgb, atol=1e-6, rtol=1e-5), (
        f"render mismatch after roundtrip: max diff "
        f"{(ref_rgb - rel_rgb).abs().max().item():.3e}"
    )


def test_frozen_model_state_dict_keys_stable():
    """The set of keys we save must include everything load_ckpt reads.

    A regression that adds/renames a parameter without updating both sides
    leaves the saved checkpoint silently incomplete -- load_state_dict reports
    missing-key warnings that are easy to overlook on CI.
    """
    model = make_frozen_model(seed=0)
    sd = dict(model.state_dict())
    sd["indices"] = model.indices
    sd["empty_indices"] = model.empty_indices

    required = {
        "interior_vertices", "ext_vertices",
        "interior_vertex_normals", "ext_vertex_normals",
        "indices", "empty_indices",
        "density", "rgb", "gradient", "sh",
        "center", "scene_scaling",
    }
    missing = required - set(sd.keys())
    assert not missing, f"FrozenTetModel state dict missing keys: {missing}"
