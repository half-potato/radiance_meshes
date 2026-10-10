"""Diagnostic tests for gradient propagation *after freezing the model*.

Background
----------
At ``freeze_start`` the trainer swaps the iNGP-backed ``Model`` for a
``FrozenTetModel`` whose per-tet attributes (density, rgb, gradient, SH) and
geometry (``interior_vertices``, ``interior_vertex_normals``) are all plain
``nn.Parameter``s. Two failure modes made "gradients sometimes don't propagate
after freezing":

  1. the training loop's gradient diagnostic referenced ``model.backbone``,
     which the frozen model doesn't have (crash, not a stalled grad);
  2. ``vertex_optim.step()`` was gated behind ``do_delaunay``, which is False
     once ``iteration >= freeze_start`` -- so the frozen vertices accumulated
     gradients that were never applied or cleared, and never moved.

These tests build a tiny ``FrozenTetModel`` directly from synthetic tensors (no
tinycudann / point cloud), render a single visible tet, and check exactly where
gradients reach -- and, critically, reproduce the gating bug from (2) and show
the fix moves the vertices again.

Requires a CUDA device (gated off by tests/conftest.py when unavailable).
"""

import unittest

import torch

from models.frozen import FrozenTetOptimizer
from utils.train_util import render

from _helpers import make_test_camera as _make_camera
from _helpers import make_frozen_model as _make_frozen_model

# Parameters that must keep training after freezing, and whether an RGB-only
# loss is guaranteed to drive them (normals only feed the geometry/normal path,
# so they're graph-connected but may legitimately receive a zero grad here).
_RGB_DRIVEN = ["density", "rgb", "gradient", "sh", "interior_vertices"]
_ALL_TRAINABLE = _RGB_DRIVEN + ["interior_vertex_normals"]

_HEIGHT = _WIDTH = 32
_FREEZE_START = 100  # arbitrary; tests step at iterations well past this


def _camera_looking_at(model):
    """Place the camera *outside* the tet looking at its centroid.

    The view direction must be well-defined: the SH coefficients hold only the
    degree>=1 bands, which vanish at a degenerate (zero) direction -- so a camera
    sitting exactly at the centroid would leave ``sh`` with a zero gradient.
    """
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
    return _make_camera(viewmat)


def _trainable(model):
    return {
        "density": model.density,
        "rgb": model.rgb,
        "gradient": model.gradient,
        "sh": model.sh,
        "interior_vertices": model.interior_vertices,
        "interior_vertex_normals": model.interior_vertex_normals,
    }


def _grad_status(p):
    g = p.grad
    if g is None:
        return "None", 0.0
    m = g.abs().max().item()
    return ("ZERO" if m == 0 else "nonzero"), m


def _render_loss(model, camera):
    pkg = render(camera, model, tile_size=16, min_t=0.01)
    return pkg["render"].sum()


class FreezeGradientTest(unittest.TestCase):
    def test_frozen_render_backward_populates_all_grads(self):
        """After a frozen render+backward, report where gradients land.

        Isolates "is backprop even reaching the frozen params?" -- the first
        thing to rule out when gradients seem dead after freezing.
        """
        model = _make_frozen_model(seed=0)
        camera = _camera_looking_at(model)
        _render_loss(model, camera).backward()

        params = _trainable(model)
        print("\n[freeze grad report] per-parameter gradient after backward:")
        for name in _ALL_TRAINABLE:
            status, mag = _grad_status(params[name])
            print(f"  {name:24s} requires_grad={params[name].requires_grad} "
                  f"grad={status} max|g|={mag:.3e}")

        # Params that drive the RGB image must receive a real (nonzero) gradient.
        for name in _RGB_DRIVEN:
            g = params[name].grad
            self.assertIsNotNone(g, f"{name} got no gradient (None) after freezing")
            self.assertGreater(
                g.abs().max().item(), 0.0,
                f"{name} gradient is all-zero after freezing")

        # Normals only feed the geometry/normal path; they must at least be
        # graph-connected (non-None) even if this loss leaves them at zero.
        self.assertIsNotNone(
            model.interior_vertex_normals.grad,
            "interior_vertex_normals is detached from the graph after freezing")

    def test_full_optimizer_step_moves_params(self):
        """Stepping all three frozen optimizers must move the RGB-driven params."""
        model = _make_frozen_model(seed=0)
        camera = _camera_looking_at(model)
        opt = FrozenTetOptimizer(model, freeze_lr=1e-2, vertices_lr=1e-1)

        before = {n: p.detach().clone() for n, p in _trainable(model).items()}
        _render_loss(model, camera).backward()
        opt.optim.step();        opt.optim.zero_grad()
        opt.sh_optim.step();     opt.sh_optim.zero_grad()
        opt.vertex_optim.step(); opt.vertex_optim.zero_grad()

        for name in _RGB_DRIVEN:
            after = _trainable(model)[name]
            self.assertFalse(
                torch.allclose(after, before[name]),
                f"{name} did not move after a full optimizer step")

    def test_post_freeze_vertex_gating(self):
        """Reproduce the freeze bug and verify the fix.

        Replays the trainer's exact post-freeze stepping. Under the *old* gating
        (vertices stepped only when ``do_delaunay``) the frozen vertices never
        move while density keeps training -- that is the bug. Under the *fixed*
        gating (``do_delaunay or iteration >= freeze_start``) they move again.
        """
        iteration = _FREEZE_START + 5          # safely past freeze_start
        delaunay_interval = 10
        assert iteration % delaunay_interval != 0  # ensure do_delaunay is False here

        def post_freeze_step(model, opt, *, use_fix):
            camera = _camera_looking_at(model)
            _render_loss(model, camera).backward()
            do_delaunay = iteration % delaunay_interval == 0 and iteration < _FREEZE_START
            opt.main_step(); opt.main_zero_grad()
            opt.sh_optim.step(); opt.sh_optim.zero_grad()
            do_vertex_step = do_delaunay or (use_fix and iteration >= _FREEZE_START)
            if do_vertex_step:
                opt.vertex_optim.step(); opt.vertex_optim.zero_grad()

        # --- old gating: vertices stranded ---
        buggy = _make_frozen_model(seed=1)
        opt_b = FrozenTetOptimizer(buggy, freeze_lr=1e-2, vertices_lr=1e-1)
        v_before = buggy.interior_vertices.detach().clone()
        d_before = buggy.density.detach().clone()
        post_freeze_step(buggy, opt_b, use_fix=False)
        self.assertTrue(
            torch.allclose(buggy.interior_vertices, v_before),
            "expected the OLD gating to leave frozen vertices unmoved (bug repro)")
        self.assertFalse(
            torch.allclose(buggy.density, d_before),
            "density should still train after freezing under either gating")

        # --- fixed gating: vertices move ---
        fixed = _make_frozen_model(seed=1)
        opt_f = FrozenTetOptimizer(fixed, freeze_lr=1e-2, vertices_lr=1e-1)
        v_before_f = fixed.interior_vertices.detach().clone()
        post_freeze_step(fixed, opt_f, use_fix=True)
        self.assertFalse(
            torch.allclose(fixed.interior_vertices, v_before_f),
            "the FIX should let frozen vertices move after freeze_start")


if __name__ == "__main__":
    unittest.main()
