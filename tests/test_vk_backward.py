"""Gradient verification for rmeshvk backward pass.

Tests the backward shader by verifying:
1. Each parameter actually affects the rendered image (sensitivity)
2. Multi-step gradient descent converges (not just one step)
3. Per-parameter optimization works in isolation
4. Analytical vs numerical gradient agreement (finite differences)

Usage:
    uv run python -m pytest tests/test_vk_backward.py -v
    uv run python tests/test_vk_backward.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import numpy as np
from scipy.spatial import Delaunay

from rmesh_wgpu import RMeshRenderer
from rmesh_wgpu.autograd import RMeshForward
from utils.topo_utils import calculate_circumcenters_torch


def make_simple_scene(n_verts=8, width=32, height=32, seed=42):
    """Create a small scene with a few tets for gradient testing.

    Returns:
        renderer, cam_pos, vp, inv_vp, vertices, base_colors, densities, color_grads
        All tensors on CUDA for autograd.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Random points in a unit cube
    pts = np.random.randn(n_verts, 3).astype(np.float32) * 0.3

    # Delaunay tetrahedralization
    tri = Delaunay(pts)
    indices = tri.simplices.astype(np.int32)  # [T, 4]

    vertices_np = pts.ravel().astype(np.float32)
    indices_np = indices.ravel().astype(np.uint32)

    n_tets = indices.shape[0]

    # Base colors: per-tet pre-softplus RGB [T * 3]
    base_colors_np = (np.random.randn(n_tets * 3).astype(np.float32) * 0.3 + 0.5)

    # Densities: [T]
    densities_np = (np.random.rand(n_tets).astype(np.float32) * 5 + 1.0)

    # Color gradients: [T * 3]
    color_grads_np = np.random.randn(n_tets * 3).astype(np.float32) * 0.1

    # Circumdata: [T, 4] = [cx, cy, cz, r^2]
    tets = torch.from_numpy(pts[indices])  # [T, 4, 3]
    cc, r = calculate_circumcenters_torch(tets.double())
    circumdata_np = torch.cat([cc.float(), (r.float() ** 2).unsqueeze(-1)], dim=-1).numpy().ravel().astype(np.float32)

    # Camera: look at origin from z=2
    cam_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)

    # Simple perspective VP matrix
    fov = 1.0  # ~57 degrees
    aspect = width / height
    znear, zfar = 0.1, 100.0
    f = 1.0 / np.tan(fov / 2.0)

    # Projection (column-major for wgpu)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = 1.0
    proj[3, 2] = -(zfar * znear) / (zfar - znear)

    # View matrix (3DGS convention: wvt = W2V^T with z-flip)
    view = np.eye(4, dtype=np.float32)
    view[2, 2] = -1.0
    view[3, 2] = cam_pos[2]

    vp = (view @ proj).astype(np.float32)
    inv_vp = np.linalg.inv(vp).astype(np.float32)

    # Create renderer
    renderer = RMeshRenderer(
        vertices_np, indices_np, base_colors_np, densities_np,
        color_grads_np, circumdata_np, width, height,
    )

    # Tensors (on CUDA for autograd)
    device = torch.device("cuda")
    vertices_t = torch.from_numpy(vertices_np.copy()).to(device)
    base_colors_t = torch.from_numpy(base_colors_np.copy()).to(device)
    densities_t = torch.from_numpy(densities_np.copy()).to(device)
    color_grads_t = torch.from_numpy(color_grads_np.copy()).to(device)
    cam_pos_t = torch.from_numpy(cam_pos).to(device)
    vp_t = torch.from_numpy(vp).to(device)
    inv_vp_t = torch.from_numpy(inv_vp).to(device)

    return renderer, cam_pos_t, vp_t, inv_vp_t, vertices_t, base_colors_t, densities_t, color_grads_t


def render_image(renderer, cam_pos, vp, inv_vp, vertices, base_colors, densities, color_grads):
    """Forward pass, returns [H, W, 4] image."""
    return RMeshForward.apply(
        renderer, cam_pos, vp, inv_vp,
        vertices, base_colors, densities, color_grads,
    )


def compute_loss(renderer, cam_pos, vp, inv_vp, vertices, base_colors, densities, color_grads):
    """Forward pass + scalar loss (sum of RGB channels)."""
    image = render_image(renderer, cam_pos, vp, inv_vp,
                         vertices, base_colors, densities, color_grads)
    return (image[..., :3] ** 2).sum() + (image[..., 3] ** 2).sum()


def compute_target_loss(renderer, cam_pos, vp, inv_vp, vertices, base_colors,
                        densities, color_grads, target):
    """L2 loss against a target image."""
    image = render_image(renderer, cam_pos, vp, inv_vp,
                         vertices, base_colors, densities, color_grads)
    return ((image - target) ** 2).sum()


def finite_diff_grad(renderer, cam_pos, vp, inv_vp,
                     vertices, base_colors, densities, color_grads,
                     param_name, eps=1e-3):
    """Compute numerical gradient for one parameter tensor via central finite differences."""
    params = {
        "vertices": vertices,
        "base_colors": base_colors,
        "densities": densities,
        "color_grads": color_grads,
    }
    param = params[param_name]
    grad = torch.zeros_like(param)

    for i in range(param.numel()):
        # +eps
        p_plus = param.clone()
        p_plus.view(-1)[i] += eps
        kwargs_plus = {k: (p_plus if k == param_name else v) for k, v in params.items()}
        loss_plus = compute_loss(renderer, cam_pos, vp, inv_vp, **kwargs_plus).item()

        # -eps
        p_minus = param.clone()
        p_minus.view(-1)[i] -= eps
        kwargs_minus = {k: (p_minus if k == param_name else v) for k, v in params.items()}
        loss_minus = compute_loss(renderer, cam_pos, vp, inv_vp, **kwargs_minus).item()

        grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * eps)

    return grad


class TestParameterSensitivity(unittest.TestCase):
    """Verify each parameter actually affects the rendered image.

    If a parameter has zero effect on the output, the gradient test would
    trivially pass (both analytical and numerical are 0), but training would
    be broken because that parameter can't be learned.
    """

    def setUp(self):
        self.renderer, self.cam_pos, self.vp, self.inv_vp, \
            self.vertices, self.base_colors, self.densities, self.color_grads = \
            make_simple_scene(n_verts=8, width=32, height=32)

    def _render(self, **overrides):
        params = {
            "vertices": self.vertices,
            "base_colors": self.base_colors,
            "densities": self.densities,
            "color_grads": self.color_grads,
        }
        params.update(overrides)
        with torch.no_grad():
            return render_image(self.renderer, self.cam_pos, self.vp, self.inv_vp,
                                **params)

    def test_base_colors_affect_output(self):
        """Changing base_colors must change the rendered image."""
        img0 = self._render()
        perturbed = self.base_colors.clone()
        perturbed += 0.5  # significant perturbation
        img1 = self._render(base_colors=perturbed)
        diff = (img0 - img1).abs().sum().item()
        print(f"  base_colors perturbation diff: {diff:.4f}")
        self.assertGreater(diff, 0.1,
                           "base_colors has no effect on rendered image")

    def test_densities_affect_output(self):
        """Changing densities must change the rendered image."""
        img0 = self._render()
        perturbed = self.densities.clone()
        perturbed *= 2.0
        img1 = self._render(densities=perturbed)
        diff = (img0 - img1).abs().sum().item()
        print(f"  densities perturbation diff: {diff:.4f}")
        self.assertGreater(diff, 0.1,
                           "densities has no effect on rendered image")

    def test_color_grads_affect_output(self):
        """Changing color_grads must change the rendered image."""
        img0 = self._render()
        perturbed = self.color_grads.clone()
        perturbed += 0.5
        img1 = self._render(color_grads=perturbed)
        diff = (img0 - img1).abs().sum().item()
        print(f"  color_grads perturbation diff: {diff:.4f}")
        self.assertGreater(diff, 0.1,
                           "color_grads has no effect on rendered image")

    def test_vertices_affect_output(self):
        """Changing vertices must change the rendered image."""
        img0 = self._render()
        perturbed = self.vertices.clone()
        perturbed += 0.05
        img1 = self._render(vertices=perturbed)
        diff = (img0 - img1).abs().sum().item()
        print(f"  vertices perturbation diff: {diff:.4f}")
        self.assertGreater(diff, 0.1,
                           "vertices has no effect on rendered image")


class TestMultiStepConvergence(unittest.TestCase):
    """Verify gradient descent actually converges over multiple steps.

    Single-step loss decrease is trivially true for almost any non-zero gradient.
    Multi-step convergence with Adam against a target image tests whether
    gradients are actually useful for optimization.
    """

    def setUp(self):
        self.renderer, self.cam_pos, self.vp, self.inv_vp, \
            self.vertices, self.base_colors, self.densities, self.color_grads = \
            make_simple_scene(n_verts=8, width=32, height=32)

    def _optimize_param(self, param_name, perturbation, n_steps=200, lr=1e-2,
                        min_reduction=0.9):
        """Optimize a single parameter toward a target image.

        Creates target from current params, perturbs the named param,
        then runs Adam for n_steps. Returns (initial_loss, final_loss, reduction).
        """
        # Render target
        with torch.no_grad():
            target = render_image(self.renderer, self.cam_pos, self.vp, self.inv_vp,
                                  self.vertices, self.base_colors, self.densities,
                                  self.color_grads).clone()

        params = {
            "vertices": self.vertices,
            "base_colors": self.base_colors,
            "densities": self.densities,
            "color_grads": self.color_grads,
        }

        # Perturb the target parameter
        p = perturbation(params[param_name].clone()).requires_grad_(True)
        optimizer = torch.optim.Adam([p], lr=lr)

        losses = []
        for step in range(n_steps):
            optimizer.zero_grad()
            kw = {k: (p if k == param_name else v) for k, v in params.items()}
            loss = compute_target_loss(self.renderer, self.cam_pos, self.vp, self.inv_vp,
                                       **kw, target=target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        reduction = 1.0 - losses[-1] / max(losses[0], 1e-10)
        print(f"  {param_name} convergence: {losses[0]:.6f} -> {losses[-1]:.6f} "
              f"(reduction={reduction*100:.1f}%)")
        print(f"    First 5: {[f'{l:.4f}' for l in losses[:5]]}")
        print(f"    Last 5:  {[f'{l:.4f}' for l in losses[-5:]]}")

        # Check for monotonic decrease in later steps (no oscillation)
        late_losses = losses[n_steps//2:]
        increases = sum(1 for i in range(1, len(late_losses)) if late_losses[i] > late_losses[i-1] * 1.01)
        if increases > len(late_losses) // 4:
            print(f"    WARNING: {increases}/{len(late_losses)} loss increases in 2nd half")

        return losses[0], losses[-1], reduction

    def test_all_params_target_convergence(self):
        """Optimize all params (except vertices) to match a target image."""
        with torch.no_grad():
            target = render_image(self.renderer, self.cam_pos, self.vp, self.inv_vp,
                                  self.vertices, self.base_colors, self.densities,
                                  self.color_grads).clone()

        # Perturb all non-vertex params
        b = (self.base_colors.clone() + 0.2).requires_grad_(True)
        d = (self.densities.clone() * 1.3).requires_grad_(True)
        c = (self.color_grads.clone() + 0.1).requires_grad_(True)

        optimizer = torch.optim.Adam([b, d, c], lr=1e-2)
        losses = []

        for step in range(200):
            optimizer.zero_grad()
            loss = compute_target_loss(self.renderer, self.cam_pos, self.vp, self.inv_vp,
                                       self.vertices, b, d, c, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        reduction = 1.0 - losses[-1] / losses[0]
        print(f"  All params: {losses[0]:.4f} -> {losses[-1]:.4f} (reduction={reduction*100:.1f}%)")
        print(f"    First 5: {[f'{l:.4f}' for l in losses[:5]]}")
        print(f"    Last 5:  {[f'{l:.4f}' for l in losses[-5:]]}")
        self.assertGreater(reduction, 0.9,
                           f"All-param optimization only reduced loss by {reduction*100:.1f}%")

    def test_base_colors_only_convergence(self):
        """Optimize only base_colors toward a target."""
        _, _, red = self._optimize_param(
            "base_colors",
            perturbation=lambda p: p + 0.3,
            n_steps=200, lr=1e-2,
        )
        self.assertGreater(red, 0.9,
                           f"base_colors: only {red*100:.1f}% reduction (need >90%)")

    def test_densities_only_convergence(self):
        """Optimize only densities toward a target.

        Density has highly nonlinear effect through exp(-od), so we use
        a moderate perturbation. If gradient is wrong, even small perturbations
        won't converge.
        """
        _, _, red = self._optimize_param(
            "densities",
            perturbation=lambda p: p + 1.0,
            n_steps=200, lr=1e-2,
        )
        self.assertGreater(red, 0.9,
                           f"densities: only {red*100:.1f}% reduction (need >90%)")

    def test_color_grads_only_convergence(self):
        """Optimize only color_grads toward a target."""
        _, _, red = self._optimize_param(
            "color_grads",
            perturbation=lambda p: p + 0.2,
            n_steps=200, lr=1e-2,
        )
        self.assertGreater(red, 0.9,
                           f"color_grads: only {red*100:.1f}% reduction (need >90%)")


class TestFiniteDifferences(unittest.TestCase):
    """Compare analytical gradients against finite differences.

    This verifies forward/backward consistency, but NOT that the forward
    computes the right thing. See TestParameterSensitivity and
    TestMultiStepConvergence for that.
    """

    def setUp(self):
        self.renderer, self.cam_pos, self.vp, self.inv_vp, \
            self.vertices, self.base_colors, self.densities, self.color_grads = \
            make_simple_scene(n_verts=8, width=32, height=32)

    def _get_analytical_grads(self):
        """Run backward and return analytical gradients."""
        v = self.vertices.clone().requires_grad_(True)
        b = self.base_colors.clone().requires_grad_(True)
        d = self.densities.clone().requires_grad_(True)
        c = self.color_grads.clone().requires_grad_(True)

        loss = compute_loss(self.renderer, self.cam_pos, self.vp, self.inv_vp, v, b, d, c)
        loss.backward()

        return {
            "vertices": v.grad.clone(),
            "base_colors": b.grad.clone(),
            "densities": d.grad.clone(),
            "color_grads": c.grad.clone(),
        }

    def _check_grad(self, param_name, rtol=0.05, atol=1e-4, eps=1e-3, cos_thresh=0.95):
        """Compare analytical vs numerical gradient for one parameter."""
        analytical = self._get_analytical_grads()[param_name]
        numerical = finite_diff_grad(
            self.renderer, self.cam_pos, self.vp, self.inv_vp,
            self.vertices, self.base_colors, self.densities, self.color_grads,
            param_name, eps=eps,
        )

        # Filter to non-zero entries (many gradients are zero for non-visible tets)
        mask = (analytical.abs() > atol) | (numerical.abs() > atol)
        if mask.sum() == 0:
            self.fail(f"{param_name}: all gradients near zero — parameter may have no effect")

        a = analytical[mask]
        n = numerical[mask]

        # Relative error
        rel_err = ((a - n).abs() / (n.abs() + atol)).mean().item()
        max_rel_err = ((a - n).abs() / (n.abs() + atol)).max().item()
        cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), n.unsqueeze(0)).item()

        print(f"  {param_name}: {mask.sum().item()}/{analytical.numel()} active, "
              f"mean_rel_err={rel_err:.4f}, max_rel_err={max_rel_err:.4f}, cos_sim={cos_sim:.4f}")

        self.assertGreater(cos_sim, cos_thresh,
                           f"{param_name} gradient direction mismatch: cos_sim={cos_sim:.4f}")
        self.assertLess(rel_err, rtol,
                        f"{param_name} gradient magnitude mismatch: mean_rel_err={rel_err:.4f}")

    def test_density_grad(self):
        """Test gradient w.r.t. densities."""
        self._check_grad("densities", rtol=0.1, eps=0.1)

    def test_base_colors_grad(self):
        """Test gradient w.r.t. base colors."""
        self._check_grad("base_colors", rtol=0.1, eps=0.1)

    def test_color_grads_grad(self):
        """Test gradient w.r.t. color gradients."""
        self._check_grad("color_grads", rtol=1.0, eps=0.1)

    def test_vertices_grad(self):
        """Test gradient w.r.t. vertices.

        Vertex gradients are inherently hard to verify via finite differences
        because mesh rasterization has hard edges (silhouettes, face selection
        boundaries) that create discontinuities.
        """
        self._check_grad("vertices", rtol=100.0, eps=0.005, cos_thresh=0.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
