"""
Compare AlphaBlendTiledRenderFast (wave-reduced backward) against
AlphaBlendTiledRender (interp, Slang autodiff backward).

Forward outputs should be bitwise identical (same math, same data path).
Backward gradients should match within tolerance (different summation order
due to wave reduction changes floating-point associativity).
"""

import math
import unittest
import torch
import numpy as np
from scipy.spatial import Delaunay

from delaunay_rasterization.internal.alphablend_tiled_slang_interp import AlphaBlendTiledRender
from delaunay_rasterization.internal.alphablend_tiled_slang_fast import AlphaBlendTiledRenderFast
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader


# ---------------------------------------------------------------------------
# Scene generation helpers
# ---------------------------------------------------------------------------

def compute_delaunay(points):
    pts_np = points.cpu().numpy()
    tri = Delaunay(pts_np)
    return torch.tensor(tri.simplices, device=points.device).int()


def generate_point_cloud(n_points, radius, device="cuda"):
    """Generate n_points inside a sphere of given radius.

    Uses rejection sampling but keeps generating until we have enough points.
    """
    collected = []
    remaining = n_points
    while remaining > 0:
        # Over-generate by 2.5x to account for sphere/cube ratio (~52%)
        batch = (torch.rand((remaining * 3, 3), device=device) * 2 - 1) * radius
        inside = batch[torch.norm(batch, dim=1) <= radius]
        collected.append(inside[:remaining])
        remaining -= len(collected[-1])
    return torch.cat(collected, dim=0)[:n_points]


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def setup_camera(height, width, fov_degrees, viewmat, device="cuda"):
    f = height / math.tan(fov_degrees * math.pi / 180 / 2.0)
    K = torch.tensor(
        [[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]],
        device=device, dtype=torch.float32,
    )
    fovx = focal2fov(K[0, 0], width)
    fovy = focal2fov(K[1, 1], height)
    cam_pos = viewmat.inverse()[:3, 3]
    distortion_params = torch.zeros(4, device=device)
    return K, cam_pos, fovy, fovx, distortion_params


def build_tcam(viewmat, K, cam_pos, fovy, fovx, distortion_params,
               height, width, tile_size, min_t, render_grid, device="cuda"):
    return dict(
        tile_height=tile_size,
        tile_width=tile_size,
        grid_height=render_grid.grid_height,
        grid_width=render_grid.grid_width,
        min_t=min_t,
        image_height=height,
        image_width=width,
        camera_type=0,
        world_view_transform=viewmat.to(device),
        K=K.to(device),
        cam_pos=cam_pos.to(device),
        distortion_params=distortion_params.to(device),
        fovy=fovy,
        fovx=fovx,
    )


def make_cell_values(n_tets, aux_dim=0, device="cuda"):
    """Random cell values: [density, r, g, b, grd_x, grd_y, grd_z, ...aux]."""
    density = torch.rand(n_tets, 1, device=device) * 5.0 + 0.5
    color = torch.rand(n_tets, 3, device=device)
    gradient = (torch.rand(n_tets, 3, device=device) - 0.5) * 0.5
    parts = [density, color, gradient]
    if aux_dim > 0:
        parts.append(torch.rand(n_tets, aux_dim, device=device) * 0.5)
    return torch.cat(parts, dim=1)


def create_view_matrix(camera_pos, look_at, device="cuda"):
    forward = look_at - camera_pos
    forward = forward / torch.norm(forward)
    right = torch.cross(forward, torch.tensor([0.0, 1.0, 0.0], device=device))
    if torch.norm(right) < 1e-6:
        right = torch.cross(forward, torch.tensor([1.0, 0.0, 0.0], device=device))
    right = right / torch.norm(right)
    up = torch.cross(right, forward)
    viewmat = torch.eye(4, device=device)
    viewmat[:3, 0] = right
    viewmat[:3, 1] = up
    viewmat[:3, 2] = forward
    viewmat[:3, 3] = camera_pos
    return torch.linalg.inv(viewmat)


# ---------------------------------------------------------------------------
# Core comparison routine
# ---------------------------------------------------------------------------

def run_both_renderers(vertices, indices, cell_values, viewmat,
                       height, width, tile_size, min_t, aux_dim,
                       device="cuda"):
    """
    Run interp and fast renderers on identical inputs.
    Returns (interp_results, fast_results).
    """
    render_grid = RenderGrid(height, width,
                             tile_height=tile_size, tile_width=tile_size)
    K, cam_pos, fovy, fovx, distortion_params = setup_camera(
        height, width, 90, viewmat, device)
    tcam = build_tcam(viewmat, K, cam_pos, fovy, fovx, distortion_params,
                      height, width, tile_size, min_t, render_grid, device)

    # Tile / vertex shader (shared between both renderers)
    sorted_idx, tile_ranges, _, _, _, _ = vertex_and_tile_shader(
        indices, vertices.detach(), tcam, render_grid)

    ray_jitter = 0.5 * torch.ones((height, width, 2), device=device)

    results = {}
    for name, Renderer in [("interp", AlphaBlendTiledRender),
                            ("fast", AlphaBlendTiledRenderFast)]:
        verts = vertices.detach().clone().requires_grad_(True)
        cv = cell_values.detach().clone().requires_grad_(True)

        output_img, xyzd_img, distortion_img, tet_alive = Renderer.apply(
            sorted_idx, tile_ranges, indices, verts, cv,
            render_grid, tcam, ray_jitter, aux_dim)

        # Use explicit contiguous gradient tensors — slangtorch rejects
        # the broadcasted/expanded views that autograd sometimes produces.
        grad_output = torch.ones_like(output_img)
        grad_xyzd = torch.ones_like(xyzd_img)
        grad_distortion = torch.ones_like(distortion_img)
        torch.autograd.backward(
            [output_img, xyzd_img, distortion_img],
            [grad_output, grad_xyzd, grad_distortion],
        )

        results[name] = {
            "output_img": output_img.detach(),
            "xyzd_img": xyzd_img.detach(),
            "distortion_img": distortion_img.detach(),
            "vertices_grad": verts.grad.detach(),
            "cell_values_grad": cv.grad.detach(),
        }

    return results["interp"], results["fast"]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestFastVsInterp(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    # ---- Forward exactness ----

    def _assert_forward_match(self, interp, fast, tag=""):
        for key in ["output_img", "xyzd_img", "distortion_img"]:
            a = interp[key]
            b = fast[key]
            max_diff = (a - b).abs().max().item()
            self.assertEqual(
                max_diff, 0.0,
                f"{tag} Forward mismatch on {key}: max_diff={max_diff}")

    # ---- Backward tolerance ----

    def _assert_backward_close(self, interp, fast, atol=1e-3, rtol=1e-3,
                                tag=""):
        for key in ["vertices_grad", "cell_values_grad"]:
            a = interp[key]
            b = fast[key]
            abs_diff = (a - b).abs()
            max_diff = abs_diff.max().item()
            # Relative tolerance where reference is non-tiny
            denom = a.abs().clamp(min=1e-7)
            rel_diff = (abs_diff / denom)
            # Check: max absolute must be under atol, OR relative under rtol
            ok = (abs_diff < atol) | (rel_diff < rtol)
            n_bad = (~ok).sum().item()
            if n_bad > 0:
                # Allow up to 1% of elements to exceed tolerance
                frac = n_bad / a.numel()
                self.assertLess(
                    frac, 0.01,
                    f"{tag} Backward mismatch on {key}: {n_bad}/{a.numel()} "
                    f"elements exceed atol={atol}, rtol={rtol}. "
                    f"max_abs_diff={max_diff:.6e}")

    # ---- Single tetrahedron ----

    def test_single_tet(self):
        device = "cuda"
        vertices = torch.tensor([
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.5, 1.0, 2.0],
            [0.5, 0.5, 3.0],
        ], device=device)
        indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device)
        cell_values = make_cell_values(1, device=device)

        viewmat = torch.eye(4, device=device)
        interp, fast = run_both_renderers(
            vertices, indices, cell_values, viewmat,
            height=16, width=16, tile_size=4, min_t=0.01, aux_dim=0)

        self._assert_forward_match(interp, fast, tag="single_tet")
        self._assert_backward_close(interp, fast, tag="single_tet")

    # ---- Small point clouds with identity camera ----

    def _run_identity_camera(self, n_points, radius):
        device = "cuda"
        points = generate_point_cloud(n_points, radius, device)
        assert len(points) >= 5, "generate_point_cloud should always return enough points"
        # Push them in front of the camera (positive z)
        points[:, 2] += radius + 2.0
        indices = compute_delaunay(points)
        cell_values = make_cell_values(indices.shape[0], device=device)

        viewmat = torch.eye(4, device=device)
        interp, fast = run_both_renderers(
            points, indices, cell_values, viewmat,
            height=32, width=32, tile_size=4, min_t=0.01, aux_dim=0)

        tag = f"identity_n{n_points}_r{radius}"
        self._assert_forward_match(interp, fast, tag=tag)
        self._assert_backward_close(interp, fast, tag=tag)

    def test_identity_camera_8pts(self):
        self._run_identity_camera(8, 0.3)

    def test_identity_camera_15pts(self):
        self._run_identity_camera(15, 0.5)

    def test_identity_camera_25pts(self):
        self._run_identity_camera(25, 1.0)

    # ---- Outside-looking-in camera ----

    def _run_outside_camera(self, n_points, radius):
        device = "cuda"
        points = generate_point_cloud(n_points, radius, device)
        assert len(points) >= 5, "generate_point_cloud should always return enough points"
        indices = compute_delaunay(points)
        cell_values = make_cell_values(indices.shape[0], device=device)

        center = points.mean(dim=0)
        direction = torch.randn(3, device=device)
        direction = direction / torch.norm(direction)
        camera_pos = center + direction * (radius + 5.0)
        viewmat = create_view_matrix(camera_pos, center, device)

        interp, fast = run_both_renderers(
            points, indices, cell_values, viewmat,
            height=32, width=32, tile_size=4, min_t=0.01, aux_dim=0)

        tag = f"outside_n{n_points}_r{radius}"
        self._assert_forward_match(interp, fast, tag=tag)
        self._assert_backward_close(interp, fast, tag=tag)

    def test_outside_camera_10pts(self):
        self._run_outside_camera(10, 5.0)

    def test_outside_camera_20pts(self):
        self._run_outside_camera(20, 10.0)

    # ---- Inside-looking-out camera (high overlap) ----

    def test_inside_camera(self):
        device = "cuda"
        n_points = 15
        radius = 5.0
        points = generate_point_cloud(n_points, radius, device)
        assert len(points) >= 5, "generate_point_cloud should always return enough points"
        indices = compute_delaunay(points)
        cell_values = make_cell_values(indices.shape[0], device=device)

        # Place camera inside the point cloud
        center_idx = torch.randint(0, len(points), (1,)).item()
        camera_pos = points[center_idx]

        # Random look direction
        look_dir = torch.randn(3, device=device)
        look_dir = look_dir / torch.norm(look_dir)
        look_at = camera_pos + look_dir

        viewmat = create_view_matrix(camera_pos, look_at, device)

        interp, fast = run_both_renderers(
            points, indices, cell_values, viewmat,
            height=32, width=32, tile_size=4, min_t=0.01, aux_dim=0)

        self._assert_forward_match(interp, fast, tag="inside_camera")
        self._assert_backward_close(interp, fast, tag="inside_camera")

    # ---- Different tile sizes ----

    def _run_tile_size(self, tile_size):
        device = "cuda"
        points = generate_point_cloud(12, 3.0, device)
        assert len(points) >= 5, "generate_point_cloud should always return enough points"
        points[:, 2] += 5.0
        indices = compute_delaunay(points)
        cell_values = make_cell_values(indices.shape[0], device=device)

        viewmat = torch.eye(4, device=device)
        interp, fast = run_both_renderers(
            points, indices, cell_values, viewmat,
            height=32, width=32, tile_size=tile_size, min_t=0.01, aux_dim=0)

        tag = f"tile{tile_size}"
        self._assert_forward_match(interp, fast, tag=tag)
        self._assert_backward_close(interp, fast, tag=tag)

    def test_tile_size_4(self):
        self._run_tile_size(4)

    def test_tile_size_8(self):
        self._run_tile_size(8)

    # ---- With aux_dim > 0 ----

    def test_with_aux_dim(self):
        device = "cuda"
        aux_dim = 2
        points = generate_point_cloud(10, 2.0, device)
        assert len(points) >= 5, "generate_point_cloud should always return enough points"
        points[:, 2] += 4.0
        indices = compute_delaunay(points)
        n_tets = indices.shape[0]
        cell_values = make_cell_values(n_tets, aux_dim=aux_dim, device=device)

        viewmat = torch.eye(4, device=device)
        interp, fast = run_both_renderers(
            points, indices, cell_values, viewmat,
            height=32, width=32, tile_size=4, min_t=0.01, aux_dim=aux_dim)

        self._assert_forward_match(interp, fast, tag="aux_dim2")
        self._assert_backward_close(interp, fast, tag="aux_dim2")

    # ---- Dense scene (stress test for wave reduction) ----

    def test_dense_scene(self):
        device = "cuda"
        points = generate_point_cloud(40, 2.0, device)
        if len(points) < 10:
            self.skipTest("Too few points after radius filter")
        points[:, 2] += 4.0
        indices = compute_delaunay(points)
        cell_values = make_cell_values(indices.shape[0], device=device)

        viewmat = torch.eye(4, device=device)
        interp, fast = run_both_renderers(
            points, indices, cell_values, viewmat,
            height=64, width=64, tile_size=4, min_t=0.01, aux_dim=0)

        self._assert_forward_match(interp, fast, tag="dense")
        self._assert_backward_close(interp, fast, atol=1e-3, rtol=1e-3,
                                     tag="dense")

    # ---- Gradient magnitude sanity check ----

    def test_gradients_nonzero(self):
        """Ensure both renderers produce nonzero gradients when the scene
        has visible tets."""
        device = "cuda"
        points = generate_point_cloud(10, 1.0, device)
        assert len(points) >= 5, "generate_point_cloud should always return enough points"
        points[:, 2] += 3.0
        indices = compute_delaunay(points)
        cell_values = make_cell_values(indices.shape[0], device=device)
        viewmat = torch.eye(4, device=device)

        interp, fast = run_both_renderers(
            points, indices, cell_values, viewmat,
            height=32, width=32, tile_size=4, min_t=0.01, aux_dim=0)

        for name, res in [("interp", interp), ("fast", fast)]:
            vg = res["vertices_grad"]
            cg = res["cell_values_grad"]
            self.assertGreater(
                vg.abs().max().item(), 0.0,
                f"{name}: vertices_grad is all zeros")
            self.assertGreater(
                cg.abs().max().item(), 0.0,
                f"{name}: cell_values_grad is all zeros")


if __name__ == "__main__":
    unittest.main()
