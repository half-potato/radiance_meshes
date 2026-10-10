"""Ray tracer robustness tests for thin (sliver) tetrahedra.

Sliver tets -- one vertex collapsing toward the plane of the other three --
are the worst case for the adjacency-walking ray tracer:

    * Three of the four faces become near-coplanar with each other, so their
      normals nearly align. Rays approximately tangent to that shared plane
      produce ``dot(face_normal, ray_dir)`` ~ 0 across multiple faces at once,
      meaning t_min / t_max become ill-conditioned.
    * The valid t-segment inside the tet is tiny, so any face plane the ray
      grazes can swap which face is "exit" between adjacent iterations and
      cause infinite loops or stalls if the algorithm isn't careful.

Each test renders a thin tet, casts rays roughly parallel to the wide pair of
faces, and compares the ray tracer output against the JAX quadrature
reference (same path as single_tet_test). It also checks that no pixel ends
up NaN/Inf -- a regression we explicitly do not want to silently absorb.
"""

import math
import random

import numpy as np
import torch
from absl.testing import absltest
from absl.testing import parameterized
from icecream import ic

# Alias so pytest doesn't collect the imported helper as a test function.
from utils.compare_quad import test_tetrahedra_rendering as run_tetrahedra_rendering
from utils.model_util import offset_normalize
from utils.test_util import bcolors, compare_dict_values
from utils.topo_utils import calculate_circumcenters_torch

torch.set_printoptions(precision=8)

key_pairs = [
    ('rt_image', 'jax_image', 'RT Image vs JAX', 1.5e-1, 1.5e-1),
    ('torch_image', 'jax_image', 'Raster Image vs JAX', 1.5e-1, 1.5e-1),
]


def _viewmat_look(cam_xyz: torch.Tensor, look_dir: torch.Tensor) -> torch.Tensor:
    """Build a view matrix at cam_xyz pointing along look_dir.

    Falls back to a different up vector when the look direction is parallel
    to z, so the basis stays well-conditioned even for thin-axis-aligned
    cameras.
    """
    forward = look_dir / torch.linalg.norm(look_dir)
    up_candidate = torch.tensor([0.0, 0.0, 1.0], device=cam_xyz.device)
    if abs(torch.dot(forward, up_candidate).item()) > 0.95:
        up_candidate = torch.tensor([0.0, 1.0, 0.0], device=cam_xyz.device)
    right = torch.cross(forward, up_candidate)
    right = right / torch.linalg.norm(right)
    up = torch.cross(right, forward)

    viewmat = torch.eye(4, device=cam_xyz.device)
    viewmat[:3, 0] = right
    viewmat[:3, 1] = up
    viewmat[:3, 2] = forward
    viewmat[:3, 3] = cam_xyz
    return torch.linalg.inv(viewmat)


def _make_thin_tet(thinness: float, device='cuda') -> torch.Tensor:
    """Tet whose first three vertices lie on z=0 and whose apex sits at
    (0, 0, thinness). As ``thinness`` -> 0 the tet collapses into a triangle."""
    return torch.tensor([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 0.0,  1.0, 0.0],
        [ 0.0,  0.0, thinness],
    ], device=device, dtype=torch.float32)


class ThinTetRayTraceTest(parameterized.TestCase):
    """Compares the slang ray tracer against JAX quadrature on thin tets."""

    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)
        random.seed(7)
        self.height = 16
        self.width = 16
        self.indices = torch.tensor([[0, 1, 2, 3]]).int().cuda()

    def _build_cell_values(self, vertices):
        # Match the magnitudes used in single_tet_test so JAX/raster/RT
        # comparisons stay in the same regime as the existing suite.
        raw = torch.tensor([
            10.0,         # density
            0.25, 0.5, 0.1,   # base color
            -0.15, 0.25, 0.1,  # color gradient
        ]).reshape(1, -1).cuda()
        circumcenters, _ = calculate_circumcenters_torch(vertices.reshape(1, 4, 3))
        new_color, new_grd = offset_normalize(
            raw[:, 1:4], raw[:, 4:7], circumcenters, vertices.reshape(1, 4, 3))
        raw[:, 1:4] = new_color
        raw[:, 4:7] = new_grd
        return raw, torch.tensor([[10.0]], device='cuda')

    def _run_compare(self, vertices, viewmat, tile_size=8):
        cell_values, tet_density = self._build_cell_values(vertices)
        results = run_tetrahedra_rendering(
            vertices, self.indices, cell_values, tet_density, viewmat,
            height=self.height, width=self.width,
            tile_size=tile_size, n_samples=20000,
            check_gradients=False,
        )
        # NaN/Inf check before any numeric comparison -- a sliver tet that
        # silently produces NaN would also slip past the threshold check.
        self.assertFalse(np.isnan(results['rt_image']).any(),
                         msg=f'NaN in rt_image\n{results["rt_image"]}')
        self.assertFalse(np.isinf(results['rt_image']).any(),
                         msg=f'Inf in rt_image\n{results["rt_image"]}')
        self.assertFalse(np.isnan(results['rt_aux']).any(),
                         msg='NaN in rt_aux')
        return results

    def _run_finite_only(self, vertices, viewmat, tile_size=8):
        """For extreme thinness where JAX may itself disagree -- we only assert
        the ray tracer doesn't NaN/explode."""
        cell_values, tet_density = self._build_cell_values(vertices)
        results = run_tetrahedra_rendering(
            vertices, self.indices, cell_values, tet_density, viewmat,
            height=self.height, width=self.width,
            tile_size=tile_size, n_samples=2000,
            check_gradients=False,
        )
        self.assertFalse(np.isnan(results['rt_image']).any(),
                         msg='NaN in rt_image')
        self.assertFalse(np.isinf(results['rt_image']).any(),
                         msg='Inf in rt_image')
        self.assertFalse(np.isnan(results['rt_aux']).any(),
                         msg='NaN in rt_aux')
        # Alpha must stay in [0, 1] regardless of geometry.
        # rt_aux columns: normal(3), depth(1), entropy(4)
        # rt_image columns: r,g,b -- no alpha here, but alpha is tested via the
        # raw buffer in TetrahedralRayTrace.apply.
        self.assertTrue((results['rt_image'] >= -1e-3).all(),
                        msg='Negative pixel values from RT')

    @parameterized.product(
        thinness=[1e-1, 1e-2, 1e-3],
    )
    def test_camera_inside_horizontal_ray(self, thinness):
        """Camera at the centroid of a thin tet, looking horizontally.
        Three of the four faces are nearly parallel to the +x ray direction --
        the worst case for plane-clipping precision."""
        vertices = _make_thin_tet(thinness)
        # Centroid; safely inside the tet for all positive thinness.
        centroid = vertices.mean(dim=0)
        viewmat = _viewmat_look(
            centroid, torch.tensor([1.0, 0.0, 0.0], device='cuda'))
        results = self._run_compare(vertices, viewmat)
        compare_dict_values(
            results, results, key_pairs, vertices, viewmat, tile_size=8)

    @parameterized.product(
        thinness=[1e-1, 1e-2, 1e-3],
        # Tilt the ray slightly off the x-axis to vary how parallel it is
        # to the near-coplanar faces.
        tilt_z=[0.0, 1e-3, 1e-2],
    )
    def test_camera_outside_grazing_ray(self, thinness, tilt_z):
        """Camera outside the tet, ray grazes the wide face stack at small
        angle. start_tet=0 + plane clipping inside the kernel must still
        produce sane output."""
        vertices = _make_thin_tet(thinness)
        cam_xyz = torch.tensor([-3.0, 0.0, thinness * 0.5], device='cuda')
        look_dir = torch.tensor([1.0, 0.0, tilt_z], device='cuda')
        viewmat = _viewmat_look(cam_xyz, look_dir)
        results = self._run_compare(vertices, viewmat)
        compare_dict_values(
            results, results, key_pairs, vertices, viewmat, tile_size=8)

    @parameterized.product(
        thinness=[1e-4, 1e-5, 1e-6],
    )
    def test_extreme_thinness_finite(self, thinness):
        """At extreme thinness JAX quadrature is itself unreliable, so we
        only assert the ray tracer stays finite and non-negative -- the
        regression we care about is NaN propagation through the kernel."""
        vertices = _make_thin_tet(thinness)
        # Sweep camera positions: inside, outside-x, above, below
        for cam_xyz, look_dir in [
            (torch.tensor([0.0, -0.25, thinness * 0.5], device='cuda'),
             torch.tensor([1.0, 0.0, 0.0], device='cuda')),
            (torch.tensor([-3.0, 0.0, thinness * 0.5], device='cuda'),
             torch.tensor([1.0, 0.0, 0.0], device='cuda')),
            (torch.tensor([0.0, 0.0, 1.0], device='cuda'),
             torch.tensor([0.0, 0.0, -1.0], device='cuda')),
            (torch.tensor([0.0, 0.0, -1.0], device='cuda'),
             torch.tensor([0.0, 0.0, 1.0], device='cuda')),
        ]:
            viewmat = _viewmat_look(cam_xyz, look_dir)
            self._run_finite_only(vertices, viewmat)

    def test_ray_exactly_in_thin_plane(self):
        """A ray that lies (numerically) in the same z-plane as the wide face
        stack: dot(n, dir) is essentially zero against three faces. The kernel
        must classify that as no contribution, not NaN."""
        thinness = 1e-3
        vertices = _make_thin_tet(thinness)
        # Place camera exactly on the z=0 base; ray runs along +x in z=0.
        cam_xyz = torch.tensor([-3.0, 0.0, 0.0], device='cuda')
        look_dir = torch.tensor([1.0, 0.0, 0.0], device='cuda')
        viewmat = _viewmat_look(cam_xyz, look_dir)
        self._run_finite_only(vertices, viewmat)


if __name__ == '__main__':
    absltest.main()
