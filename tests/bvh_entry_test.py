"""BVH entry finder + adjacency walker vs the JAX quadrature reference.

Property under test: for rays whose origins lie outside a tet mesh, seeding
``TetrahedralRayTrace`` with the start tet found by the Warp boundary BVH
(``bvh_entry.BoundaryEntryFinder``) reproduces the JAX quadrature render of
the same rays. This is the role the old OptiX tracer played; the walker
itself is covered by single_tet/raytrace_thin_tet tests, so failures here
point at the entry finder (wrong tet, wrong face->tet map, missed hits).

Thresholds: the JAX reference integrates with 10k uniform samples over a
t-range of ~7 scene units => dt ~ 7e-4; with density 10 the per-segment
discretization error is O(sigma*dt) ~ 7e-3. The walker is analytic per
segment, so 1e-1 max / 1e-2 mean on RGB+alpha is pure quadrature headroom,
not a recorded baseline.
"""

import numpy as np
import torch
from absl.testing import absltest, parameterized
from scipy.spatial import Delaunay

from delaunay_rasterization.internal.bvh_entry import BoundaryEntryFinder
from delaunay_rasterization.internal.ray_trace import TetrahedralRayTrace
from utils.topo_utils import build_adj
from jaxutil import tetra_quad

TMIN = 0.01
MAX_ITERS = 200


def make_scene(seed, n_points=12, density=10.0):
    """Random Delaunay tet cluster inside the unit cube, on cuda."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1.0, 1.0, (n_points, 3))
    tri = Delaunay(pts)
    vertices = torch.tensor(pts, dtype=torch.float32, device="cuda")
    indices = torch.tensor(tri.simplices, dtype=torch.int32, device="cuda")
    T = indices.shape[0]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cell_values = torch.empty((T, 7), device="cuda")
    cell_values[:, 0] = density
    cell_values[:, 1:4] = 0.1 + 0.8 * torch.rand((T, 3), generator=gen,
                                                 device="cuda")
    cell_values[:, 4:7] = 0.05 * torch.randn((T, 3), generator=gen,
                                             device="cuda")
    return vertices, indices, cell_values


def make_rays(seed, vertices, indices, cam_dist=4.0, n_hit=64, n_miss=8):
    """Rays from a single outside viewpoint: most aimed at barycentric
    points strictly inside random tets (guaranteed hits), a few aimed away.
    Returns rays [N, 6] and a world->cam viewmat (needed by the JAX
    reference to bound its sample range)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cam_dir = torch.randn(3, generator=gen, device="cuda")
    cam_pos = cam_dist * cam_dir / cam_dir.norm()

    tet_ids = torch.randint(0, indices.shape[0], (n_hit,), generator=gen,
                            device="cuda")
    bary = torch.rand((n_hit, 4), generator=gen, device="cuda") + 0.05
    bary = bary / bary.sum(dim=-1, keepdim=True)
    targets = (vertices[indices[tet_ids].long()] * bary[..., None]).sum(dim=1)
    away = cam_pos[None] * 2.0 + torch.randn((n_miss, 3), generator=gen,
                                             device="cuda")
    dirs_hit = targets - cam_pos[None]
    dirs_miss = away - cam_pos[None]
    dirs = torch.cat([dirs_hit, dirs_miss], dim=0)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    origins = cam_pos[None].expand_as(dirs)
    rays = torch.cat([origins, dirs], dim=-1).contiguous()

    # Look-at world->camera matrix (camera z toward the origin).
    forward = -cam_pos / cam_pos.norm()
    tmp = torch.tensor([0.0, 0.0, 1.0], device="cuda")
    if forward.cross(tmp, dim=0).norm() < 1e-3:
        tmp = torch.tensor([0.0, 1.0, 0.0], device="cuda")
    right = torch.cross(forward, tmp, dim=0)
    right = right / right.norm()
    up = torch.cross(right, forward, dim=0)
    c2w = torch.eye(4, device="cuda")
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, forward, cam_pos
    return rays, torch.linalg.inv(c2w), n_hit


def brute_force_entry_t(rays, vertices, indices):
    """[N, T] entry distance of each ray into each tet (inf if missed).

    A tet is the intersection of four half-spaces; clip the ray against
    them. Independent of the BVH and of the walker's face conventions.
    """
    o, d = rays[:, :3], rays[:, 3:]                      # [N,3]
    v = vertices[indices.long()]                         # [T,4,3]
    opp = torch.stack([v[:, 3], v[:, 0], v[:, 1], v[:, 2]], dim=1)  # [T,4,3]
    va = torch.stack([v[:, 0], v[:, 1], v[:, 0], v[:, 3]], dim=1)
    vb = torch.stack([v[:, 2], v[:, 2], v[:, 3], v[:, 0]], dim=1)
    vc = torch.stack([v[:, 1], v[:, 3], v[:, 2], v[:, 1]], dim=1)
    n = torch.cross(vb - va, vc - va, dim=-1)            # [T,4,3]
    flip = ((opp - va) * n).sum(-1, keepdim=True).sign()
    n = n * flip                                         # inward normals

    num = ((va[None] - o[:, None, None]) * n[None]).sum(-1)   # [N,T,4]
    den = (d[:, None, None] * n[None]).sum(-1)                # [N,T,4]
    t0 = num / den
    lower = torch.where(den > 1e-12, t0, torch.full_like(t0, -torch.inf))
    upper = torch.where(den < -1e-12, t0, torch.full_like(t0, torch.inf))
    # den ~ 0: ray parallel to plane -- inside iff num <= 0 (n inward).
    parallel_out = (den.abs() <= 1e-12) & (num > 0)
    t_in = lower.max(dim=-1).values.clamp_min(0.0)
    t_out = upper.min(dim=-1).values
    hit = (t_in <= t_out) & ~parallel_out.any(dim=-1)
    return torch.where(hit, t_in, torch.full_like(t_in, torch.inf))


class BvhEntryTest(parameterized.TestCase):

    @parameterized.product(seed=[0, 1, 2, 3])
    def test_entry_matches_brute_force(self, seed):
        vertices, indices, _ = make_scene(seed)
        tet_adj = build_adj(vertices, indices.long(), device="cuda").int()
        finder = BoundaryEntryFinder(vertices, indices, tet_adj)
        rays, _, n_hit = make_rays(seed + 100, vertices, indices)

        start_tet, t_enter = finder.find_entry(rays)
        entry_t = brute_force_entry_t(rays, vertices, indices)   # [N,T]
        t_min = entry_t.min(dim=-1).values

        bf_hit = torch.isfinite(t_min)
        bvh_hit = start_tet >= 0
        torch.testing.assert_close(bvh_hit, bf_hit)
        self.assertEqual(int(bvh_hit[:n_hit].sum()), n_hit,
                         "rays aimed inside tets must all hit")

        hit = bvh_hit
        # The BVH hit distance is the first entry into the volume...
        torch.testing.assert_close(t_enter[hit], t_min[hit],
                                   atol=1e-4, rtol=1e-4)
        # ...and the reported tet is one whose own entry equals it (tet id
        # may legitimately differ from argmin on shared edges/vertices).
        own_t = entry_t[hit].gather(
            1, start_tet[hit].long().unsqueeze(1)).squeeze(1)
        torch.testing.assert_close(own_t, t_min[hit], atol=1e-4, rtol=1e-4)

    @parameterized.product(seed=[0, 1, 2, 3])
    def test_render_matches_jax(self, seed):
        vertices, indices, cell_values = make_scene(seed)
        tet_adj = build_adj(vertices, indices.long(), device="cuda").int()
        finder = BoundaryEntryFinder(vertices, indices, tet_adj)
        rays, viewmat, _ = make_rays(seed + 100, vertices, indices)
        N = rays.shape[0]

        start_tet, _ = finder.find_entry(rays)
        hit = start_tet >= 0

        output = torch.zeros((N, 4), device="cuda")
        with torch.no_grad():
            out_hit, _ = TetrahedralRayTrace.apply(
                rays[hit], indices, vertices, cell_values, tet_adj,
                start_tet[hit], TMIN, MAX_ITERS)
        output[hit] = out_hit

        jax_img, _ = tetra_quad.render_rays(
            vertices.cpu().numpy(), indices.cpu().numpy(),
            cell_values[:, 1:7].cpu().numpy(),
            cell_values[:, :1].cpu().numpy(),
            rays[:, :3].reshape(N, 1, 3).cpu().numpy(),
            rays[:, 3:].reshape(N, 1, 3).cpu().numpy(),
            viewmat.cpu().numpy(), TMIN, np.linspace(0, 1, 10000))
        jax_rgba = np.asarray(jax_img).reshape(N, -1)[:, :4]

        diff = np.abs(output.cpu().numpy() - jax_rgba)
        self.assertLess(diff.mean(), 1e-2,
                        f"mean RGBA error vs JAX too high: {diff.mean():.4f}")
        self.assertLess(diff.max(), 1e-1,
                        f"max RGBA error vs JAX too high: {diff.max():.4f}")
        # Missed rays must contribute exactly nothing.
        np.testing.assert_array_equal(
            output[~hit].cpu().numpy(),
            np.zeros((int((~hit).sum()), 4), np.float32))
        self.assertGreater(jax_rgba[hit.cpu().numpy(), 3].max(), 0.5,
                           "scene should be substantially visible")


if __name__ == "__main__":
    absltest.main()
