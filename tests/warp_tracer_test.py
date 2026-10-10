"""Warp face-marching tracer (OptiX port) vs the JAX quadrature reference.

Property under test: ``WarpFaceTracer`` -- the Warp port of the retired
OptiX spline tracer (BVH over all tet faces, segment-per-face-crossing
integration) -- reproduces the JAX reference render of arbitrary rays,
for origins both outside AND inside the mesh. Inside origins are the case
the walker needs dtlookup for; the face-marcher needs nothing.

Thresholds are quadrature headroom, not baselines: the JAX reference uses
10k uniform samples over a t-range of ~7 scene units (dt ~ 7e-4, density 10
=> per-segment discretization error O(sigma*dt) ~ 7e-3), while the tracer
integrates each segment analytically. Hence 1e-2 mean / 1e-1 max on RGBA.
"""

import numpy as np
import torch
from absl.testing import absltest, parameterized
from scipy.spatial import Delaunay

from delaunay_rasterization.internal.warp_tracer import WarpFaceTracer
from jaxutil import tetra_quad

TMIN = 0.01
MAX_ITERS = 512


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


def lookat_viewmat(cam_pos):
    """World->camera matrix with camera z toward the origin (the JAX
    reference uses it only to bound its sample range)."""
    forward = -cam_pos / cam_pos.norm()
    tmp = torch.tensor([0.0, 0.0, 1.0], device="cuda")
    if torch.cross(forward, tmp, dim=0).norm() < 1e-3:
        tmp = torch.tensor([0.0, 1.0, 0.0], device="cuda")
    right = torch.cross(forward, tmp, dim=0)
    right = right / right.norm()
    up = torch.cross(right, forward, dim=0)
    c2w = torch.eye(4, device="cuda")
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, forward, cam_pos
    return torch.linalg.inv(c2w)


def rays_from(origin, vertices, indices, seed, n_hit=64, n_miss=8):
    """Rays from a shared origin aimed at barycentric points inside random
    tets, plus a few aimed away from the scene."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tet_ids = torch.randint(0, indices.shape[0], (n_hit,), generator=gen,
                            device="cuda")
    bary = torch.rand((n_hit, 4), generator=gen, device="cuda") + 0.05
    bary = bary / bary.sum(dim=-1, keepdim=True)
    targets = (vertices[indices[tet_ids].long()] * bary[..., None]).sum(dim=1)
    away = origin[None] * 2.0 + 10.0 * origin.norm() * torch.randn(
        (n_miss, 3), generator=gen, device="cuda")
    dirs = torch.cat([targets - origin[None], away - origin[None]], dim=0)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    origins = origin[None].expand_as(dirs)
    return torch.cat([origins, dirs], dim=-1).contiguous()


def jax_reference(rays, vertices, indices, cell_values, viewmat,
                  n_samples=10000):
    N = rays.shape[0]
    jax_img, _ = tetra_quad.render_rays(
        vertices.cpu().numpy(), indices.cpu().numpy(),
        cell_values[:, 1:7].cpu().numpy(),
        cell_values[:, :1].cpu().numpy(),
        rays[:, :3].reshape(N, 1, 3).cpu().numpy(),
        rays[:, 3:].reshape(N, 1, 3).cpu().numpy(),
        viewmat.cpu().numpy(), TMIN, np.linspace(0, 1, n_samples))
    return np.asarray(jax_img).reshape(N, -1)[:, :4]


try:
    import torchoptix  # noqa: F401
    OPTIX_AVAILABLE = True
except ImportError:
    OPTIX_AVAILABLE = False


def trace_any(tracer, mode, rays, cell_values, vertices, indices,
              t_far=1e30, **kw):
    """Run the pure BVH face-marcher, the hybrid adjacency walk, or the
    torchoptix hardware wavefront."""
    if mode == "bvh":
        return tracer.trace(rays, cell_values, TMIN, t_far=t_far,
                            max_iters=MAX_ITERS, **kw)
    if mode == "optix":
        from delaunay_rasterization.internal.optix_tracer import (
            OptixSplineTracer)
        ot = OptixSplineTracer(vertices, indices)
        return ot.trace(rays, cell_values, TMIN, t_far=t_far,
                        max_iters=MAX_ITERS, **kw)
    from utils.topo_utils import build_adj
    tet_adj = build_adj(vertices, indices.long(), device="cuda").int()
    return tracer.trace_walk(rays, cell_values, tet_adj, TMIN, t_far=t_far,
                             max_iters=MAX_ITERS, **kw)


class WarpTracerTest(parameterized.TestCase):

    def _check(self, rays, vertices, indices, cell_values, viewmat,
               mode="bvh"):
        if mode == "optix" and not OPTIX_AVAILABLE:
            self.skipTest("torchoptix not installed")
        tracer = WarpFaceTracer(vertices, indices)
        out = trace_any(tracer, mode, rays, cell_values, vertices, indices)
        ref = jax_reference(rays, vertices, indices, cell_values, viewmat)

        out_np = out.cpu().numpy()
        self.assertTrue(np.isfinite(out_np).all(), "non-finite output")
        diff = np.abs(out_np - ref)
        self.assertLess(diff.mean(), 1e-2,
                        f"mean RGBA error vs JAX: {diff.mean():.4f}")
        self.assertLess(diff.max(), 1e-1,
                        f"max RGBA error vs JAX: {diff.max():.4f}")
        self.assertGreater(ref[:, 3].max(), 0.5,
                           "scene should be substantially visible")

    @parameterized.product(seed=[0, 1, 2, 3], mode=["bvh", "walk", "optix"])
    def test_outside_origin(self, seed, mode):
        """Camera-style rays from outside the hull (entry via boundary)."""
        vertices, indices, cell_values = make_scene(seed)
        gen = torch.Generator(device="cuda").manual_seed(seed + 50)
        cam_dir = torch.randn(3, generator=gen, device="cuda")
        cam_pos = 4.0 * cam_dir / cam_dir.norm()
        rays = rays_from(cam_pos, vertices, indices, seed + 100)
        self._check(rays, vertices, indices, cell_values,
                    lookat_viewmat(cam_pos), mode=mode)

    @parameterized.product(seed=[0, 1, 2, 3], mode=["bvh", "walk", "optix"])
    def test_inside_origin(self, seed, mode):
        """Origin inside a tet -- the case the old OptiX tracer needed
        dtlookup for; the face-marcher classifies the first segment from
        its first face crossing instead."""
        vertices, indices, cell_values = make_scene(seed)
        gen = torch.Generator(device="cuda").manual_seed(seed + 60)
        t0 = int(torch.randint(0, indices.shape[0], (1,), generator=gen,
                               device="cuda"))
        bary = torch.rand(4, generator=gen, device="cuda") + 0.1
        bary = bary / bary.sum()
        origin = (vertices[indices[t0].long()] * bary[:, None]).sum(dim=0)
        rays = rays_from(origin, vertices, indices, seed + 100, n_miss=0)
        # The JAX sample range is bounded by a viewmat; place its virtual
        # camera behind the origin so the whole hull is in front.
        cam_pos = origin + 4.0 * origin / max(origin.norm(), 1e-3)
        self._check(rays, vertices, indices, cell_values,
                    lookat_viewmat(cam_pos), mode=mode)

    @parameterized.product(thinness=[1e-3, 1e-5, 1e-7])
    def test_sliver_tet_finite(self, thinness):
        """Extreme slivers must never produce NaN/Inf or out-of-range alpha.

        This is the adjacency walker's failure mode (bogus exit latching);
        the face-marcher classifies each segment independently, so the worst
        a sliver can do is contribute ~nothing."""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.4, thinness],
        ], dtype=torch.float32, device="cuda")
        indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32,
                               device="cuda")
        cell_values = torch.tensor(
            [[1e4, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]], device="cuda")
        # Grazing and piercing rays through the sliver.
        origins = torch.tensor([[0.5, 0.4, 1.0],
                                [0.5, 0.4, 1.0],
                                [-1.0, 0.4, thinness / 2]], device="cuda")
        dirs = torch.tensor([[0.0, 0.0, -1.0],
                             [0.01, 0.0, -1.0],
                             [1.0, 0.0, 0.0]], device="cuda")
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        rays = torch.cat([origins, dirs], dim=-1).contiguous()
        tracer = WarpFaceTracer(vertices, indices)
        out = tracer.trace(rays, cell_values, TMIN).cpu().numpy()
        self.assertTrue(np.isfinite(out).all(), f"non-finite: {out}")
        self.assertTrue((out[:, 3] >= 0).all() and (out[:, 3] <= 1).all(),
                        f"alpha out of range: {out[:, 3]}")

    @parameterized.product(seed=[0, 1, 2], mode=["bvh", "walk", "optix"])
    def test_moments_and_tfar_vs_jax(self, seed, mode):
        """Finite per-ray t_far cutting INSIDE the volume + VSM moments +
        t_saturate, against per-ray JAX quadrature. This is the shadow-ray
        configuration of renderer.pbr_renderer.ShadowTracer: light->surface
        rays with t_far = surface_dist + margin -- the case where the final
        partial segment (no face hit before t_far) must still integrate."""
        if mode == "optix" and not OPTIX_AVAILABLE:
            self.skipTest("torchoptix not installed")
        vertices, indices, cell_values = make_scene(seed)
        gen = torch.Generator(device="cuda").manual_seed(seed + 70)
        # Light inside the hull (shared origin), rays toward tet interiors.
        t0 = int(torch.randint(0, indices.shape[0], (1,), generator=gen,
                               device="cuda"))
        bary = torch.rand(4, generator=gen, device="cuda") + 0.1
        bary = bary / bary.sum()
        origin = (vertices[indices[t0].long()] * bary[:, None]).sum(dim=0)
        rays = rays_from(origin, vertices, indices, seed + 100, n_miss=0)
        N = rays.shape[0]
        # t_far landing inside the volume, like surf_dist + small margin.
        t_far = 0.2 + 0.8 * torch.rand((N,), generator=gen, device="cuda")

        tracer = WarpFaceTracer(vertices, indices)
        out_w, extras = trace_any(tracer, mode, rays, cell_values,
                                  vertices, indices, t_far=t_far,
                                  return_extras=True)

        # Per-ray quadrature reference over [TMIN, t_far_i].
        S = 8192
        lin = np.linspace(0.0, 1.0, S + 1)[None]                 # [1, S+1]
        tdist = TMIN + (t_far.cpu().numpy()[:, None] - TMIN) * lin
        ref, ref_extras = tetra_quad.render_tetrahedra_volume(
            rays[:, :3].cpu().numpy(), rays[:, 3:].cpu().numpy(), tdist,
            vertices.cpu().numpy(), indices.cpu().numpy(),
            cell_values[:, :1].cpu().numpy(),
            cell_values[:, 1:7].cpu().numpy(),
            return_extras=True, dist_tmin=TMIN)
        ref = np.asarray(ref).reshape(N, -1)
        w = np.asarray(ref_extras["weights"])                    # [N, S]
        t_avg = 0.5 * (tdist[:, 1:] + tdist[:, :-1])
        ref_m1 = (w * t_avg).sum(-1)
        ref_m2 = (w * t_avg ** 2).sum(-1)
        # Quadrature t_saturate: first t where accumulated od crosses 3
        # (saturate_log_t = -3 => alpha 0.95).
        od_cum = np.cumsum(np.asarray(ref_extras["total_density"])
                           * np.diff(tdist), axis=-1)
        crossed = od_cum >= 3.0
        first = crossed.argmax(-1)
        ref_tsat = np.where(crossed.any(-1), t_avg[np.arange(N), first], -1.0)

        diff = np.abs(out_w.cpu().numpy() - ref[:, :4])
        self.assertLess(diff.mean(), 1e-2, f"rgba mean: {diff.mean():.4f}")
        self.assertLess(diff.max(), 1e-1, f"rgba max: {diff.max():.4f}")
        d_m1 = np.abs(extras["m1"].cpu().numpy() - ref_m1)
        d_m2 = np.abs(extras["m2"].cpu().numpy() - ref_m2)
        self.assertLess(d_m1.max(), 5e-2, f"m1 max: {d_m1.max():.4f}")
        self.assertLess(d_m2.max(), 5e-2, f"m2 max: {d_m2.max():.4f}")
        sat_w = extras["t_saturate"].cpu().numpy()
        np.testing.assert_array_equal(sat_w >= 0, ref_tsat >= 0)
        both = sat_w >= 0
        if both.any():
            self.assertLess(np.abs(sat_w[both] - ref_tsat[both]).max(), 1e-2)

    def test_miss_rays_zero(self):
        vertices, indices, cell_values = make_scene(7)
        cam_pos = torch.tensor([4.0, 0.0, 0.0], device="cuda")
        dirs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device="cuda")
        rays = torch.cat([cam_pos.expand(2, 3), dirs], dim=-1).contiguous()
        tracer = WarpFaceTracer(vertices, indices)
        out = tracer.trace(rays, cell_values, TMIN)
        np.testing.assert_array_equal(out.cpu().numpy(),
                                      np.zeros((2, 4), np.float32))


if __name__ == "__main__":
    absltest.main()
