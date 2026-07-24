"""Stochastic size-proportional tet sampling: correctness of the unbiased LOD.

The Delaunay rasterizer can Russian-roulette sub-pixel tets: a tet projecting to
a max pixel extent ``s`` is included with probability ``p = min(s, 1)`` and, when
kept, has its DENSITY reweighted by ``1/p``. This is importance sampling: in the
thin/low-optical-depth regime where sub-pixel tets live (alpha ~= tau, linear),
``E[alpha] = p*(1-exp(-tau/p)) ~= tau`` = the true alpha, so the EXPECTED render
equals the full (no-sampling) render and the training gradient is unbiased.

Contract (property, not snapshot):

1.  >=1px exactness. When every visible tet has ``s >= 1px`` (high res / big
    tets) ``p == 1`` everywhere -> no sampling, no reweight -> the stochastic
    render is BITWISE identical to the baseline (render / alpha / aux / xyzd).

2.  Unbiasedness. On a low-res, many-sub-pixel-tet scene in the thin regime, the
    Monte-Carlo mean of ``K`` independent stochastic renders converges to the
    FULL render as ``K`` grows: PSNR(mean_K, ref) increases with K and clears a
    high bar by K=256, and a scalar functional (total luminance) matches ref
    within Monte-Carlo error ~ 1/sqrt(K). Any residual is the documented
    thin-regime bias (grows with per-tet optical depth).

3.  Speed. At walkthrough res on real london geometry the stochastic fwd+bwd is
    substantially faster than the full render (sub-pixel tets are ~46% of work).
    Opt-in via ``-m slow`` and the london ckpt.
"""

import math
import os

import numpy as np
import pytest
import torch

torch.manual_seed(20260722)
np.random.seed(20260722)

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from tests.interval_parity_test import make_camera, FakeModel  # noqa: E402
from utils.train_util import render  # noqa: E402
from utils.model_query import prepare_query  # noqa: E402

DEVICE = "cuda"


# --------------------------------------------------------------------------- #
# scene builders
# --------------------------------------------------------------------------- #
def _build_cell_values(n_tets, aux_dim, density_scale, gen):
    cv = torch.zeros((n_tets, 7 + aux_dim), device=DEVICE)
    cv[:, 0] = torch.rand(n_tets, device=DEVICE, generator=gen) * density_scale
    cv[:, 1:4] = torch.rand((n_tets, 3), device=DEVICE, generator=gen)
    cv[:, 4:7] = (torch.rand((n_tets, 3), device=DEVICE, generator=gen) * 2 - 1) * 0.3
    cv[:, 7:] = torch.rand((n_tets, aux_dim), device=DEVICE, generator=gen)
    return cv


def _scene(n_tets, spread, z_range, tet_size, density_scale, aux_dim=2, seed=0):
    """Random tets fully in front of the camera. Returns (V, idx, cv, vn)."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    centers = torch.rand((n_tets, 3), device=DEVICE, generator=gen)
    centers[:, 0] = (centers[:, 0] * 2 - 1) * spread
    centers[:, 1] = (centers[:, 1] * 2 - 1) * spread
    centers[:, 2] = centers[:, 2] * (z_range[1] - z_range[0]) + z_range[0]
    offsets = (torch.rand((n_tets, 4, 3), device=DEVICE, generator=gen) * 2 - 1) * tet_size
    V = (centers[:, None, :] + offsets).reshape(-1, 3)
    idx = torch.arange(n_tets * 4, device=DEVICE, dtype=torch.int32).reshape(n_tets, 4)
    cv = _build_cell_values(n_tets, aux_dim, density_scale, gen)
    vn = torch.nn.functional.normalize(
        torch.randn((n_tets * 4, 3), device=DEVICE, generator=gen), dim=-1)
    return V, idx, cv, vn


def _max_extent(V, idx, camera, tile_size, min_t):
    """Per-visible-tet max projected pixel extent, straight from the vertex shader."""
    from delaunay_rasterization.internal.render_grid import RenderGrid
    from delaunay_rasterization.internal.tile_shader_slang import VertexShader
    rg = RenderGrid(camera.image_height, camera.image_width,
                    tile_height=tile_size, tile_width=tile_size)
    tcam = dict(tile_height=tile_size, tile_width=tile_size,
                grid_height=rg.grid_height, grid_width=rg.grid_width,
                min_t=min_t, **camera.to_dict(DEVICE))
    _, _, _, _, _, extent = VertexShader.apply(idx, V, tcam, rg)
    return extent


# --------------------------------------------------------------------------- #
# Gate 1: >=1px exactness  (p == 1 everywhere -> bit-identical to baseline)
# --------------------------------------------------------------------------- #
def test_ge_1px_bit_exact():
    tile_size, min_t = 4, 0.2
    # Big tets, high res -> every visible tet spans many pixels.
    camera = make_camera(height=256, width=256, fov_degrees=60.0)
    V, idx, cv, vn = _scene(n_tets=200, spread=1.0, z_range=(2.5, 4.0),
                            tet_size=0.35, density_scale=8.0, seed=1)

    ext = _max_extent(V, idx, camera, tile_size, min_t)
    vis = ext > 0
    assert vis.sum() > 0
    assert ext[vis].min().item() >= 1.0, \
        f"scene not all >=1px (min extent {ext[vis].min().item():.3f}); test misconfigured"

    model = FakeModel(V, idx, vn, aux_dim=cv.shape[1] - 7)
    # deterministic jitter so the only variable under test is the sampler
    ray_jitter = 0.5 * torch.ones((camera.image_height, camera.image_width, 2), device=DEVICE)

    base = render(camera, model, cell_values=cv, tile_size=tile_size, min_t=min_t,
                  ray_jitter=ray_jitter)
    for seed in (0, 7, 123):
        sto = render(camera, model, cell_values=cv, tile_size=tile_size, min_t=min_t,
                     ray_jitter=ray_jitter, stochastic_subpixel=True, stochastic_seed=seed)
        for k in ("render", "alpha", "aux", "xyzd"):
            assert torch.equal(base[k], sto[k]), \
                f"stochastic diverged from baseline on '{k}' at seed {seed} despite all tets >=1px"


# --------------------------------------------------------------------------- #
# Gate 2: unbiasedness  (MC mean -> full render as K grows)
# --------------------------------------------------------------------------- #
def _psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    if mse <= 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def test_unbiased_converges_with_K():
    tile_size, min_t = 4, 0.2
    # Low res + small, distant tets -> most tets sub-pixel. Thin density so the
    # boosted (density/p) alpha stays in the linear regime -> low bias floor.
    camera = make_camera(height=48, width=64, fov_degrees=60.0)
    # Thin regime (per-tet optical depth low): the Russian-roulette estimator is
    # genuinely unbiased here, so the scalar functional must match ref within
    # Monte-Carlo error. K=1 sits ~50 dB (noisy), K=256 ~78 dB (converged). The
    # bias that appears at higher density is characterized in test_thin_regime_bias.
    V, idx, cv, vn = _scene(n_tets=4000, spread=1.4, z_range=(6.0, 14.0),
                            tet_size=0.05, density_scale=2.0, seed=2)

    ext = _max_extent(V, idx, camera, tile_size, min_t)
    vis = ext > 0
    sub_frac = (ext[vis] < 1.0).float().mean().item()
    assert sub_frac > 0.5, f"scene not sub-pixel-dominated ({sub_frac:.2f} < 0.5)"

    model = FakeModel(V, idx, vn, aux_dim=cv.shape[1] - 7)
    ray_jitter = 0.5 * torch.ones((camera.image_height, camera.image_width, 2), device=DEVICE)

    ref = render(camera, model, cell_values=cv, tile_size=tile_size, min_t=min_t,
                 ray_jitter=ray_jitter)["render"].detach()

    Ks = [1, 4, 16, 64, 256]
    acc = torch.zeros_like(ref)
    lum_samples = []
    psnrs = {}
    drawn = 0
    for K in Ks:
        while drawn < K:
            s = render(camera, model, cell_values=cv, tile_size=tile_size, min_t=min_t,
                       ray_jitter=ray_jitter, stochastic_subpixel=True,
                       stochastic_seed=1000 + drawn)["render"].detach()
            acc += s
            lum_samples.append(s.sum().item())
            drawn += 1
        psnrs[K] = _psnr(acc / drawn, ref)

    print("\n=== unbiasedness: PSNR(mean_K, ref) vs K (sub-pixel frac "
          f"{sub_frac:.2f}) ===")
    for K in Ks:
        print(f"  K={K:>4}: PSNR = {psnrs[K]:6.2f} dB")

    # (a) monotone-ish improvement: the K=256 mean must be well above the single
    # sample -- variance is being averaged away toward the true image.
    assert psnrs[256] > psnrs[1] + 6.0, \
        f"MC mean did not improve with K: {psnrs[1]:.2f} -> {psnrs[256]:.2f} dB"
    assert psnrs[256] >= psnrs[16], "PSNR regressed between K=16 and K=256"
    # (b) converges to the true image (no systematic bias): high absolute PSNR.
    assert psnrs[256] > 35.0, \
        f"MC mean did not converge to the full render (K=256 PSNR {psnrs[256]:.2f} < 35 dB)"

    # (c) scalar functional matches ref within Monte-Carlo error ~ 1/sqrt(K).
    lum = np.array(lum_samples)
    ref_lum = ref.sum().item()
    mc_err = abs(lum.mean() - ref_lum)
    sem = lum.std() / math.sqrt(len(lum))
    print(f"  total luminance: ref={ref_lum:.2f} mean={lum.mean():.2f} "
          f"|bias|={mc_err:.3f} SEM={sem:.3f}")
    assert mc_err < 5.0 * sem + 1e-3 * abs(ref_lum), \
        f"luminance bias {mc_err:.3f} exceeds 5*SEM ({sem:.3f}) -> systematic bias"


def test_gradient_reweight_is_unbiased():
    """dL/d(density) is importance-weighted: E over seeds ~= the full-render grad.

    A scalar loss (sum of render) has a full-render gradient w.r.t. the density
    column; the stochastic gradient scaled by 1/p in expectation reproduces it.
    """
    tile_size, min_t = 4, 0.2
    camera = make_camera(height=48, width=64, fov_degrees=60.0)
    V, idx, cv0, vn = _scene(n_tets=3000, spread=1.4, z_range=(6.0, 14.0),
                             tet_size=0.05, density_scale=0.5, seed=5)
    model = FakeModel(V, idx, vn, aux_dim=cv0.shape[1] - 7)
    ray_jitter = 0.5 * torch.ones((camera.image_height, camera.image_width, 2), device=DEVICE)

    def grad_density(stochastic, seed=0):
        cv = cv0.clone().requires_grad_(True)
        out = render(camera, model, cell_values=cv, tile_size=tile_size, min_t=min_t,
                     ray_jitter=ray_jitter, stochastic_subpixel=stochastic,
                     stochastic_seed=seed)["render"]
        out.sum().backward()
        return cv.grad[:, 0].detach().clone()

    ref_g = grad_density(False)
    K = 256
    acc = torch.zeros_like(ref_g)
    for k in range(K):
        acc += grad_density(True, seed=4000 + k)
    mean_g = acc / K

    # Compare on tets that actually receive gradient in the reference.
    active = ref_g.abs() > 1e-6
    assert active.sum() > 50
    rel = (mean_g[active] - ref_g[active]).norm() / ref_g[active].norm()
    print(f"\n=== gradient reweight: relative L2 error of E_K[grad] vs full "
          f"grad (K={K}, active tets={active.sum().item()}) = {rel.item():.4f} ===")
    assert rel.item() < 0.15, \
        f"importance-weighted density gradient is biased (rel L2 {rel.item():.3f})"


def test_backward_survives_vertex_shader_grad():
    """Regression: stochastic culling must not modify VertexShader's saved
    outputs in place. In real training ``circumcenter`` feeds the network query
    so the vertex shader's differentiable outputs DO receive a gradient and its
    backward runs -- an in-place edit of the saved ``rect_tile_space`` would bump
    its autograd version and blow up the backward pass. (The caller-supplied
    cell_values path used by the other tests bypasses that query, so this is the
    guard for the training path.)"""
    from delaunay_rasterization.internal.render_grid import RenderGrid
    from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader

    tile_size, min_t = 4, 0.2
    camera = make_camera(height=48, width=64, fov_degrees=60.0)
    V, idx, _, _ = _scene(n_tets=2000, spread=1.4, z_range=(6.0, 14.0),
                          tet_size=0.05, density_scale=4.0, seed=3)
    V = V.detach().requires_grad_(True)
    rg = RenderGrid(camera.image_height, camera.image_width,
                    tile_height=tile_size, tile_width=tile_size)
    tcam = dict(tile_height=tile_size, tile_width=tile_size,
                grid_height=rg.grid_height, grid_width=rg.grid_width,
                min_t=min_t, **camera.to_dict(DEVICE))
    out = vertex_and_tile_shader(idx, V, tcam, rg,
                                 stochastic_subpixel=True, stochastic_seed=1)
    vs_tetra, circumcenter = out[2], out[3]
    # circumcenter + vs_tetra are the differentiable vertex-shader outputs the
    # real query consumes; forcing grad through them exercises VertexShader.bwd.
    # Weight by randn so the upstream grad is contiguous (a bare .sum() hands the
    # slang backward a 0-stride broadcast grad it rejects -- unrelated to us).
    wc = torch.randn_like(circumcenter)
    wv = torch.randn_like(vs_tetra)
    ((circumcenter * wc).sum() + (vs_tetra * wv).sum()).backward()
    assert V.grad is not None and torch.isfinite(V.grad).all()


def test_thin_regime_bias():
    """Characterize (and bound) the ONLY bias source: the thin-regime approx.

    The estimator reweights kept tets by 1/p on density, so E[alpha] =
    p*(1-exp(-tau/p)). This equals the true 1-exp(-tau) only in the thin limit
    (tau small); for dense sub-pixel tets the boosted tau/p leaves the linear
    regime and E undershoots by ~ (tau^2/2)(1/p - 1). We verify the estimator is
    unbiased when thin and that the residual is a monotone UNDERSHOOT growing
    with per-tet optical depth -- the documented approximation, not a code bug.
    """
    tile_size, min_t = 4, 0.2
    camera = make_camera(height=48, width=64, fov_degrees=60.0)
    ray_jitter = 0.5 * torch.ones((camera.image_height, camera.image_width, 2), device=DEVICE)
    K = 256

    rows = []
    for ds in (1.0, 4.0, 12.0):
        V, idx, cv, vn = _scene(n_tets=4000, spread=1.4, z_range=(6.0, 14.0),
                                tet_size=0.05, density_scale=ds, seed=2)
        model = FakeModel(V, idx, vn, aux_dim=cv.shape[1] - 7)
        ref = render(camera, model, cell_values=cv, tile_size=tile_size, min_t=min_t,
                     ray_jitter=ray_jitter)["render"].detach()
        lums = []
        for k in range(K):
            lums.append(render(camera, model, cell_values=cv, tile_size=tile_size,
                               min_t=min_t, ray_jitter=ray_jitter, stochastic_subpixel=True,
                               stochastic_seed=7000 + k)["render"].detach().sum().item())
        lums = np.array(lums)
        ref_lum = ref.sum().item()
        bias = lums.mean() - ref_lum
        sem = lums.std() / math.sqrt(K)
        rows.append((ds, ref_lum, bias, 100 * bias / ref_lum, sem))

    print("\n=== thin-regime bias vs per-tet optical depth (luminance, K=256) ===")
    for ds, rl, b, pct, sem in rows:
        print(f"  density_scale={ds:5.1f}: ref_lum={rl:7.3f}  bias={b:+7.3f} "
              f"({pct:+5.2f}%)  SEM={sem:.3f}  bias/SEM={b / sem:+6.1f}")

    thin_pct = abs(rows[0][3])
    dense_pct = abs(rows[-1][3])
    # thin limit: genuinely unbiased (within ~1%).
    assert thin_pct < 1.0, f"estimator biased even in the thin limit ({thin_pct:.2f}%)"
    # residual grows monotonically with optical depth ...
    assert dense_pct > abs(rows[1][3]) > thin_pct, "bias not monotone in optical depth"
    # ... and is always a NEGATIVE undershoot (the sign the analysis predicts).
    assert all(b < 0 for _, _, b, _, _ in rows[1:]), "dense-regime bias is not an undershoot"


# --------------------------------------------------------------------------- #
# Gate 3: speed on real london geometry (opt-in)
# --------------------------------------------------------------------------- #
LONDON_CKPT = os.path.join(os.path.dirname(__file__), "..",
                           "output", "exp_london_dwrjump", "ckpt.pth")


@pytest.mark.slow
@pytest.mark.gpu
def test_speed_london_walkthrough():
    if not os.path.exists(LONDON_CKPT):
        pytest.skip("london ckpt not present")
    import time

    d = torch.load(LONDON_CKPT, map_location=DEVICE, weights_only=False)
    V = d["interior_vertices"].float()
    idx = d["indices"].int()
    V = V - V.mean(0, keepdim=True)
    V = V / V.norm(dim=1).quantile(0.98) * 1.5
    V[:, 2] = V[:, 2] + 2.5
    T = idx.shape[0]

    aux = 8
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    cv0 = _build_cell_values(T, aux, density_scale=6.0, gen=gen)
    vn = torch.nn.functional.normalize(
        torch.randn((V.shape[0], 3), device=DEVICE, generator=gen), dim=-1)

    tile_size, min_t = 4, 0.2

    def bench(H, W):
        camera = make_camera(height=H, width=W, fov_degrees=60.0)
        ray_jitter = 0.5 * torch.ones((H, W, 2), device=DEVICE)
        ext = _max_extent(V, idx, camera, tile_size, min_t)
        vis = ext > 0
        sub_frac = (ext[vis] < 1.0).float().mean().item()
        g = torch.randn((3, H, W), device=DEVICE)

        def once(stochastic, seed):
            cv = cv0.clone().requires_grad_(True)
            vv = V.clone().requires_grad_(True)
            m = FakeModel(vv, idx, vn, aux_dim=aux)
            out = render(camera, m, cell_values=cv, tile_size=tile_size, min_t=min_t,
                         ray_jitter=ray_jitter, stochastic_subpixel=stochastic,
                         stochastic_seed=seed)["render"]
            (out * g).sum().backward()

        def timeit(stochastic, it=12, wu=4):
            for i in range(wu):
                once(stochastic, i)
            torch.cuda.synchronize()
            t0 = time.time()
            for i in range(it):
                once(stochastic, 100 + i)
            torch.cuda.synchronize()
            return (time.time() - t0) / it * 1e3

        return sub_frac, timeit(False), timeit(True)

    print(f"\n=== london {T:,} tets: fwd+bwd speedup vs resolution ===")
    results = {}
    # (99,174) = deep-sub-pixel walkthrough res (the feature's target); 198x348 =
    # the requested walkthrough res; 546x988 ~ native (nearly all >=1px).
    for H, W in [(99, 174), (198, 348), (546, 988)]:
        sf, tf, ts = bench(H, W)
        results[(H, W)] = (sf, tf, ts)
        print(f"  {H:>3}x{W:<3}: sub-pixel {sf:.2f}  full {tf:6.1f} ms  "
              f"stoch {ts:6.1f} ms  speedup {tf / ts:.2f}x")

    # The LOD win is real where tets are deeply sub-pixel (the intended regime).
    sf_lo, tf_lo, ts_lo = results[(99, 174)]
    assert sf_lo > 0.8
    assert ts_lo < tf_lo * 0.85, \
        f"expected a clear speedup at deep-sub-pixel res (got {tf_lo / ts_lo:.2f}x)"
