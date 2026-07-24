"""Correctness harness for the hand-written CUDA alphablend backward.

The CUDA kernel (delaunay_rasterization/internal/cuda/alphablend_bwd.cu) replaces
ONLY the Slang reverse pass; it reads the same saved images the Slang backward
reads and walks the composite backward via undo_alpha_out. Because both the Slang
backward and this kernel scatter grads with non-deterministic per-scalar atomics
and reconstruct state in float32, BITWISE parity is impossible (see the design
doc + project-renderer-backward-roofline memory). The contract here mirrors
triton_hashgrid_test.py's contended-scatter contract:

  * per-primitive VJPs (A-H) vs a double-precision autograd oracle: tight
    (the analytic derivatives must BE the derivative of the forward);
  * full-chain: cosine ~1 + per-column-sum parity vs the Slang backward.

The one channel that legitimately drifts is `density` at *extreme* composite
depth (100s of tets): its grad flows through the accumulated log-transmittance
cotangent, which compounds float32 reconstruction ULP differences between the
two forward implementations. At realistic depth it matches ground truth to
cos>0.999 (test_density_matches_double_ground_truth pins this).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

DEV = "cuda"
DT = torch.float64


@pytest.fixture(scope="module")
def cu():
    try:
        from delaunay_rasterization.internal.cuda import alphablend_bwd_cuda
    except Exception as e:  # compile failure off-rig
        pytest.skip(f"cuda backward extension unavailable: {e}")
    return alphablend_bwd_cuda


# --------------------------------------------------------------------------- refs
def _scale_dist(x):
    p = 0.1
    z = 160 * x
    c = abs(p - 1.0)
    xs = z.abs() / c
    return torch.sign(z) * c / p * ((xs + 1) ** p - 1)


def _m0(a, b, d, T):
    dt = (b - a).clamp_min(0)
    return T * (1 - torch.exp(-d * dt)).clamp_min(0)


def _m1(a, b, d, T):
    dt = (b - a).clamp_min(0)
    u = d * dt
    e = torch.exp(-u)
    return T * (a * (1 - e).clamp_min(0) + (1 - (1 + u) * e) / d)


def _self(a, b, d, T):
    dt = (b - a).clamp_min(0)
    u = d * dt
    e = torch.exp(-u)
    return T * T * (1 - 2 * u * e - e * e) / d


def update_alpha_out_ref(prev, ctrl, AUX):
    """Double-precision transliteration of update_alpha_out (primitives A,B,C,E)."""
    T = torch.exp(prev[:, 3])
    rgb = prev[:, 0:3] + ctrl[:, 0:3] * T[:, None]
    a = prev[:, 3] - ctrl[:, 3]
    d0, d1, dens = ctrl[:, 4], ctrl[:, 5], ctrl[:, 6]
    st = _scale_dist(d0)
    en = _scale_dist(torch.maximum(d0, d1))
    m0 = _m0(st, en, dens, T)
    m1 = _m1(st, en, dens, T)
    sf = _self(st, en, dens, T)
    ox, oy, oz, ow, ov = prev[:, 4], prev[:, 5], prev[:, 6], prev[:, 7], prev[:, 8]
    ds = torch.stack([ox + 2 * m1 * oz, oy + 2 * m0 * ow, oz + m0, ow + m1, ov + sf], 1)
    xyzd = prev[:, 9:13] + T[:, None] * ctrl[:, 7:11]
    aux = prev[:, 13:13 + AUX] + ctrl[:, 11:11 + AUX] * T[:, None]
    return torch.cat([rgb, a[:, None], ds, xyzd, aux], 1)


def _cross(a, b):
    return torch.stack([a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0]])


def _phi(x):
    if abs(x.item()) < 1e-6:
        return 1.0 - x / 2
    return -torch.expm1(-x) / x


def _cint(c0, c1, ddt):
    a = torch.exp(-ddt)
    pv = _phi(ddt)
    return (pv - a) * c0 + (1 - pv) * c1


def evaluate_ref(verts, base, grd, density, aux, vn, o, d, ortho):
    """Double-precision transliteration of evaluate_tetra_interp (integrate + intersect)."""
    FA, FB, FC, FD = [0, 0, 0, 1], [1, 1, 2, 2], [2, 3, 3, 3], [3, 2, 1, 0]
    t_enter = torch.tensor(-1e30, dtype=DT, device=DEV)
    t_exit = torch.tensor(1e30, dtype=DT, device=DEV)
    ef = xf = -1
    te_v, tx_v, eps = -1e30, 1e30, 1e-10
    for i in range(4):
        vA, vB, vC, vD = verts[FA[i]], verts[FB[i]], verts[FC[i]], verts[FD[i]]
        n = _cross(vB - vA, vC - vA)
        if torch.dot(n, vD - vA).item() > 0:
            n = -n
        dd = -torch.dot(n, vA - o)
        denom = torch.dot(n, d)
        if abs(denom.item()) < eps:
            if dd.item() > 0:
                return None
        else:
            tp = -dd / denom
            if denom.item() < 0:
                if tp.item() > te_v:
                    te_v, t_enter, ef = tp.item(), tp, i
            else:
                if tp.item() < tx_v:
                    tx_v, t_exit, xf = tp.item(), tp, i
    if te_v > tx_v or tx_v <= 0:
        return None
    enter = torch.clamp(t_enter, min=0.0)
    exit = torch.maximum(t_exit, enter)
    dt = torch.clamp(exit - enter, min=0.0)
    v_enter = enter * d + o - verts[0]
    v_exit = exit * d + o - verts[0]
    ge = torch.dot(grd, v_enter)
    gx = torch.dot(grd, v_exit)
    c_enter = torch.clamp(ge + base, min=0)
    c_exit = torch.clamp(gx + base, min=0)
    ddt = dt * density
    final_color = _cint(c_exit, c_enter, ddt)
    alpha = torch.clamp(-torch.expm1(-density * dt), min=0)
    e1, e2, e3 = verts[1] - verts[0], verts[2] - verts[0], verts[3] - verts[0]
    c23, c31, c12 = _cross(e2, e3), _cross(e3, e1), _cross(e1, e2)
    inv_det = 1.0 / torch.dot(e1, c23)
    b1e = torch.dot(v_enter, c23) * inv_det
    b2e = torch.dot(v_enter, c31) * inv_det
    b3e = torch.dot(v_enter, c12) * inv_det
    n_enter = (1 - b1e - b2e - b3e) * vn[0] + b1e * vn[1] + b2e * vn[2] + b3e * vn[3]
    b1x = torch.dot(v_exit, c23) * inv_det
    b2x = torch.dot(v_exit, c31) * inv_det
    b3x = torch.dot(v_exit, c12) * inv_det
    n_exit = (1 - b1x - b2x - b3x) * vn[0] + b1x * vn[1] + b2x * vn[2] + b3x * vn[3]
    normal_contrib = _cint(n_exit, n_enter, ddt)
    depth = _cint(exit, enter, ddt) / ortho
    rgba = torch.cat([final_color, (density * dt).reshape(1)])
    dist = torch.stack([enter, exit])
    xyzd = torch.cat([normal_contrib, depth.reshape(1)])
    return rgba, dist, density, xyzd, aux * alpha


# --------------------------------------------------------------------------- A-E
def test_update_alpha_out_vjp(cu):
    """Primitives A (pixel), B (xyzd), C (aux), E (5-moment distortion)."""
    torch.manual_seed(0)
    AUX, N, PS, CS = 2, 128, 15, 13
    prev = (torch.randn(N, PS, device=DEV) * 0.3).double()
    prev[:, 3] = -torch.rand(N, device=DEV).double() * 3          # logT in [-3,0]
    ctrl = (torch.randn(N, CS, device=DEV) * 0.2).double()
    ctrl[:, 6] = torch.rand(N, device=DEV).double() * 5 + 0.1     # density>0
    ctrl[:, 4] = torch.rand(N, device=DEV).double() * 0.5 + 0.1   # dist0
    ctrl[:, 5] = ctrl[:, 4] + torch.rand(N, device=DEV).double() * 0.5 + 0.05  # dist1>=dist0
    d_res = torch.randn(N, PS, device=DEV).double()

    pv = prev.clone().requires_grad_(True)
    cv = ctrl.clone().requires_grad_(True)
    (update_alpha_out_ref(pv, cv, AUX) * d_res).sum().backward()

    d_prev, d_ctrl = cu.test_update_bwd(prev.float(), ctrl.float(), d_res.float(), AUX)
    torch.testing.assert_close(d_prev.double(), pv.grad, rtol=2e-3, atol=1e-4)
    torch.testing.assert_close(d_ctrl.double(), cv.grad, rtol=2e-3, atol=1e-4)


# --------------------------------------------------------------------------- F,G,H
def test_evaluate_vjp(cu):
    """Primitives F (integrate), G (intersect branch-selected d_verts), H (scatter
    gates). d_verts is float32-conditioning-limited on near-degenerate tets, so it
    gets a looser bound; material/normal grads are tight."""
    torch.manual_seed(3)
    AUX, N = 2, 48
    verts = torch.zeros(N, 4, 3, device=DEV)
    for i in range(N):
        z = 1.0 + i * 0.05
        base_c = torch.tensor([0., 0., z], device=DEV)
        off = torch.tensor([[-.25, -.25, -.15], [.3, -.2, .1], [-.2, .3, .12], [.15, .18, .25]], device=DEV)
        verts[i] = base_c + off + torch.randn(4, 3, device=DEV) * 0.03
    base = torch.rand(N, 3, device=DEV)
    grd = (torch.rand(N, 3, device=DEV) * 2 - 1) * 0.3
    density = torch.rand(N, device=DEV) * 5 + 0.2
    aux = torch.rand(N, AUX, device=DEV)
    vn = torch.nn.functional.normalize(torch.randn(N, 4, 3, device=DEV), dim=-1)
    ray = [0., 0., 0., 0., 0., 1., 1.0]
    o = torch.tensor(ray[0:3], dtype=DT, device=DEV)
    d = torch.tensor(ray[3:6], dtype=DT, device=DEV)
    ortho = torch.tensor(ray[6], dtype=DT, device=DEV)

    rgba, dist, dens, xyzd, auxo, hit = cu.test_eval_fwd(verts, base, grd, density, aux, vn, ray, 0.2)
    assert hit.sum().item() > N // 2, "test scene must hit most tets"
    d_rgba = torch.randn(N, 4, device=DEV)
    d_dist = torch.randn(N, 2, device=DEV)
    d_densc = torch.randn(N, device=DEV)
    d_xyzd = torch.randn(N, 4, device=DEV)
    d_aux = torch.randn(N, AUX, device=DEV)
    dg_v, dg_b, dg_g, dg_d, dg_a, dg_n = cu.test_eval_bwd(
        verts, base, grd, density, aux, vn, ray, 0.2, d_rgba, d_dist, d_densc, d_xyzd, d_aux)

    max_err = {"verts": 0., "base": 0., "grd": 0., "density": 0., "aux": 0., "vn": 0.}
    mag = dict(max_err)
    for i in range(N):
        if not hit[i]:
            continue
        vv = verts[i].to(DT).clone().requires_grad_(True)
        bb = base[i].to(DT).clone().requires_grad_(True)
        gg = grd[i].to(DT).clone().requires_grad_(True)
        dd = density[i].to(DT).clone().requires_grad_(True)
        aa = aux[i].to(DT).clone().requires_grad_(True)
        nn = vn[i].to(DT).clone().requires_grad_(True)
        r = evaluate_ref(vv, bb, gg, dd, aa, nn, o, d, ortho)
        if r is None:
            continue
        rr, di, de, xy, ax = r
        L = ((rr * d_rgba[i].to(DT)).sum() + (di * d_dist[i].to(DT)).sum()
             + de * d_densc[i].to(DT) + (xy * d_xyzd[i].to(DT)).sum()
             + (ax * d_aux[i].to(DT)).sum())
        L.backward()
        for key, cug, ref in [("verts", dg_v[i], vv.grad), ("base", dg_b[i], bb.grad),
                              ("grd", dg_g[i], gg.grad), ("density", dg_d[i], dd.grad),
                              ("aux", dg_a[i][:AUX], aa.grad), ("vn", dg_n[i], nn.grad)]:
            max_err[key] = max(max_err[key], (cug.to(DT) - ref).abs().max().item())
            mag[key] = max(mag[key], ref.abs().max().item())
    # material / normal grads: exact derivative of forward -> tight
    for key in ("base", "grd", "density", "aux", "vn"):
        assert max_err[key] / (mag[key] + 1e-9) < 1e-3, f"{key} VJP off: {max_err[key]:.2e}"
    # verts: intersect + Cramer inverse -> float32 conditioning on thin tets
    assert max_err["verts"] / (mag["verts"] + 1e-9) < 5e-2, f"verts VJP off: {max_err['verts']:.2e}"


# --------------------------------------------------------------------------- full chain
def _make_scene(n_tets, AUX, seed, spread=0.35, zr=(1.0, 5.0), sz=0.15, dens=6.0):
    torch.manual_seed(seed)
    c = torch.rand((n_tets, 3), device=DEV)
    c[:, 0] = (c[:, 0] * 2 - 1) * spread
    c[:, 1] = (c[:, 1] * 2 - 1) * spread
    c[:, 2] = c[:, 2] * (zr[1] - zr[0]) + zr[0]
    off = (torch.rand((n_tets, 4, 3), device=DEV) * 2 - 1) * sz
    verts = (c[:, None, :] + off).reshape(-1, 3)
    idx = torch.arange(n_tets * 4, device=DEV, dtype=torch.int32).reshape(n_tets, 4)
    cv = torch.zeros((n_tets, 7 + AUX), device=DEV)
    cv[:, 0] = torch.rand(n_tets, device=DEV) * dens
    cv[:, 1:4] = torch.rand((n_tets, 3), device=DEV)
    cv[:, 4:7] = (torch.rand((n_tets, 3), device=DEV) * 2 - 1) * 0.4
    cv[:, 7:] = torch.rand((n_tets, AUX), device=DEV)
    vn = torch.nn.functional.normalize(torch.randn_like(verts), dim=-1)
    return verts, idx, cv, vn


def _make_camera(H, W):
    from data.camera import Camera
    fovx = math.radians(60.0)
    fovy = 2 * math.atan(math.tan(fovx / 2) * H / W)
    return Camera(colmap_id=0, R=np.eye(3), T=np.zeros(3), fovx=fovx, fovy=fovy,
                  image=torch.zeros((3, H, W), device=DEV), gt_alpha_mask=None,
                  uid=0, cx=-1, cy=-1, data_device=DEV)


class _FakeModel:
    def __init__(self, v, idx, vn, aux):
        self.vertices, self.indices, self.vertex_normals = v, idx, vn
        self.additional_attr, self.device = aux, DEV
        self.mask_values, self.feature_dim = False, 7 + aux


@pytest.mark.parametrize("n_tets,AUX", [(60, 2), (60, 8), (800, 8)])
def test_fullchain_parity_vs_slang(cu, n_tets, AUX):
    """CUDA backward vs the Slang backward through AlphaBlendTiledRender: cosine ~1
    and per-column-sum parity (contended-scatter contract)."""
    import delaunay_rasterization.internal.alphablend_tiled_slang_interp as AB
    from utils.train_util import render
    H = W = 32 if n_tets > 100 else 48
    camera = _make_camera(H, W)
    verts, idx, cv0, vn0 = _make_scene(n_tets, AUX, seed=1)
    target = torch.rand((3, H, W), device=DEV)

    def run(use_cuda):
        AB._CUDA_BWD_FLAG[0] = use_cuda
        try:
            v = verts.detach().clone().requires_grad_(True)
            cv = cv0.detach().clone().requires_grad_(True)
            n = vn0.detach().clone().requires_grad_(True)
            pkg = render(camera, _FakeModel(v, idx, n, AUX), cell_values=cv,
                         tile_size=4, min_t=0.2)
            loss = ((pkg["render"] - target).abs().mean() + 0.3 * pkg["aux"].mean()
                    + 0.2 * pkg["xyzd"][..., :3].abs().mean() + 0.05 * pkg["distortion_loss"])
            loss.backward()
            return v.grad.clone(), cv.grad.clone(), n.grad.clone()
        finally:
            AB._CUDA_BWD_FLAG[0] = False

    gs = run(False)
    gc = run(True)
    for name, s, c in zip(("vertices", "cell_values", "vert_normals"), gs, gc):
        assert torch.isfinite(c).all(), f"{name}: non-finite CUDA grad"
        cos = torch.nn.functional.cosine_similarity(s.reshape(1, -1), c.reshape(1, -1)).item()
        assert cos > 0.999, f"{name}: cosine {cos:.5f} vs Slang"
        # per-column-sum invariance (order-independent gradient mass)
        cs = (s.sum(0) - c.sum(0)).abs()
        den = s.sum(0).abs().clamp_min(1e-3)
        assert (cs / den).max().item() < 5e-2, f"{name}: column-sum parity off"


def test_density_matches_double_ground_truth(cu):
    """The density channel is the one sensitive to float32 reconstruction drift.
    At realistic composite depth it must match a double-precision full-composite
    ground truth as well as Slang does (both float32 implementations agree with GT
    to cos>0.999). Pins that there is no density *formula* bug."""
    import delaunay_rasterization.internal.alphablend_tiled_slang_interp as AB
    from utils.train_util import render
    from delaunay_rasterization.internal.render_grid import RenderGrid
    from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader

    AUX, K, H, W = 2, 40, 4, 4
    torch.manual_seed(11)
    c = torch.stack([torch.tensor([0., 0., 1.2 + i * 0.06], device=DEV) for i in range(K)])
    off = (torch.rand(K, 4, 3, device=DEV) * 2 - 1) * 0.5
    verts = (c[:, None, :] + off).reshape(-1, 3)
    idx = torch.arange(K * 4, device=DEV, dtype=torch.int32).reshape(K, 4)
    cv0 = torch.zeros(K, 7 + AUX, device=DEV)
    cv0[:, 0] = torch.rand(K, device=DEV) * 0.8       # low density -> deep composite
    cv0[:, 1:4] = torch.rand(K, 3, device=DEV)
    cv0[:, 4:7] = (torch.rand(K, 3, device=DEV) * 2 - 1) * 0.3
    cv0[:, 7:] = torch.rand(K, AUX, device=DEV)
    vn0 = torch.nn.functional.normalize(torch.randn_like(verts), dim=-1)
    camera = _make_camera(H, W)
    g_img = torch.randn(3 + AUX, H, W, device=DEV)
    g_xyzd = torch.randn(4, H, W, device=DEV) * 0.3

    def run(use_cuda):
        AB._CUDA_BWD_FLAG[0] = use_cuda
        try:
            cv = cv0.clone().requires_grad_(True)
            v = verts.clone().requires_grad_(True)
            n = vn0.clone().requires_grad_(True)
            pkg = render(camera, _FakeModel(v, idx, n, AUX), cell_values=cv,
                         tile_size=4, min_t=0.2)
            img = torch.cat([pkg["render"], pkg["aux"]], 0)
            (((img * g_img).sum() + (pkg["xyzd"].permute(2, 0, 1) * g_xyzd).sum())).backward()
            return cv.grad[:, 0].clone()
        finally:
            AB._CUDA_BWD_FLAG[0] = False

    d_slang = run(False)
    d_cuda = run(True)

    # double-precision full-composite ground truth over the actual sorted order
    rg = RenderGrid(H, W, tile_height=4, tile_width=4)
    tcam = dict(tile_height=4, tile_width=4, grid_height=rg.grid_height,
                grid_width=rg.grid_width, min_t=0.2, **camera.to_dict(DEV))
    sti, tr, *_ = vertex_and_tile_shader(idx, verts, tcam, rg)
    sti, tr = sti.long(), tr.long()
    Kc = camera.to_dict(DEV)["K"]
    wvt = camera.to_dict(DEV)["world_view_transform"]
    fx, fy, cx, cy = Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2]
    Vrot = wvt[:3, :3].t().to(DT)
    campos = camera.to_dict(DEV)["cam_pos"].to(DT)
    cvg = cv0.to(DT).clone().requires_grad_(True)
    vg = verts.to(DT).clone().requires_grad_(True)
    ng = vn0.to(DT).clone().requires_grad_(True)
    total = torch.zeros((), dtype=DT, device=DEV)
    for py in range(H):
        for px in range(W):
            tile = (py // 4) * rg.grid_width + (px // 4)
            s0, s1 = tr[tile, 0].item(), tr[tile, 1].item()
            dirc = torch.tensor([(px + 0.5 - cx) / fx, (py + 0.5 - cy) / fy, 1.0],
                                dtype=DT, device=DEV)
            dirc = dirc / dirc.norm()
            raw = Vrot @ dirc
            ortho = raw.norm()
            dw = raw / ortho
            o = campos + dw * 0.2
            ao = dict(rgb=torch.zeros(3, dtype=DT, device=DEV), a=torch.zeros((), dtype=DT, device=DEV),
                      ds=torch.zeros(5, dtype=DT, device=DEV), xyzd=torch.zeros(4, dtype=DT, device=DEV),
                      aux=torch.zeros(AUX, dtype=DT, device=DEV))
            for si in range(s0, s1):
                gg = sti[si].item()
                r = evaluate_ref(vg[idx[gg].long()], cvg[gg, 1:4], cvg[gg, 4:7],
                                 cvg[gg, 0].clamp_min(0), cvg[gg, 7:7 + AUX], ng[idx[gg].long()],
                                 o, dw, ortho)
                if r is None or not torch.isfinite(r[0][3]) or r[0][3].item() <= 0:
                    continue
                rgba, dist, dens, xyzd, auxo = r
                T = torch.exp(ao["a"])
                st = _scale_dist(dist[0])
                en = _scale_dist(torch.maximum(dist[0], dist[1]))
                m0 = _m0(st, en, dens, T)
                m1 = _m1(st, en, dens, T)
                sf = _self(st, en, dens, T)
                ds = ao["ds"]
                ao = dict(rgb=ao["rgb"] + rgba[:3] * T, a=ao["a"] - rgba[3],
                          ds=torch.stack([ds[0] + 2 * m1 * ds[2], ds[1] + 2 * m0 * ds[3],
                                          ds[2] + m0, ds[3] + m1, ds[4] + sf]),
                          xyzd=ao["xyzd"] + T * xyzd, aux=ao["aux"] + auxo * T)
                if ao["a"].item() < -5.5:
                    break
            img = torch.cat([ao["rgb"], ao["aux"]])
            total = total + (img * g_img[:, py, px].to(DT)).sum() + (ao["xyzd"] * g_xyzd[:, py, px].to(DT)).sum()
    total.backward()
    d_gt = cvg.grad[:, 0]

    cos_slang = torch.nn.functional.cosine_similarity(d_slang.double().reshape(1, -1), d_gt.reshape(1, -1)).item()
    cos_cuda = torch.nn.functional.cosine_similarity(d_cuda.double().reshape(1, -1), d_gt.reshape(1, -1)).item()
    assert cos_slang > 0.999, f"sanity: Slang density vs GT {cos_slang:.5f}"
    assert cos_cuda > 0.999, f"CUDA density vs double GT {cos_cuda:.5f} (Slang {cos_slang:.5f})"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
