"""Component-wise comparison of torch vs JAX-quadrature distortion state.

Single tet, fixed camera; dumps the torch distortion_img state (x,y,z,w,v)
at a hit pixel and recomputes the same recursion from quadrature in numpy.
Run: CUDA_VISIBLE_DEVICES=2 uv run python analysis/debug_dist_loss.py
"""

import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from data.camera import Camera  # noqa: E402
from utils.train_util import render  # noqa: E402
from tests.interval_parity_test import FakeModel  # noqa: E402

H = W = 33
TMIN = 0.01


def make_camera():
    fovx = math.radians(60.0)
    fovy = 2 * math.atan(math.tan(fovx / 2) * H / W)
    return Camera(colmap_id=0, R=np.eye(3), T=np.zeros(3), fovx=fovx, fovy=fovy,
                  image=torch.zeros((3, H, W), device="cuda"), uid=0, cx=-1, cy=-1,
                  data_device="cuda")


def tukey(x, p=0.1):
    xs = np.abs(x) / max(1e-20, abs(p - 1))
    return np.sign(x) * abs(p - 1) / p * ((xs + 1) ** p - 1)


def scale_dist(x):
    return tukey(160 * x)


def main():
    cam = make_camera()
    verts = torch.tensor([
        [-0.4, -0.4, 3.0],
        [0.5, -0.3, 3.2],
        [0.0, 0.5, 3.1],
        [0.05, 0.0, 3.8],
    ], device="cuda")
    idx = torch.tensor([[0, 1, 2, 3]], device="cuda", dtype=torch.int32)
    cv = torch.zeros((1, 7), device="cuda")
    cv[0, 0] = 5.0
    cv[0, 1:4] = 0.5
    model = FakeModel(verts, idx, torch.zeros_like(verts), aux_dim=0)

    pkg = render(cam, model, cell_values=cv, tile_size=8, min_t=TMIN)

    # pick the center pixel
    py, px = H // 2, W // 2
    # re-render to grab distortion_img directly
    from delaunay_rasterization.internal.alphablend_tiled_slang_interp import AlphaBlendTiledRender
    from delaunay_rasterization.internal.render_grid import RenderGrid
    from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader

    rg = RenderGrid(H, W, tile_height=8, tile_width=8)
    tcam = dict(tile_height=8, tile_width=8, grid_height=rg.grid_height,
                grid_width=rg.grid_width, min_t=TMIN, **cam.to_dict("cuda"))
    sorted_idx, tile_ranges, _, _, _, _ = vertex_and_tile_shader(idx, verts, tcam, rg)
    rj = 0.5 * torch.ones((H, W, 2), device="cuda")
    out, xyzd, dist_img, alive, med = AlphaBlendTiledRender.apply(
        sorted_idx, tile_ranges, idx, verts, torch.zeros_like(verts), cv, rg, tcam, rj, 0)
    print("torch distortion state [x,y,z,w,v] @ center:", dist_img[py, px].tolist())
    print("torch alpha @ center:", (1 - out[py, px, 3].exp()).item())
    x_, y_, z_, w_, v_ = dist_img[py, px].tolist()
    print("torch pixel loss:", max(x_ - y_ + v_, 0) / max(z_ * z_, 1e-6))

    # --- numpy quadrature replication along the center pixel ray ---
    # build the same ray the shader uses (pixel center + 0.5 jitter)
    fx = W / (2 * math.tan(math.radians(60.0) / 2))
    fy = fx
    cx_, cy_ = W / 2, H / 2
    pix = np.array([px + 0.5, py + 0.5])
    d_cam = np.array([(pix[0] - cx_) / fx, (pix[1] - cy_) / fy, 1.0])
    d_cam /= np.linalg.norm(d_cam)
    o = np.zeros(3) + d_cam * TMIN  # ray starts at origin + min_t*d (camera at origin)

    v_np = verts.cpu().numpy()
    # ray-tet intersection via the 4 faces (plane tests)
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    t_enter, t_exit = -1e9, 1e9
    for f in faces:
        p0, p1, p2 = v_np[list(f)]
        n = np.cross(p1 - p0, p2 - p0)
        # orient outward: opposite vertex inside
        opp = v_np[[i for i in range(4) if i not in f][0]]
        if np.dot(n, opp - p0) > 0:
            n = -n
        denom = np.dot(n, d_cam)
        tval = np.dot(n, p0 - o) / denom
        if denom < 0:
            t_enter = max(t_enter, tval)
        else:
            t_exit = min(t_exit, tval)
    print("ray-tet enter/exit t:", t_enter, t_exit)

    sigma = 5.0
    # torch exact segment moments (single segment, T=1)
    a, b = scale_dist(t_enter), scale_dist(t_exit)
    u = sigma * (b - a)
    alpha = 1 - np.exp(-u)
    m0 = alpha
    m1 = a * alpha + (1 - (1 + u) * np.exp(-u)) / sigma
    selfv = (1 - 2 * u * np.exp(-u) - np.exp(-2 * u)) / sigma
    print("exact single-segment: m0", m0, "m1", m1, "self", selfv)
    print("exact pixel loss:", max(0 + selfv, 0) / max(m0 * m0, 1e-6))

    # quadrature version of my port
    N = 10000
    z0 = v_np[:, 2].min()
    start = max(z0, TMIN)
    end = v_np[:, 2].max() * 1.2
    t = np.linspace(start, end, N + 1)
    tm = 0.5 * (t[1:] + t[:-1])
    sig = np.where((tm > t_enter + TMIN) & (tm < t_exit + TMIN), sigma, 0.0)
    # NOTE: jax t is distance from camera origin; torch t from o=origin+TMIN*d
    s = scale_dist(t - TMIN)
    dt = np.diff(t)
    od = sig * dt
    T = np.exp(-np.concatenate([[0], np.cumsum(od[:-1])]))
    aq, bq = s[:-1], s[1:]
    uq = sig * (bq - aq)
    e = np.exp(-uq)
    al = 1 - e
    m0q = T * al
    m1q = T * (aq * al + np.where(sig > 0, (1 - (1 + uq) * e) / np.where(sig > 0, sig, 1), 0))
    sq = T * T * np.where(sig > 0, (1 - 2 * uq * e - e * e) / np.where(sig > 0, sig, 1), 0)
    Zp = np.concatenate([[0], np.cumsum(m0q[:-1])])
    Wp = np.concatenate([[0], np.cumsum(m1q[:-1])])
    xq = (2 * m1q * Zp).sum()
    yq = (2 * m0q * Wp).sum()
    vq = sq.sum()
    Zq = m0q.sum()
    Wq = m1q.sum()
    print("quad recursion [x,y,z,w,v]:", [xq, yq, Zq, Wq, vq])
    print("quad pixel loss:", max(xq - yq + vq, 0) / max(Zq * Zq, 1e-6))


if __name__ == "__main__":
    main()
