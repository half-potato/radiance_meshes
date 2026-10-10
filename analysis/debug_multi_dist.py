"""Reproduce multi_tet test_outside_view0 and introspect the distortion gap.

Replicates the exact RNG sequence of the test (seed 189710234, first
iteration of n_points=10, radius=10, offset_mag=1), renders both paths,
diffs the per-pixel distortion images, and dumps segment-level state for
the worst pixel. Run: CUDA_VISIBLE_DEVICES=2 uv run python analysis/debug_multi_dist.py
"""

import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from tests.multi_tet_test import generate_point_cloud, create_view_matrix  # noqa: E402
from utils.test_util import compute_delaunay, generate_color_palette  # noqa: E402
from utils.topo_utils import calculate_circumcenters_torch  # noqa: E402
from utils.model_util import offset_normalize  # noqa: E402

H = W = 16
TILE = 8
N_SAMPLES = 10000
RADIUS, OFFSET = 10, 1


def main():
    torch.manual_seed(189710234)
    points = generate_point_cloud(10, RADIUS)
    direction = torch.randn(3, device=points.device)
    direction = direction / torch.norm(direction)
    camera_pos = direction * (RADIUS + OFFSET)
    center = torch.mean(points, dim=0)
    viewmat = create_view_matrix(camera_pos, center)

    indices = compute_delaunay(points)
    N = indices.shape[0]
    allc = generate_color_palette(N * 2)[:, :3]
    base_colors = allc[:N].reshape(-1, 3)
    raw_grd = (2 * allc[N:].reshape(-1, 3) - 1) / math.sqrt(3)
    tets = points[indices]
    circumcenters, _ = calculate_circumcenters_torch(tets)
    new_color, new_grd = offset_normalize(base_colors, raw_grd, circumcenters, tets)
    tet_density = torch.ones((len(indices), 1), device="cuda")
    cell_values = torch.cat([tet_density, new_color, new_grd.reshape(-1, 3)], dim=1)

    from utils.compare_quad import test_tetrahedra_rendering as run_tetrahedra_rendering

    res = run_tetrahedra_rendering(points, indices, cell_values, tet_density, viewmat,
                                   height=H, width=W, tile_size=TILE,
                                   n_samples=N_SAMPLES, check_gradients=False)
    print("torch dist loss:", res["torch_dist_loss"])
    print("jax dist loss:", res["jax_dist_loss"])

    # Per-pixel torch distortion: re-render to get distortion_img
    from data.camera import Camera
    from utils.compare_quad import setup_camera
    from delaunay_rasterization.internal.alphablend_tiled_slang_interp import AlphaBlendTiledRender
    from delaunay_rasterization.internal.render_grid import RenderGrid
    from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader

    vm, K, cam_pos2, fovy, fovx, fx, fy = setup_camera(H, W, 90, viewmat)
    V = vm
    camera = Camera(colmap_id=0, R=V[:3, :3].T.cpu().numpy(), T=V[:3, 3].cpu().numpy(),
                    fovx=fovx, fovy=fovy, image=torch.zeros((3, H, W), device="cuda"),
                    uid=0, cx=-1, cy=-1, data_device="cuda")
    rg = RenderGrid(H, W, tile_height=TILE, tile_width=TILE)
    tcam = dict(tile_height=TILE, tile_width=TILE, grid_height=rg.grid_height,
                grid_width=rg.grid_width, min_t=0.01, **camera.to_dict("cuda"))
    sorted_idx, tile_ranges, _, _, _, _ = vertex_and_tile_shader(indices, points, tcam, rg)
    rj = 0.5 * torch.ones((H, W, 2), device="cuda")
    out, xyzd, dist_img, alive, med = AlphaBlendTiledRender.apply(
        sorted_idx, tile_ranges, indices, points, torch.zeros_like(points),
        cell_values, rg, tcam, rj, 0)
    x_, y_, z_, w_, v_ = [dist_img[..., i] for i in range(5)]
    torch_px = ((x_ - y_ + v_) / (z_ ** 2).clip(min=1e-6)).clip(min=0)

    jax_px = torch.as_tensor(np.asarray(res["jax_dist_img"]), device="cuda").float()
    diff = (torch_px - jax_px).abs()
    print("per-pixel |diff|: max", diff.max().item(), "mean", diff.mean().item())
    py, px = np.unravel_index(diff.argmax().item(), diff.shape)
    print(f"worst pixel ({py},{px}): torch={torch_px[py,px].item():.6f} jax={jax_px[py,px].item():.6f}")
    print("torch state [x,y,z,w,v]:", dist_img[py, px].tolist())
    print("torch alpha:", (1 - out[py, px, 3].exp()).item())
    print("torch n pixels nonzero:", (torch_px > 1e-6).sum().item(),
          "jax:", (jax_px > 1e-6).sum().item())

    # Dump jax segment view of that ray
    import jax.numpy as jnp
    from jaxutil import tetra_quad

    vm_np = viewmat.cpu().numpy()
    cam_pos_np = np.linalg.inv(vm_np)[:3, 3]
    Vrot = vm_np[:3, :3].T
    pix = np.array([px + 0.5, py + 0.5])
    d_cam = np.array([(pix[0] - W / 2) / fx.item(), (pix[1] - H / 2) / fy.item(), 1.0])
    d_cam /= np.linalg.norm(d_cam)
    d_world = Vrot @ d_cam

    v_np = points.cpu().numpy()
    pd = vm_np @ np.concatenate([v_np.T, np.ones((1, len(v_np)))], axis=0)
    start = max(pd[2].min(), 0.01)
    vert_dist = np.linalg.norm(v_np - cam_pos_np[None], axis=1)
    end = max(pd[2].max() * 1.2, vert_dist.max() * 1.05)
    t = np.linspace(start, end, N_SAMPLES + 1)
    tm = 0.5 * (t[1:] + t[:-1])
    sig, _ = tetra_quad.query_tetrahedra_kernel(
        jnp.array(tm)[None], jnp.array(cam_pos_np)[None], jnp.array(d_world)[None],
        jnp.array(v_np), jnp.array(indices.cpu().numpy()),
        jnp.array(tet_density.cpu().numpy()), jnp.array(cell_values[:, 1:].cpu().numpy()))
    sig = np.asarray(sig)[0]
    # segment dump
    bnd = np.concatenate([[True], sig[1:] != sig[:-1]])
    seg_starts = np.where(bnd)[0]
    print("jax ray segments (t_a, t_b, sigma):")
    for i, s0 in enumerate(seg_starts):
        s1 = seg_starts[i + 1] if i + 1 < len(seg_starts) else len(sig)
        if sig[s0] > 0:
            print(f"  [{t[s0]:9.4f}, {t[s1]:9.4f}] sigma={sig[s0]:.4f}")
    loss_1d = tetra_quad.warped_dist_loss(
        jnp.array(t)[None], jnp.array(sig)[None], 0.01)
    print("jax recomputed pixel loss:", float(np.asarray(loss_1d)[0]))


if __name__ == "__main__":
    main()
