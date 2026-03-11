"""Compare backward's forward replay compositing against the actual forward render.

This tests that the backward shader's ray-tet intersection, iteration order,
and compositing exactly match the forward hardware rasterizer.

Usage:
    uv run python tests/test_backward_replay.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.spatial import Delaunay
from rmesh_wgpu import RMeshRenderer
from utils.topo_utils import calculate_circumcenters_torch


def make_scene(n_verts=8, sh_degree=0, width=32, height=32, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    pts = np.random.randn(n_verts, 3).astype(np.float32) * 0.3
    tri = Delaunay(pts)
    indices = tri.simplices.astype(np.int32)

    vertices_np = pts.ravel().astype(np.float32)
    indices_np = indices.ravel().astype(np.uint32)
    n_tets = indices.shape[0]
    nc = (sh_degree + 1) ** 2

    sh_coeffs_np = np.random.randn(n_tets * nc * 3).astype(np.float32) * 0.5
    densities_np = (np.random.rand(n_tets).astype(np.float32) * 5 + 1.0)
    color_grads_np = np.random.randn(n_tets * 3).astype(np.float32) * 0.1

    tets = torch.from_numpy(pts[indices])
    cc, r = calculate_circumcenters_torch(tets.double())
    circumdata_np = torch.cat([cc.float(), (r.float() ** 2).unsqueeze(-1)], dim=-1).numpy().ravel().astype(np.float32)

    cam_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)

    fov = 1.0
    aspect = width / height
    znear, zfar = 0.1, 100.0
    f = 1.0 / np.tan(fov / 2.0)

    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = 1.0
    proj[3, 2] = -(zfar * znear) / (zfar - znear)

    view = np.eye(4, dtype=np.float32)
    view[2, 2] = -1.0
    view[3, 2] = cam_pos[2]

    vp = (view @ proj).astype(np.float32)
    inv_vp = np.linalg.inv(vp).astype(np.float32)

    renderer = RMeshRenderer(
        vertices_np, indices_np, sh_coeffs_np, densities_np,
        color_grads_np, circumdata_np, sh_degree, width, height,
    )

    return renderer, cam_pos, vp, inv_vp


def main():
    renderer, cam_pos, vp, inv_vp = make_scene()

    # Run forward
    fwd_image = renderer.forward(cam_pos, vp.ravel(), inv_vp.ravel())
    print(f"Forward image shape: {fwd_image.shape}, range: [{fwd_image.min():.4f}, {fwd_image.max():.4f}]")

    # Run backward with dummy loss gradient (all ones)
    h, w = fwd_image.shape[:2]
    dl_d_image = np.ones((h, w, 4), dtype=np.float32)
    grads = renderer.backward(dl_d_image)

    # Read debug image (backward's forward replay)
    debug_image = renderer.read_debug_image()
    print(f"Debug image shape: {debug_image.shape}, range: [{debug_image.min():.4f}, {debug_image.max():.4f}]")

    # Compare
    diff = np.abs(fwd_image - debug_image)
    print(f"\n=== Forward vs Backward Replay Comparison ===")
    print(f"Max absolute diff: {diff.max():.6f}")
    print(f"Mean absolute diff: {diff.mean():.6f}")

    # Per-channel comparison
    for c, name in enumerate(["R", "G", "B", "A"]):
        ch_diff = diff[:, :, c]
        fwd_ch = fwd_image[:, :, c]
        dbg_ch = debug_image[:, :, c]
        nonzero = (np.abs(fwd_ch) > 1e-6) | (np.abs(dbg_ch) > 1e-6)
        n_active = nonzero.sum()
        print(f"  {name}: max_diff={ch_diff.max():.6f}, mean_diff={ch_diff.mean():.6f}, "
              f"active_pixels={n_active}/{h*w}")
        if n_active > 0:
            rel_err = ch_diff[nonzero] / (np.abs(fwd_ch[nonzero]) + 1e-6)
            print(f"       mean_rel_err={rel_err.mean():.4f}, max_rel_err={rel_err.max():.4f}")

    # Show a few pixels with the biggest differences
    flat_diff = diff.sum(axis=2).ravel()
    worst_idxs = np.argsort(flat_diff)[-5:][::-1]
    if flat_diff[worst_idxs[0]] > 1e-4:
        print(f"\nWorst pixels:")
        for idx in worst_idxs:
            py, px = divmod(idx, w)
            print(f"  pixel ({px},{py}): fwd={fwd_image[py,px]} dbg={debug_image[py,px]} diff={diff[py,px]}")

    # Check if Y-flipped debug image matches
    debug_flipped = debug_image[::-1, :, :]
    diff_flipped = np.abs(fwd_image - debug_flipped)
    print(f"\n=== Y-Flipped Comparison ===")
    print(f"Max absolute diff (Y-flipped): {diff_flipped.max():.6f}")
    print(f"Mean absolute diff (Y-flipped): {diff_flipped.mean():.6f}")
    for c, name in enumerate(["R", "G", "B", "A"]):
        ch_diff = diff_flipped[:, :, c]
        print(f"  {name}: max_diff={ch_diff.max():.6f}")

    # Verify NDC→world mapping vs expected
    # The VP maps world_y > 0 to ndc_y > 0 (bottom in wgpu: pixel_y > H/2)
    # The backward at pixel (15, 19) has ndc_y = (2*19.5/32) - 1 = 0.21875
    # The backward at pixel (15, 12) has ndc_y = (2*12.5/32) - 1 = -0.21875
    print(f"\n=== NDC Analysis ===")
    for py_test in [12, 15, 19]:
        ndc_y = (2 * (py_test + 0.5) / h) - 1.0
        # World-space ray via inv_vp
        ndc_pt_near = np.array([0, ndc_y, 0, 1], dtype=np.float32)
        ndc_pt_far = np.array([0, ndc_y, 1, 1], dtype=np.float32)
        # shader_inv_vp = numpy_inv_vp^T
        shader_inv_vp = inv_vp.T
        near_clip = shader_inv_vp @ ndc_pt_near
        far_clip = shader_inv_vp @ ndc_pt_far
        near_w = near_clip[:3] / near_clip[3]
        far_w = far_clip[:3] / far_clip[3]
        ray_d = far_w - near_w
        ray_d = ray_d / np.linalg.norm(ray_d)
        print(f"  pixel y={py_test}: ndc_y={ndc_y:.4f}, ray_dir_y={ray_d[1]:.4f}, fwd_val={fwd_image[py_test, 15, :3].round(3)}, dbg_val={debug_image[py_test, 15, :3].round(3)}")

    # Also check: does the WGSL shader's VP work the same way?
    # shader_vp = numpy_vp^T
    shader_vp = vp.T
    test_pt = np.array([0, 0.1, 0, 1], dtype=np.float32)
    clip = shader_vp @ test_pt
    ndc = clip[:3] / clip[3]
    pix_y = (ndc[1] + 1) * 0.5 * h
    print(f"\n  World (0,0.1,0) -> clip.y={clip[1]:.4f}, ndc.y={ndc[1]:.4f}, pixel_y={pix_y:.1f}")

    # Overall verdict
    if diff.max() < 0.01:
        print(f"\nPASS: Forward replay matches forward render (max diff < 0.01)")
    else:
        print(f"\nFAIL: Forward replay does NOT match forward render (max diff = {diff.max():.4f})")
        # Check if alpha channel is the issue
        rgb_diff = diff[:, :, :3].max()
        alpha_diff = diff[:, :, 3].max()
        print(f"  RGB max diff: {rgb_diff:.6f}")
        print(f"  Alpha max diff: {alpha_diff:.6f}")


if __name__ == "__main__":
    main()
