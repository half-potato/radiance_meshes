"""Minimal gradient test: single tet, tiny image."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.spatial import Delaunay
from rmesh_wgpu import RMeshRenderer
from rmesh_wgpu.autograd import RMeshForward
from utils.topo_utils import calculate_circumcenters_torch


def make_single_tet_scene(width=16, height=16):
    """Create a scene with exactly 1 tet, positioned to cover several pixels."""
    # 5 points that make a nice tet arrangement
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.15, 0.3, 0.0],
        [0.15, 0.1, 0.3],
        [0.15, 0.1, -0.3],  # 5th point to ensure Delaunay works
    ], dtype=np.float32)

    tri = Delaunay(pts)
    indices = tri.simplices.astype(np.int32)
    n_tets = indices.shape[0]
    print(f"Points: {pts.shape[0]}, Tets: {n_tets}")
    print(f"Tet indices: {indices}")

    vertices_np = pts.ravel().astype(np.float32)
    indices_np = indices.ravel().astype(np.uint32)
    sh_degree = 0
    nc = 1

    sh_coeffs_np = np.ones(n_tets * nc * 3, dtype=np.float32) * 0.3
    densities_np = np.ones(n_tets, dtype=np.float32) * 3.0
    color_grads_np = np.zeros(n_tets * 3, dtype=np.float32)

    tets = torch.from_numpy(pts[indices])
    cc, r = calculate_circumcenters_torch(tets.double())
    circumdata_np = torch.cat([cc.float(), (r.float() ** 2).unsqueeze(-1)], dim=-1).numpy().ravel().astype(np.float32)

    cam_pos = np.array([0.0, 0.0, 1.5], dtype=np.float32)
    fov = 1.0
    aspect = width / height
    znear, zfar = 0.01, 100.0
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

    device = torch.device("cuda")
    return (renderer,
            torch.from_numpy(cam_pos).to(device),
            torch.from_numpy(vp).to(device),
            torch.from_numpy(inv_vp).to(device),
            torch.from_numpy(vertices_np.copy()).to(device),
            torch.from_numpy(sh_coeffs_np.copy()).to(device),
            torch.from_numpy(densities_np.copy()).to(device),
            torch.from_numpy(color_grads_np.copy()).to(device),
            n_tets)


def compute_loss(renderer, cam_pos, vp, inv_vp, vertices, sh_coeffs, densities, color_grads):
    image = RMeshForward.apply(renderer, cam_pos, vp, inv_vp, vertices, sh_coeffs, densities, color_grads)
    return (image[..., :3] ** 2).sum() + (image[..., 3] ** 2).sum()


def finite_diff_grad(renderer, cam_pos, vp, inv_vp, vertices, sh_coeffs, densities, color_grads,
                     param_name, eps=1e-3):
    params = {"vertices": vertices, "sh_coeffs": sh_coeffs, "densities": densities, "color_grads": color_grads}
    param = params[param_name]
    grad = torch.zeros_like(param)
    for i in range(param.numel()):
        p_plus = param.clone()
        p_plus.view(-1)[i] += eps
        kwargs_plus = {k: (p_plus if k == param_name else v) for k, v in params.items()}
        loss_plus = compute_loss(renderer, cam_pos, vp, inv_vp, **kwargs_plus).item()

        p_minus = param.clone()
        p_minus.view(-1)[i] -= eps
        kwargs_minus = {k: (p_minus if k == param_name else v) for k, v in params.items()}
        loss_minus = compute_loss(renderer, cam_pos, vp, inv_vp, **kwargs_minus).item()

        grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def main():
    renderer, cam_pos, vp, inv_vp, vertices, sh_coeffs, densities, color_grads, n_tets = \
        make_single_tet_scene()

    # Show forward image
    image = RMeshForward.apply(renderer, cam_pos, vp, inv_vp, vertices, sh_coeffs, densities, color_grads)
    print(f"\nForward image: shape={image.shape}, range=[{image.min():.4f}, {image.max():.4f}]")
    active = (image[..., 3] > 0.001).sum().item()
    print(f"Active pixels: {active}")

    # Run backward and check debug image
    cam_np = cam_pos.detach().cpu().numpy().ravel().astype(np.float32)
    vp_np = vp.detach().cpu().numpy().ravel().astype(np.float32)
    inv_vp_np = inv_vp.detach().cpu().numpy().ravel().astype(np.float32)
    fwd = renderer.forward(cam_np, vp_np, inv_vp_np)
    dl = np.ones_like(fwd)
    grads = renderer.backward(dl)
    dbg = renderer.read_debug_image()
    print(f"\nForward: active={(fwd[...,3]>0.001).sum()}, range=[{fwd.min():.4f}, {fwd.max():.4f}]")
    print(f"Debug:   active={(dbg[...,3]>0.001).sum()}, range=[{dbg.min():.4f}, {dbg.max():.4f}]")
    diff = np.abs(fwd - dbg)
    print(f"Diff:    max={diff.max():.6f}")

    # Show per-pixel comparison
    for py in range(fwd.shape[0]):
        for px in range(fwd.shape[1]):
            if fwd[py, px, 3] > 0.001 or dbg[py, px, 3] > 0.001:
                print(f"  px={px:2d} py={py:2d}: fwd_a={fwd[py,px,3]:.4f} dbg_a={dbg[py,px,3]:.4f}")

    # Show gradients returned
    for k, v in grads.items():
        nz = np.sum(np.abs(v) > 1e-6)
        print(f"  grad {k}: shape={v.shape}, nonzero={nz}, range=[{v.min():.6f}, {v.max():.6f}]")

    # Test with multiple epsilon values to confirm f16 noise hypothesis
    for eps_val in [1e-3, 1e-2, 1e-1]:
        print(f"\n{'='*60}")
        print(f"  eps = {eps_val}")
        print(f"{'='*60}")
        for param_name in ["densities", "sh_coeffs", "color_grads"]:
            params = {"vertices": vertices.clone().requires_grad_(True),
                      "sh_coeffs": sh_coeffs.clone().requires_grad_(True),
                      "densities": densities.clone().requires_grad_(True),
                      "color_grads": color_grads.clone().requires_grad_(True)}
            loss = compute_loss(renderer, cam_pos, vp, inv_vp, **params)
            loss.backward()
            analytical = params[param_name].grad

            numerical = finite_diff_grad(renderer, cam_pos, vp, inv_vp,
                                         vertices, sh_coeffs, densities, color_grads,
                                         param_name, eps=eps_val)

            print(f"\n=== {param_name} (eps={eps_val}) ===")
            for i in range(analytical.numel()):
                a = analytical.view(-1)[i].item()
                n = numerical.view(-1)[i].item()
                if abs(a) > 1e-5 or abs(n) > 1e-5:
                    ratio = a / n if abs(n) > 1e-6 else float('inf')
                    print(f"  [{i:2d}] analytical={a:12.8f}  numerical={n:12.8f}  ratio={ratio:8.4f}")

            mask = (analytical.abs() > 1e-4) | (numerical.abs() > 1e-4)
            if mask.sum() > 0:
                a = analytical[mask]
                n = numerical[mask]
                cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), n.unsqueeze(0)).item()
                rel_err = ((a - n).abs() / (n.abs() + 1e-4)).mean().item()
                print(f"  cos_sim={cos_sim:.6f}, mean_rel_err={rel_err:.6f}")


if __name__ == "__main__":
    main()
