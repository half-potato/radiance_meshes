"""Reference Python backward implementation to debug gradient math."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import Delaunay
from rmesh_wgpu import RMeshRenderer
from utils.topo_utils import calculate_circumcenters_torch
import torch


def softplus(x):
    if x > 8.0:
        return x
    return 0.1 * np.log(1.0 + np.exp(10.0 * x))

def dsoftplus(x):
    if x > 8.0:
        return 1.0
    e = np.exp(10.0 * x)
    return e / (1.0 + e)

def phi(x):
    if abs(x) < 1e-6:
        return 1.0 - x * 0.5
    return (1.0 - np.exp(-x)) / x

def dphi_dx(x):
    if abs(x) < 1e-6:
        return -0.5 + x / 3.0
    return (np.exp(-x) * (1.0 + x) - 1.0) / (x * x)


FACES = [(0,2,1), (1,2,3), (0,3,2), (3,0,1)]
C0_0 = 0.28209479177387814


def ray_tet_intersect(cam, ray_dir, verts):
    """Returns (t_min, t_max, min_face, max_face) or None if no intersection."""
    t_min = -1e38
    t_max = 1e38
    min_face = 0
    max_face = 0

    for fi, (a, b, c) in enumerate(FACES):
        va, vb, vc = verts[a], verts[b], verts[c]
        n = np.cross(vc - va, vb - va)
        num = np.dot(n, va - cam)
        den = np.dot(n, ray_dir)

        if abs(den) < 1e-20:
            if num > 0:
                return None
            continue

        t = num / den
        if den > 0:
            if t > t_min:
                t_min = t
                min_face = fi
        else:
            if t < t_max:
                t_max = t
                max_face = fi

    if t_min >= t_max:
        return None
    return t_min, t_max, min_face, max_face


def forward_pixel(cam, ray_dir, tet_list):
    """Forward composite for one pixel. Returns (color, alpha, per_tet_data)."""
    color = np.zeros(3)
    log_t = 0.0
    per_tet = []

    for tet_data in tet_list:
        verts, density, colors_tet, grad = tet_data

        result = ray_tet_intersect(cam, ray_dir, verts)
        if result is None:
            per_tet.append(None)
            continue

        t_min, t_max, min_face, max_face = result

        base_offset = np.dot(grad, cam - verts[0])
        base_color = colors_tet + base_offset
        dc_dt = np.dot(grad, ray_dir)

        c_start_raw = base_color + dc_dt * t_min
        c_end_raw = base_color + dc_dt * t_max
        c_start = np.maximum(c_start_raw, 0.0)
        c_end = np.maximum(c_end_raw, 0.0)

        dist = t_max - t_min
        od = max(density * dist, 1e-8)

        alpha_t = np.exp(-od)
        phi_val = phi(od)
        w0 = phi_val - alpha_t
        w1 = 1.0 - phi_val
        c_premul = c_end * w0 + c_start * w1

        T_j = np.exp(log_t)
        color += c_premul * T_j
        log_t -= od

        per_tet.append({
            't_min': t_min, 't_max': t_max, 'min_face': min_face, 'max_face': max_face,
            'c_start_raw': c_start_raw, 'c_end_raw': c_end_raw,
            'c_start': c_start, 'c_end': c_end,
            'od': od, 'dist': dist, 'alpha_t': alpha_t,
            'phi_val': phi_val, 'w0': w0, 'w1': w1,
            'c_premul': c_premul, 'T_j': T_j,
            'base_color': base_color, 'dc_dt': dc_dt,
        })

    T_final = np.exp(log_t)
    alpha = 1.0 - T_final
    return color, alpha, per_tet


def backward_pixel(cam, ray_dir, tet_list, dl_d_color, dl_d_alpha, color_final_fwd,
                   alpha_final_fwd):
    """Reference backward for one pixel. Returns per-tet gradients."""
    # Forward replay
    color, alpha, per_tet = forward_pixel(cam, ray_dir, tet_list)

    # d_color = dl_d_image_rgb (constant per pixel)
    d_color = dl_d_color

    # d_log_t_final
    T_final_fwd = 1.0 - alpha_final_fwd
    d_log_t_final = -dl_d_alpha * T_final_fwd

    # Use color_final from our own replay (not fwd, to avoid f16 issues)
    color_final = color

    # Forward state
    color_accum = np.zeros(3)
    log_t = 0.0

    grads = []
    for i, tet_data in enumerate(tet_list):
        if per_tet[i] is None:
            grads.append(None)
            continue

        verts, density, colors_tet, grad = tet_data
        d = per_tet[i]

        T_j = d['T_j']
        c_premul = d['c_premul']
        od = d['od']

        color_after = color_accum + c_premul * T_j
        log_t_after = log_t - od

        # Closed-form d_log_t
        d_log_t_j = d_log_t_final + np.dot(d_color, color_final - color_after)

        # Gradients
        d_c_premul = d_color * T_j
        d_od_state = -d_log_t_j

        dphi_val = dphi_dx(od)
        dw0_dod = dphi_val + d['alpha_t']
        dw1_dod = -dphi_val

        d_c_end_integral = d_c_premul * d['w0']
        d_c_start_integral = d_c_premul * d['w1']
        d_od_integral = np.dot(d_c_premul, d['c_end'] * dw0_dod + d['c_start'] * dw1_dod)

        d_od = d_od_state + d_od_integral
        d_density = d_od * d['dist']
        d_dist = d_od * density
        d_t_min = -d_dist
        d_t_max = d_dist

        # Through ReLU
        d_c_start_raw = d_c_start_integral * (d['c_start_raw'] > 0).astype(float)
        d_c_end_raw = d_c_end_integral * (d['c_end_raw'] > 0).astype(float)

        d_base_color = d_c_start_raw + d_c_end_raw
        d_dc_dt = (np.sum(d_c_start_raw) * d['t_min'] +
                   np.sum(d_c_end_raw) * d['t_max'])
        d_t_min += np.sum(d_c_start_raw) * d['dc_dt']
        d_t_max += np.sum(d_c_end_raw) * d['dc_dt']

        # base_color = colors_tet + base_offset; base_offset = dot(grad, cam - v0)
        d_base_offset = np.sum(d_base_color)
        d_grad = (cam - verts[0]) * d_base_offset

        # dc_dt = dot(grad, ray_dir)
        d_grad += ray_dir * d_dc_dt

        # Softplus backward
        centroid = np.mean(verts, axis=0)
        sh_dir = centroid - cam
        sh_dir = sh_dir / np.linalg.norm(sh_dir)

        # For SH degree 0: sh_result = sh_coeff * C0_0
        # sp_input = sh_result + 0.5 + offset_val
        offset_val = np.dot(grad, verts[0] - centroid)
        # We don't have sh_coeffs directly, but colors_tet = softplus(sp_input)
        # We need sp_input to compute dsoftplus
        # sp_input[c] = sh_coeffs[tet*3+c] * C0_0 + 0.5 + offset_val
        # colors_tet[c] = softplus(sp_input[c])

        d_sp_input = d_base_color * np.array([dsoftplus(np.log(np.exp(10*c)-1)/10 if c > 0 else 0)
                                               for c in colors_tet])
        # Actually, we can find sp_input from colors_tet by inverting softplus
        # But dsoftplus at the softplus input that gave colors_tet
        # Since softplus(x) ≈ x for x > 0.5, dsoftplus ≈ 1
        # Let me just use the correct formula

        d_offset_scalar = np.sum(d_sp_input)
        d_grad += (verts[0] - centroid) * d_offset_scalar

        # d_sh_coeffs (for SH degree 0, just d_sp_input * C0_0)
        d_sh = d_sp_input * C0_0

        color_accum = color_after
        log_t = log_t_after

        grads.append({
            'd_density': d_density,
            'd_grad': d_grad,
            'd_sh': d_sh,
            'd_log_t_j': d_log_t_j,
            'd_od': d_od,
            'd_c_premul': d_c_premul,
            'd_base_color': d_base_color,
            'd_dc_dt': d_dc_dt,
            'd_base_offset': d_base_offset,
            'd_offset_scalar': d_offset_scalar,
        })

    return grads


def main():
    width, height = 16, 16

    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.15, 0.3, 0.0],
        [0.15, 0.1, 0.3],
        [0.15, 0.1, -0.3],
    ], dtype=np.float32)

    tri = Delaunay(pts)
    indices = tri.simplices.astype(np.int32)
    n_tets = indices.shape[0]
    print(f"Tets: {n_tets}, indices: {indices}")

    sh_degree = 0
    sh_coeffs = np.ones(n_tets * 3, dtype=np.float32) * 0.3
    densities = np.ones(n_tets, dtype=np.float32) * 3.0
    color_grads = np.zeros(n_tets * 3, dtype=np.float32)

    cam = np.array([0.0, 0.0, 1.5], dtype=np.float32)

    # VP matrix
    fov = 1.0
    f = 1.0 / np.tan(fov / 2.0)
    znear, zfar = 0.01, 100.0
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f
    proj[1, 1] = f
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = 1.0
    proj[3, 2] = -(zfar * znear) / (zfar - znear)
    view = np.eye(4, dtype=np.float32)
    view[2, 2] = -1.0
    view[3, 2] = cam[2]
    vp = view @ proj
    inv_vp = np.linalg.inv(vp)

    # Compute colors_tet (post-softplus) for each tet
    tet_list = []
    for t in range(n_tets):
        verts = pts[indices[t]]  # [4, 3]
        density = densities[t]
        grad = color_grads[t*3:(t+1)*3]

        centroid = np.mean(verts, axis=0)
        sh_dir = centroid - cam
        sh_dir = sh_dir / np.linalg.norm(sh_dir)

        sh_result = np.array([sh_coeffs[t*3+c] * C0_0 for c in range(3)])
        offset_val = np.dot(grad, verts[0] - centroid)
        sp_input = sh_result + 0.5 + offset_val
        colors_tet = np.array([softplus(sp_input[c]) for c in range(3)])

        tet_list.append((verts, density, colors_tet, grad))

    # Run wgpu forward for comparison
    vertices_np = pts.ravel()
    indices_np = indices.ravel().astype(np.uint32)
    tets_torch = torch.from_numpy(pts[indices])
    cc, r = calculate_circumcenters_torch(tets_torch.double())
    circumdata_np = torch.cat([cc.float(), (r.float()**2).unsqueeze(-1)], dim=-1).numpy().ravel()

    renderer = RMeshRenderer(
        vertices_np, indices_np, sh_coeffs, densities,
        color_grads, circumdata_np.astype(np.float32), sh_degree, width, height,
    )
    fwd_image = renderer.forward(cam, vp.ravel(), inv_vp.ravel())

    # Find active pixels and compare with reference
    print(f"\n=== Forward comparison ===")
    for py in range(height):
        for px in range(width):
            if fwd_image[py, px, 3] > 0.001:
                # Compute reference ray
                # wgpu Y-flip: the backward's pixel (px, h-1-py) maps to forward pixel (px, py)
                bwd_py = height - 1 - py
                ndc_x = (2.0 * (px + 0.5) / width) - 1.0
                ndc_y = (2.0 * (bwd_py + 0.5) / height) - 1.0

                inv_vp_shader = inv_vp.T  # shader convention
                near_clip = inv_vp_shader @ np.array([ndc_x, ndc_y, 0, 1])
                far_clip = inv_vp_shader @ np.array([ndc_x, ndc_y, 1, 1])
                near_w = near_clip[:3] / near_clip[3]
                far_w = far_clip[:3] / far_clip[3]
                ray_dir = far_w - near_w
                ray_dir = ray_dir / np.linalg.norm(ray_dir)

                ref_color, ref_alpha, per_tet = forward_pixel(cam, ray_dir, tet_list)

                fwd_rgb = fwd_image[py, px, :3]
                fwd_a = fwd_image[py, px, 3]

                print(f"  px={px:2d} py={py:2d}: fwd=({fwd_rgb[0]:.4f},{fwd_rgb[1]:.4f},{fwd_rgb[2]:.4f},{fwd_a:.4f}) "
                      f"ref=({ref_color[0]:.4f},{ref_color[1]:.4f},{ref_color[2]:.4f},{ref_alpha:.4f})")

                # Compute reference backward gradients
                dl_d_rgb = 2 * fwd_rgb
                dl_d_alpha = 2 * fwd_a

                bwd_grads = backward_pixel(cam, ray_dir, tet_list, dl_d_rgb, dl_d_alpha,
                                           ref_color, ref_alpha)

                for t in range(n_tets):
                    if bwd_grads[t] is not None:
                        g = bwd_grads[t]
                        print(f"    tet {t}: d_density={g['d_density']:.6f}, "
                              f"d_sh=[{g['d_sh'][0]:.6f},{g['d_sh'][1]:.6f},{g['d_sh'][2]:.6f}], "
                              f"d_grad=[{g['d_grad'][0]:.6f},{g['d_grad'][1]:.6f},{g['d_grad'][2]:.6f}]")
                        print(f"           d_log_t_j={g['d_log_t_j']:.6f}, d_od={g['d_od']:.6f}, "
                              f"d_dc_dt={g['d_dc_dt']:.6f}, d_base_offset={g['d_base_offset']:.6f}")

    # Accumulate reference gradients over all active pixels
    print(f"\n=== Accumulated reference gradients ===")
    acc_d_density = np.zeros(n_tets)
    acc_d_sh = np.zeros(n_tets * 3)
    acc_d_grad = np.zeros(n_tets * 3)

    for py in range(height):
        for px in range(width):
            if fwd_image[py, px, 3] > 0.001:
                bwd_py = height - 1 - py
                ndc_x = (2.0 * (px + 0.5) / width) - 1.0
                ndc_y = (2.0 * (bwd_py + 0.5) / height) - 1.0

                inv_vp_shader = inv_vp.T
                near_clip = inv_vp_shader @ np.array([ndc_x, ndc_y, 0, 1])
                far_clip = inv_vp_shader @ np.array([ndc_x, ndc_y, 1, 1])
                near_w = near_clip[:3] / near_clip[3]
                far_w = far_clip[:3] / far_clip[3]
                ray_dir = far_w - near_w
                ray_dir = ray_dir / np.linalg.norm(ray_dir)

                dl_d_rgb = 2 * fwd_image[py, px, :3]
                dl_d_alpha = 2 * fwd_image[py, px, 3]

                ref_color, ref_alpha, per_tet = forward_pixel(cam, ray_dir, tet_list)
                bwd_grads = backward_pixel(cam, ray_dir, tet_list, dl_d_rgb, dl_d_alpha,
                                           ref_color, ref_alpha)

                for t in range(n_tets):
                    if bwd_grads[t] is not None:
                        acc_d_density[t] += bwd_grads[t]['d_density']
                        acc_d_sh[t*3:(t+1)*3] += bwd_grads[t]['d_sh']
                        acc_d_grad[t*3:(t+1)*3] += bwd_grads[t]['d_grad']

    print(f"  d_density: {acc_d_density}")
    print(f"  d_sh_coeffs: {acc_d_sh}")
    print(f"  d_color_grads: {acc_d_grad}")

    # Run wgpu backward for comparison
    dl_d_image = 2 * fwd_image  # Same as PyTorch autograd
    grads = renderer.backward(dl_d_image)

    print(f"\n=== wgpu backward gradients ===")
    print(f"  d_density: {grads['d_densities']}")
    print(f"  d_sh_coeffs: {grads['d_sh_coeffs']}")
    print(f"  d_color_grads: {grads['d_color_grads']}")

    # Numerical gradients for comparison
    print(f"\n=== Numerical gradients (central diff, eps=1e-3) ===")
    for param_name, param_arr in [("densities", densities), ("sh_coeffs", sh_coeffs), ("color_grads", color_grads)]:
        num_grad = np.zeros_like(param_arr)
        eps = 1e-3
        for i in range(len(param_arr)):
            p_plus = param_arr.copy()
            p_plus[i] += eps

            # Update renderer and run forward
            if param_name == "densities":
                renderer.update_params(vertices_np, sh_coeffs, p_plus, color_grads)
            elif param_name == "sh_coeffs":
                renderer.update_params(vertices_np, p_plus, densities, color_grads)
            else:
                renderer.update_params(vertices_np, sh_coeffs, densities, p_plus)
            img_plus = renderer.forward(cam, vp.ravel(), inv_vp.ravel())
            loss_plus = (img_plus[..., :3] ** 2).sum() + (img_plus[..., 3] ** 2).sum()

            p_minus = param_arr.copy()
            p_minus[i] -= eps
            if param_name == "densities":
                renderer.update_params(vertices_np, sh_coeffs, p_minus, color_grads)
            elif param_name == "sh_coeffs":
                renderer.update_params(vertices_np, p_minus, densities, color_grads)
            else:
                renderer.update_params(vertices_np, sh_coeffs, densities, p_minus)
            img_minus = renderer.forward(cam, vp.ravel(), inv_vp.ravel())
            loss_minus = (img_minus[..., :3] ** 2).sum() + (img_minus[..., 3] ** 2).sum()

            num_grad[i] = (loss_plus - loss_minus) / (2 * eps)

        # Restore original params
        renderer.update_params(vertices_np, sh_coeffs, densities, color_grads)

        print(f"  {param_name}: {num_grad}")


if __name__ == "__main__":
    main()
