"""Test that Slang interp and rmeshvk tiled forward produce identical output.

Creates a synthetic scene with known geometry, renders through both pipelines,
and compares the RGB output pixel-by-pixel.

Requires: CUDA (for Slang), rmesh_wgpu (built via maturin).
Run: python tests/test_slang_vs_vk.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import math

# ---------------------------------------------------------------------------
# 1. Build a synthetic tet scene
# ---------------------------------------------------------------------------

def make_regular_tet(center=(0, 0, 3), scale=1.0):
    """Regular tetrahedron centered at `center`."""
    # Vertices of a regular tet inscribed in a unit sphere
    a = scale
    verts = torch.tensor([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ], dtype=torch.float32) * (a / math.sqrt(3))
    verts += torch.tensor(center, dtype=torch.float32)
    return verts


def build_synthetic_scene(n_tets=4):
    """Build a multi-tet scene with varied colors, densities, color_grads."""
    torch.manual_seed(42)
    all_verts = []
    all_indices = []
    all_densities = []
    all_base_colors = []
    all_color_grads = []

    for i in range(n_tets):
        offset = (i - n_tets // 2) * 1.5
        v = make_regular_tet(center=(offset, 0, 4 + i * 0.5), scale=0.8)
        base_idx = len(all_verts)
        all_verts.extend(v.tolist())
        all_indices.append([base_idx, base_idx + 1, base_idx + 2, base_idx + 3])
        all_densities.append(2.0 + i * 0.5)
        all_base_colors.append([0.3 + i * 0.1, 0.5, 0.7 - i * 0.1])
        all_color_grads.append([0.1 * (i - 1), 0.05, -0.05 * i])

    vertices = torch.tensor(all_verts, dtype=torch.float32)
    indices = torch.tensor(all_indices, dtype=torch.int32)
    densities = torch.tensor(all_densities, dtype=torch.float32)
    base_colors = torch.tensor(all_base_colors, dtype=torch.float32)
    color_grads = torch.tensor(all_color_grads, dtype=torch.float32)

    return vertices, indices, densities, base_colors, color_grads


# ---------------------------------------------------------------------------
# 2. Camera setup
# ---------------------------------------------------------------------------

def make_camera_matrices(cam_pos, look_at_pt, up, fovx, fovy, W, H):
    """Build world_view_transform, K, projection, and VP matrices.

    Camera conventions:
      - R: 3x3 rotation transforming world → camera
      - world_view: stored as [R^T | t; 0 1] for Slang/reference compatibility
        (Slang receives world_view.T, extracts 3x3 block, transposes to get R_c2w)
      - VP for rmeshvk: must account for mat4_from_flat using Mat4::from_cols()
        which interprets row-major flat data as columns (effectively transposing).
        So we pass (P @ V_std).T so WGSL receives the correct VP = P @ V_std.
    """
    # Build R (world → camera rotation) using COLMAP-like convention
    f = (look_at_pt - cam_pos)
    f = f / f.norm()
    r = torch.linalg.cross(up, f)
    r = r / r.norm()
    u = torch.linalg.cross(f, r)

    # R: world → camera.  Camera axes: x=right, y=up, z=forward
    R = torch.stack([r, u, f], dim=0)  # [3, 3]
    t = -R @ cam_pos  # translation in camera coords

    # world_view for Slang/reference: [R^T | t; 0 1]
    # Reference extracts R_c2w = world_view[:3,:3] (= R^T) for ray construction
    # Slang receives world_view.T and construct_view_matrix transposes 3x3 to get R_c2w
    world_view = torch.eye(4)
    world_view[:3, :3] = R.T
    world_view[:3, 3] = t

    # K (intrinsics)
    fx = W / (2 * math.tan(fovx / 2))
    fy = H / (2 * math.tan(fovy / 2))
    cx, cy = W / 2, H / 2
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=torch.float32)

    # Projection matrix (column-vector convention, y-flipped for pinhole K compat)
    # P[1,1] is negated so that camera y-up → NDC y-down → screen y-down,
    # matching pinhole convention where image row 0 = top = negative camera y.
    tanHalfFovX = math.tan(fovx / 2)
    tanHalfFovY = math.tan(fovy / 2)
    znear, zfar = 0.01, 100.0
    P = torch.tensor([
        [1 / tanHalfFovX, 0, 0, 0],
        [0, -1 / tanHalfFovY, 0, 0],
        [0, 0, zfar / (zfar - znear), -(zfar * znear) / (zfar - znear)],
        [0, 0, 1, 0],
    ], dtype=torch.float32)

    # Standard view matrix: V_std @ [pos;1] = [R@pos + t; 1] (camera coords)
    V_std = torch.eye(4)
    V_std[:3, :3] = R
    V_std[:3, 3] = t

    # VP for rmeshvk: mat4_from_flat (Rust) uses Mat4::from_cols() which
    # interprets the row-major flat array as columns, effectively transposing.
    # WGSL does vp * vec4(pos, 1.0) (column-major mat-vec).
    # So we pass VP_std.T: after from_cols transpose, WGSL gets VP_std = P @ V_std.
    VP_std = P @ V_std
    vp = VP_std.T
    inv_vp = torch.inverse(vp)

    return world_view, K, vp, inv_vp, cam_pos


# ---------------------------------------------------------------------------
# 3. Pure-Python reference renderer (Slang interp math)
# ---------------------------------------------------------------------------

def phi(x):
    """phi(x) = (1 - exp(-x)) / x, stable near 0."""
    if abs(x) < 1e-6:
        return 1.0 - x * 0.5
    return (1.0 - math.exp(-x)) / x


def render_pixel_reference(
    cam_pos, ray_dir,
    verts,           # [4, 3]
    density,         # scalar
    base_color,      # [3]
    color_grad,      # [3]
):
    """Single-pixel volume integral matching Slang interp_version.slang.

    Ray: origin = cam_pos, direction = ray_dir (normalized).
    Color model: c(x) = max(base_color + dot(color_grad, x - verts[0]), 0)
    Volume integral: phi/w0/w1 linear color weighting.

    Uses the same slab intersection as Slang's ray_tetrahedron_intersect_fused:
    compute inward-facing normals by flipping if they point away from the
    opposite vertex, then classify enter/exit by normal·ray sign.
    """
    # 4 faces as (v_a, v_b, v_c, v_opposite)
    FACES = [
        (0, 1, 2, 3),
        (0, 1, 3, 2),
        (0, 2, 3, 1),
        (1, 2, 3, 0),
    ]

    t_min = -1e30
    t_max = 1e30
    eps = 1e-10

    for a, b, c, opp in FACES:
        va, vb, vc = verts[a], verts[b], verts[c]
        # Normal — flip to point inward (toward opposite vertex)
        n = np.cross(vb - va, vc - va)
        if np.dot(n, verts[opp] - va) > 0:
            n = -n

        dist = -np.dot(n, va - cam_pos)
        denom = np.dot(n, ray_dir)

        if abs(denom) < eps:
            if dist > 0:
                return np.array([0, 0, 0, 0], dtype=np.float32)
            continue

        tplane = -dist / denom
        if denom < 0:
            # Ray entering (going against inward normal from outside)
            t_min = max(t_min, tplane)
        else:
            # Ray exiting (going with inward normal from inside)
            t_max = min(t_max, tplane)

    if t_min > t_max:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    if t_max <= 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    t_min = max(t_min, 0.0)

    # Color at entry/exit
    v0 = verts[0]
    v_enter = t_min * ray_dir + cam_pos - v0
    v_exit = t_max * ray_dir + cam_pos - v0
    c_enter = np.maximum(np.dot(color_grad, v_enter) + base_color, 0)
    c_exit = np.maximum(np.dot(color_grad, v_exit) + base_color, 0)

    dt = t_max - t_min
    od = max(density * dt, 1e-8)
    alpha_t = math.exp(-od)
    phi_val = phi(od)
    w0 = phi_val - alpha_t    # weight for c_exit  (c0 in Slang)
    w1 = 1.0 - phi_val        # weight for c_enter (c1 in Slang)
    c_premul = c_exit * w0 + c_enter * w1

    alpha = 1 - alpha_t
    return np.array([c_premul[0], c_premul[1], c_premul[2], alpha], dtype=np.float32)


def render_scene_reference(cam_pos_np, W, H, fovx, fovy, world_view_np, K_np,
                           vertices_np, indices_np, densities_np,
                           base_colors_np, color_grads_np):
    """CPU reference renderer matching Slang interp math.

    Uses K + world_view_transform to construct rays (same as Slang camera.slang).
    Front-to-back compositing.
    """
    n_tets = len(indices_np)
    fx, fy = K_np[0, 0], K_np[1, 1]
    cx, cy = K_np[0, 2], K_np[1, 2]

    # Extract camera→world rotation from world_view = [R^T | t; 0 1]
    # world_view[:3,:3] = R^T = R_c2w (camera → world)
    V_rot = world_view_np[:3, :3]  # R_c2w

    image = np.zeros((H, W, 4), dtype=np.float32)

    for py in range(H):
        for px in range(W):
            # Ray construction matching Slang camera.slang get_ray()
            p_x = (px + 0.5 - cx) / fx
            p_y = (py + 0.5 - cy) / fy
            dir_cam = np.array([p_x, p_y, 1.0])
            dir_cam = dir_cam / np.linalg.norm(dir_cam)
            ray_dir = V_rot @ dir_cam
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            # Front-to-back compositing
            accum_rgb = np.zeros(3, dtype=np.float64)
            log_T = 0.0

            # Simple front-to-back: process all tets (no sorting for simplicity,
            # since we want to match the formula, not the sort order)
            for ti in range(n_tets):
                idx = indices_np[ti]
                tet_verts = vertices_np[idx]
                rgba = render_pixel_reference(
                    cam_pos_np, ray_dir,
                    tet_verts, densities_np[ti],
                    base_colors_np[ti], color_grads_np[ti],
                )
                if rgba[3] > 0:
                    T = math.exp(log_T)
                    accum_rgb += rgba[:3] * T
                    od = -math.log(max(1.0 - rgba[3], 1e-20))
                    log_T -= od

            final_T = math.exp(log_T)
            image[py, px, :3] = accum_rgb
            image[py, px, 3] = 1.0 - final_T

    return image


# ---------------------------------------------------------------------------
# 4. rmeshvk renderer
# ---------------------------------------------------------------------------

def render_scene_vk(cam_pos_t, vp_t, inv_vp_t, W, H,
                    vertices_t, indices_t, densities_t,
                    base_colors_t, color_grads_t):
    """Render via rmeshvk (wgpu tiled forward)."""
    from rmesh_wgpu import RMeshRenderer
    from rmesh_wgpu.autograd import RMeshForward

    # Compute circumdata
    n_tets = indices_t.shape[0]
    tets = vertices_t[indices_t]  # [T, 4, 3]
    centroids = tets.mean(dim=1)
    # Rough circumsphere: use centroid + max vertex distance
    radii = torch.norm(tets - centroids.unsqueeze(1), dim=-1).max(dim=1).values
    circumdata = torch.cat([centroids, (radii ** 2).unsqueeze(-1)], dim=-1)

    renderer = RMeshRenderer(
        vertices_t.numpy().ravel().astype(np.float32),
        indices_t.numpy().ravel().astype(np.uint32),
        base_colors_t.numpy().ravel().astype(np.float32),
        densities_t.numpy().ravel().astype(np.float32),
        color_grads_t.numpy().ravel().astype(np.float32),
        circumdata.numpy().ravel().astype(np.float32),
        W, H,
    )

    # Debug: also try the non-tiled forward path
    cam_np = cam_pos_t.detach().numpy().ravel().astype(np.float32)
    vp_np = vp_t.detach().numpy().ravel().astype(np.float32)
    inv_vp_np = inv_vp_t.detach().numpy().ravel().astype(np.float32)
    try:
        nontiled_img = renderer.forward(cam_np, vp_np, inv_vp_np)
        nt_sum = nontiled_img[:, :, 3].sum()
        print(f"  VK non-tiled: alpha_sum={nt_sum:.2f}, "
              f"RGB range=[{nontiled_img[:,:,:3].min():.4f}, {nontiled_img[:,:,:3].max():.4f}]")
    except Exception as e:
        print(f"  VK non-tiled failed: {e}")

    image_rgba = RMeshForward.apply(
        renderer, cam_pos_t, vp_t, inv_vp_t,
        vertices_t.reshape(-1), base_colors_t.reshape(-1),
        densities_t, color_grads_t.reshape(-1),
    )
    return image_rgba.detach().numpy()


# ---------------------------------------------------------------------------
# 5. Slang renderer
# ---------------------------------------------------------------------------

def render_scene_slang(cam_pos_t, world_view_t, K_t, W, H, fovx, fovy, min_t,
                       vertices_t, indices_t, cell_values_t):
    """Render via Slang interp pipeline."""
    from delaunay_rasterization.internal.render_grid import RenderGrid
    from delaunay_rasterization.internal.alphablend_tiled_slang_interp import AlphaBlendTiledRender
    from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader

    device = vertices_t.device
    tile_size = 4
    render_grid = RenderGrid(H, W, tile_height=tile_size, tile_width=tile_size)

    tcam = dict(
        tile_height=tile_size,
        tile_width=tile_size,
        grid_height=render_grid.grid_height,
        grid_width=render_grid.grid_width,
        min_t=min_t,
        world_view_transform=world_view_t,
        cam_pos=cam_pos_t,
        K=K_t,
        image_height=H,
        image_width=W,
        fovy=fovy,
        fovx=fovx,
        distortion_params=torch.zeros(4, device=device),
        camera_type=0,  # PERSPECTIVE
    )

    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _ = \
        vertex_and_tile_shader(indices_t, vertices_t, tcam, render_grid)

    ray_jitter = 0.5 * torch.ones((H, W, 2), device=device)
    aux_dim = 0

    image_rgb, xyzd_img, distortion_img, tet_alive = AlphaBlendTiledRender.apply(
        sorted_tetra_idx, tile_ranges, indices_t, vertices_t,
        cell_values_t, render_grid, tcam, ray_jitter, aux_dim,
    )

    # image_rgb: [H, W, 4+aux] where channel 3 = log_transmittance
    rgb = image_rgb[:, :, :3]
    log_T = image_rgb[:, :, 3]
    alpha = 1.0 - torch.exp(log_T)

    out = torch.zeros(H, W, 4, device=device)
    out[:, :, :3] = rgb
    out[:, :, 3] = alpha
    return out.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 6. Backward gradient comparison
# ---------------------------------------------------------------------------

def compute_vk_gradients(cam_pos, vp, inv_vp, W, H,
                         vertices, indices, densities, base_colors, color_grads,
                         loss_weights, return_debug_image=False):
    """Render via rmeshvk and compute parameter gradients."""
    from rmesh_wgpu import RMeshRenderer
    from rmesh_wgpu.autograd import RMeshForward

    n_tets = indices.shape[0]
    tets = vertices[indices]
    centroids = tets.float().mean(dim=1)
    radii = torch.norm(tets.float() - centroids.unsqueeze(1), dim=-1).max(dim=1).values
    circumdata = torch.cat([centroids, (radii ** 2).unsqueeze(-1)], dim=-1)

    renderer = RMeshRenderer(
        vertices.numpy().ravel().astype(np.float32),
        indices.numpy().ravel().astype(np.uint32),
        base_colors.numpy().ravel().astype(np.float32),
        densities.numpy().ravel().astype(np.float32),
        color_grads.numpy().ravel().astype(np.float32),
        circumdata.numpy().ravel().astype(np.float32),
        W, H,
    )

    verts = vertices.clone().reshape(-1).requires_grad_(True)
    bc = base_colors.clone().reshape(-1).requires_grad_(True)
    dens = densities.clone().requires_grad_(True)
    cg = color_grads.clone().reshape(-1).requires_grad_(True)

    image = RMeshForward.apply(
        renderer, cam_pos, vp, inv_vp,
        verts, bc, dens, cg,
    )

    loss = (image[:, :, :3] * loss_weights).sum()
    loss.backward()

    result = {
        'd_vertices': verts.grad.reshape(-1, 3),
        'd_base_colors': bc.grad.reshape(-1, 3),
        'd_densities': dens.grad,
        'd_color_grads': cg.grad.reshape(-1, 3),
        'loss': loss.item(),
    }
    if return_debug_image:
        result['debug_image'] = renderer.read_debug_image()
    return result


def compute_slang_gradients(cam_pos, world_view, K, W, H, fovx, fovy, min_t,
                            vertices, indices, densities, base_colors, color_grads,
                            loss_weights):
    """Render via Slang interp and compute parameter gradients."""
    from delaunay_rasterization.internal.render_grid import RenderGrid
    from delaunay_rasterization.internal.alphablend_tiled_slang_interp import AlphaBlendTiledRender
    from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader

    device = torch.device("cuda")

    cell_values_data = torch.cat([
        densities.unsqueeze(1), base_colors, color_grads,
    ], dim=1)

    tile_size = 4
    render_grid = RenderGrid(H, W, tile_height=tile_size, tile_width=tile_size)

    tcam = dict(
        tile_height=tile_size,
        tile_width=tile_size,
        grid_height=render_grid.grid_height,
        grid_width=render_grid.grid_width,
        min_t=min_t,
        world_view_transform=world_view.T.to(device),
        cam_pos=cam_pos.to(device),
        K=K.to(device),
        image_height=H,
        image_width=W,
        fovy=fovy,
        fovx=fovx,
        distortion_params=torch.zeros(4, device=device),
        camera_type=0,
    )

    # Tile computation (non-differentiable)
    indices_cuda = indices.to(device)
    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _ = \
        vertex_and_tile_shader(indices_cuda, vertices.to(device).detach(), tcam, render_grid)

    # Differentiable inputs
    verts = vertices.clone().to(device).requires_grad_(True)
    cv = cell_values_data.clone().to(device).requires_grad_(True)

    ray_jitter = 0.5 * torch.ones((H, W, 2), device=device)
    aux_dim = 0

    image_rgb, xyzd_img, distortion_img, tet_alive = AlphaBlendTiledRender.apply(
        sorted_tetra_idx, tile_ranges, indices_cuda, verts,
        cv, render_grid, tcam, ray_jitter, aux_dim,
    )

    loss = (image_rgb[:, :, :3] * loss_weights.to(device)).sum()
    loss.backward()

    d_cv = cv.grad.cpu()
    return {
        'd_vertices': verts.grad.cpu(),
        'd_base_colors': d_cv[:, 1:4],
        'd_densities': d_cv[:, 0],
        'd_color_grads': d_cv[:, 4:7],
        'loss': loss.item(),
    }


def dphi_dx(x):
    """Derivative of phi(x) = (1 - exp(-x)) / x."""
    if abs(x) < 1e-6:
        return -0.5 + x / 3.0
    ex = math.exp(-x)
    return (ex * (x + 1.0) - 1.0) / (x * x)


def cpu_backward_vertex_gradients(cam_pos_np, W, H, vp_np, inv_vp_np,
                                   vertices_np, indices_np, densities_np,
                                   base_colors_np, color_grads_np, loss_weights_np,
                                   return_per_pixel=False):
    """CPU implementation of the VK backward vertex gradient formulas.

    Implements the exact same computation as backward_tiled_compute.wgsl,
    but in Python for debugging.
    """
    FACES = [
        (0, 2, 1, 3),
        (1, 2, 3, 0),
        (0, 3, 2, 1),
        (3, 0, 1, 2),
    ]

    n_verts = vertices_np.shape[0]
    n_tets = indices_np.shape[0]
    d_vertices = np.zeros_like(vertices_np)
    if return_per_pixel:
        # Per-pixel debug: [H, W, 4] = (d_vert_local[0].xyz, d_t_min)
        per_pixel_debug = np.zeros((H, W, 4), dtype=np.float64)

    # First do a full forward pass to get the rendered image
    # (needed for backward replay state)
    # Simple front-to-back compositing per pixel
    image = np.zeros((H, W, 4), dtype=np.float64)

    for py in range(H):
        for px in range(W):
            ndc_x = (2.0 * (px + 0.5) / W) - 1.0
            ndc_y = 1.0 - (2.0 * (py + 0.5) / H)
            near_clip = np.array([ndc_x, ndc_y, 0.0, 1.0])
            far_clip = np.array([ndc_x, ndc_y, 1.0, 1.0])
            near_h = inv_vp_np @ near_clip
            far_h = inv_vp_np @ far_clip
            near_world = near_h[:3] / near_h[3]
            far_world = far_h[:3] / far_h[3]
            ray_dir = far_world - near_world
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            cam = cam_pos_np

            accum_rgb = np.zeros(3, dtype=np.float64)
            log_T = 0.0

            for ti in range(n_tets):
                idx = indices_np[ti]
                verts = vertices_np[idx]
                density_raw = float(densities_np[ti])
                colors_tet = base_colors_np[ti].astype(np.float64)
                grad_vec = color_grads_np[ti].astype(np.float64)

                # Ray-tet intersection (matching VK)
                t_min_val = -1e30
                t_max_val = 1e30
                valid = True
                min_face = 0
                max_face = 0

                for fi, (a, b, c, opp) in enumerate(FACES):
                    va, vb, vc = verts[a], verts[b], verts[c]
                    n = np.cross(vc - va, vb - va)
                    v_opp = verts[opp]
                    if np.dot(n, v_opp - va) < 0:
                        n = -n
                    num = np.dot(n, va - cam)
                    den = np.dot(n, ray_dir)
                    if abs(den) < 1e-20:
                        if num > 0:
                            valid = False
                        continue
                    t = num / den
                    if den > 0:
                        if t > t_min_val:
                            t_min_val = t
                            min_face = fi
                    else:
                        if t < t_max_val:
                            t_max_val = t
                            max_face = fi

                if not valid or t_min_val >= t_max_val:
                    continue

                base_offset = np.dot(grad_vec, cam - verts[0])
                base_color = colors_tet + base_offset
                dc_dt = np.dot(grad_vec, ray_dir)
                c_start_raw = base_color + dc_dt * t_min_val
                c_end_raw = base_color + dc_dt * t_max_val
                c_start = np.maximum(c_start_raw, 0.0)
                c_end = np.maximum(c_end_raw, 0.0)

                dist = t_max_val - t_min_val
                od = max(density_raw * dist, 1e-8)
                alpha_t = math.exp(-od)
                phi_val = phi(od)
                w0 = phi_val - alpha_t
                w1 = 1.0 - phi_val
                c_premul = c_end * w0 + c_start * w1

                T_j = math.exp(log_T)
                color_after = accum_rgb + c_premul * T_j
                log_T -= od
                accum_rgb = color_after

            final_T = math.exp(log_T)
            image[py, px, :3] = accum_rgb
            image[py, px, 3] = 1.0 - final_T

    # Now backward pass
    dl_d_image = np.zeros((H, W, 4), dtype=np.float64)
    dl_d_image[:, :, 0] = loss_weights_np[0]
    dl_d_image[:, :, 1] = loss_weights_np[1]
    dl_d_image[:, :, 2] = loss_weights_np[2]
    # dl_d_image[:, :, 3] = 0  (no alpha loss)

    for py in range(H):
        for px in range(W):
            ndc_x = (2.0 * (px + 0.5) / W) - 1.0
            ndc_y = 1.0 - (2.0 * (py + 0.5) / H)
            near_clip = np.array([ndc_x, ndc_y, 0.0, 1.0])
            far_clip = np.array([ndc_x, ndc_y, 1.0, 1.0])
            near_h = inv_vp_np @ near_clip
            far_h = inv_vp_np @ far_clip
            near_world = near_h[:3] / near_h[3]
            far_world = far_h[:3] / far_h[3]
            ray_dir = far_world - near_world
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            cam = cam_pos_np

            d_color = dl_d_image[py, px, :3]
            alpha_final = image[py, px, 3]
            d_log_t_final = -dl_d_image[py, px, 3] * (1.0 - alpha_final)
            color_final = image[py, px, :3]

            # Forward replay (front-to-back, matching VK backward's replay)
            log_t_before = 0.0
            color_accum_before = np.zeros(3, dtype=np.float64)

            for ti in range(n_tets):
                idx = indices_np[ti]
                verts = vertices_np[idx]
                density_raw = float(densities_np[ti])
                colors_tet = base_colors_np[ti].astype(np.float64)
                grad_vec = color_grads_np[ti].astype(np.float64)

                # Ray-tet intersection
                t_min_val = -1e30
                t_max_val = 1e30
                valid = True
                min_face = 0
                max_face = 0

                for fi, (a, b, c, opp) in enumerate(FACES):
                    va, vb, vc = verts[a], verts[b], verts[c]
                    n = np.cross(vc - va, vb - va)
                    v_opp = verts[opp]
                    if np.dot(n, v_opp - va) < 0:
                        n = -n
                    num = np.dot(n, va - cam)
                    den = np.dot(n, ray_dir)
                    if abs(den) < 1e-20:
                        if num > 0:
                            valid = False
                        continue
                    t = num / den
                    if den > 0:
                        if t > t_min_val:
                            t_min_val = t
                            min_face = fi
                    else:
                        if t < t_max_val:
                            t_max_val = t
                            max_face = fi

                if not valid or t_min_val >= t_max_val:
                    continue

                base_offset = np.dot(grad_vec, cam - verts[0])
                base_color = colors_tet + base_offset
                dc_dt = np.dot(grad_vec, ray_dir)
                c_start_raw = base_color + dc_dt * t_min_val
                c_end_raw = base_color + dc_dt * t_max_val
                c_start = np.maximum(c_start_raw, 0.0)
                c_end = np.maximum(c_end_raw, 0.0)

                dist = t_max_val - t_min_val
                od = max(density_raw * dist, 1e-8)
                alpha_t = math.exp(-od)
                phi_val = phi(od)
                w0 = phi_val - alpha_t
                w1 = 1.0 - phi_val
                c_premul = c_end * w0 + c_start * w1

                T_j = math.exp(log_t_before)
                color_after = color_accum_before + c_premul * T_j

                # === Backward ===
                d_log_t_j = d_log_t_final + np.dot(d_color, color_final - color_after)
                d_c_premul = d_color * T_j
                d_od_state = -d_log_t_j

                dphi_val = dphi_dx(od)
                dw0_dod = dphi_val + math.exp(-od)
                dw1_dod = -dphi_val

                d_c_end_integral = d_c_premul * w0
                d_c_start_integral = d_c_premul * w1
                d_od_integral = np.dot(d_c_premul, c_end * dw0_dod + c_start * dw1_dod)

                d_od = d_od_state + d_od_integral
                d_dist = d_od * density_raw
                d_t_min = -d_dist
                d_t_max = d_dist

                # ReLU gradient
                d_c_start_raw = np.where(c_start_raw > 0, d_c_start_integral, 0.0)
                d_c_end_raw = np.where(c_end_raw > 0, d_c_end_integral, 0.0)

                d_t_min += d_c_start_raw.sum() * dc_dt
                d_t_max += d_c_end_raw.sum() * dc_dt

                # Intersection gradients (VK formulas, using RAW normal)
                for face_idx, d_t in [(min_face, d_t_min), (max_face, d_t_max)]:
                    a, b, c, opp = FACES[face_idx]
                    va = verts[a]; vb = verts[b]; vc = verts[c]
                    n = np.cross(vc - va, vb - va)
                    den = np.dot(n, ray_dir)
                    t_val = t_min_val if face_idx == min_face else t_max_val
                    hit = cam + ray_dir * t_val
                    dt_dva = (np.cross(va - hit, vb - vc) + n) / den
                    dt_dvb = np.cross(va - hit, vc - va) / den
                    dt_dvc = np.cross(va - hit, va - vb) / den
                    d_vertices[idx[a]] += dt_dva * d_t
                    d_vertices[idx[b]] += dt_dvb * d_t
                    d_vertices[idx[c]] += dt_dvc * d_t

                # d_v0_from_base
                d_base_color_vec = d_c_start_raw + d_c_end_raw
                d_base_offset_scalar = d_base_color_vec.sum()
                d_v0_from_base = -grad_vec * d_base_offset_scalar
                d_vertices[idx[0]] += d_v0_from_base

                if return_per_pixel:
                    # Output max_face dt_dvb and d_t_max for debug
                    a_mx, b_mx, c_mx, opp_mx = FACES[max_face]
                    va_mx = verts[a_mx]; vb_mx = verts[b_mx]; vc_mx = verts[c_mx]
                    n_mx = np.cross(vc_mx - va_mx, vb_mx - va_mx)
                    den_mx = np.dot(n_mx, ray_dir)
                    hit_mx = cam + ray_dir * t_max_val
                    dt_dvb_mx = np.cross(va_mx - hit_mx, vc_mx - va_mx) / den_mx
                    per_pixel_debug[py, px, 0] = dt_dvb_mx[0]
                    per_pixel_debug[py, px, 1] = dt_dvb_mx[1]
                    per_pixel_debug[py, px, 2] = dt_dvb_mx[2]
                    per_pixel_debug[py, px, 3] = d_t_max

                # Update forward replay state
                color_accum_before = color_after
                log_t_before -= od

    if return_per_pixel:
        return d_vertices, per_pixel_debug
    return d_vertices


def compute_fd_vertex_gradients(cam_pos, world_view, K, vp, inv_vp, W, H, fovx, fovy, min_t,
                                vertices, indices, densities, base_colors, color_grads,
                                loss_weights, eps=1e-3, use_slang=True):
    """Finite-difference vertex gradient check using the CPU reference renderer.

    For each vertex coordinate, perturb by +/- eps and compute
    (loss_plus - loss_minus) / (2 * eps).
    """
    n_verts = vertices.shape[0]
    fd_grad = torch.zeros_like(vertices)

    for vi in range(n_verts):
        for ci in range(3):
            verts_plus = vertices.clone()
            verts_plus[vi, ci] += eps
            verts_minus = vertices.clone()
            verts_minus[vi, ci] -= eps

            if use_slang:
                # Use Slang for more accurate FD (matches Slang's own ray construction)
                device = torch.device("cuda")
                cv_plus = torch.cat([densities.unsqueeze(1), base_colors, color_grads], dim=1)
                cv_minus = cv_plus.clone()

                img_plus = render_scene_slang(
                    cam_pos.to(device), world_view.T.to(device), K.to(device),
                    W, H, fovx, fovy, min_t,
                    verts_plus.to(device), indices.to(device), cv_plus.to(device),
                )
                img_minus = render_scene_slang(
                    cam_pos.to(device), world_view.T.to(device), K.to(device),
                    W, H, fovx, fovy, min_t,
                    verts_minus.to(device), indices.to(device), cv_minus.to(device),
                )
            else:
                # Use CPU reference
                img_plus = render_scene_reference(
                    cam_pos.numpy(), W, H, fovx, fovy,
                    world_view.numpy(), K.numpy(),
                    verts_plus.numpy(), indices.numpy(), densities.numpy(),
                    base_colors.numpy(), color_grads.numpy(),
                )
                img_minus = render_scene_reference(
                    cam_pos.numpy(), W, H, fovx, fovy,
                    world_view.numpy(), K.numpy(),
                    verts_minus.numpy(), indices.numpy(), densities.numpy(),
                    base_colors.numpy(), color_grads.numpy(),
                )

            lw = loss_weights.numpy()
            loss_plus = (img_plus[:, :, :3] * lw).sum()
            loss_minus = (img_minus[:, :, :3] * lw).sum()
            fd_grad[vi, ci] = float((loss_plus - loss_minus) / (2 * eps))

    return fd_grad


def test_backward_gradients(W, H, fovx, fovy, min_t, cam_pos, look_at_pt, up,
                            vertices, indices, densities, base_colors, color_grads,
                            world_view, K, vp, inv_vp):
    """Compare backward gradients between Slang interp and rmeshvk."""
    loss_weights = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float32)

    print("\n" + "=" * 60)
    print("BACKWARD GRADIENT COMPARISON")
    print("=" * 60)
    print(f"Loss = (image[:,:,:3] * {loss_weights.tolist()}).sum()")

    has_vk = False
    vk_grads = None
    try:
        print("\nComputing VK gradients...")
        vk_grads = compute_vk_gradients(
            cam_pos, vp, inv_vp, W, H,
            vertices, indices, densities, base_colors, color_grads,
            loss_weights,
        )
        has_vk = True
        print(f"  VK loss = {vk_grads['loss']:.6f}")
        for name in ['d_vertices', 'd_base_colors', 'd_densities', 'd_color_grads']:
            g = vk_grads[name]
            print(f"  VK {name}: shape={list(g.shape)}, "
                  f"abs_sum={g.abs().sum():.6f}, max={g.abs().max():.6f}")
    except Exception as e:
        import traceback
        print(f"  VK backward failed: {e}")
        traceback.print_exc()

    has_slang = False
    slang_grads = None
    try:
        print("\nComputing Slang gradients...")
        slang_grads = compute_slang_gradients(
            cam_pos, world_view, K, W, H, fovx, fovy, min_t,
            vertices, indices, densities, base_colors, color_grads,
            loss_weights,
        )
        has_slang = True
        print(f"  Slang loss = {slang_grads['loss']:.6f}")
        for name in ['d_vertices', 'd_base_colors', 'd_densities', 'd_color_grads']:
            g = slang_grads[name]
            print(f"  Slang {name}: shape={list(g.shape)}, "
                  f"abs_sum={g.abs().sum():.6f}, max={g.abs().max():.6f}")
    except Exception as e:
        import traceback
        print(f"  Slang backward failed: {e}")
        traceback.print_exc()

    # Compare gradients
    results = {}
    if has_vk and has_slang:
        print(f"\n  Loss: VK={vk_grads['loss']:.6f}, Slang={slang_grads['loss']:.6f}, "
              f"diff={abs(vk_grads['loss'] - slang_grads['loss']):.6f}")

        for name in ['d_vertices', 'd_base_colors', 'd_densities', 'd_color_grads']:
            g_vk = vk_grads[name].detach().numpy()
            g_slang = slang_grads[name].detach().numpy()
            diff = np.abs(g_vk - g_slang)
            # Relative error (for values with significant magnitude)
            scale = np.maximum(np.abs(g_slang), np.abs(g_vk)).clip(min=1e-7)
            rel_diff = diff / scale

            print(f"\n  {name}:")
            print(f"    Abs diff  — mean={diff.mean():.6f}, max={diff.max():.6f}")
            print(f"    Rel diff  — mean={rel_diff.mean():.4f}, max={rel_diff.max():.4f}")
            print(f"    VK range  = [{g_vk.min():.6f}, {g_vk.max():.6f}]")
            print(f"    Slang rng = [{g_slang.min():.6f}, {g_slang.max():.6f}]")

            # Show worst element
            flat_idx = np.argmax(diff.ravel())
            idx = np.unravel_index(flat_idx, diff.shape)
            print(f"    Worst element at {idx}: VK={g_vk[idx]:.8f}, Slang={g_slang[idx]:.8f}")

            # Tolerance: allow up to 1% relative error or 0.01 absolute
            tol_abs = 0.01
            tol_rel = 0.01
            ok = np.all((diff < tol_abs) | (rel_diff < tol_rel))
            n_bad = np.sum((diff >= tol_abs) & (rel_diff >= tol_rel))
            if ok:
                print(f"    PASS")
            else:
                print(f"    FAIL: {n_bad}/{diff.size} elements exceed tolerance")
            results[f'grad_{name}'] = ok

    # Isolate dt/dv vs d_v0_from_base: test with zero color grads
    if has_vk and has_slang:
        print("\n--- Isolation test: zero color_grads (pure intersection gradient) ---")
        zero_cg = torch.zeros_like(color_grads)
        try:
            vk_g0 = compute_vk_gradients(
                cam_pos, vp, inv_vp, W, H,
                vertices, indices, densities, base_colors, zero_cg, loss_weights)
            slang_g0 = compute_slang_gradients(
                cam_pos, world_view, K, W, H, fovx, fovy, min_t,
                vertices, indices, densities, base_colors, zero_cg, loss_weights)
            dv_vk = vk_g0['d_vertices'].detach().numpy()
            dv_sl = slang_g0['d_vertices'].detach().numpy()
            diff0 = np.abs(dv_vk - dv_sl)
            print(f"  VK d_v abs_sum={np.abs(dv_vk).sum():.4f}, Slang abs_sum={np.abs(dv_sl).sum():.4f}")
            print(f"  Abs diff — mean={diff0.mean():.6f}, max={diff0.max():.6f}")
            if diff0.max() < 0.01:
                print(f"  PASS: intersection gradient matches with zero color_grads")
                print(f"  → Bug is in the color gradient path (d_v0_from_base / dc_dt)")
            else:
                print(f"  FAIL: intersection gradient differs even with zero color_grads")
                print(f"  → Bug is in the intersection gradient (dt/dv)")
        except Exception as e:
            print(f"  Zero color_grads test failed: {e}")

    # Finite-difference vertex gradient check (ground truth)
    if has_vk or has_slang:
        print("\n--- Finite-difference vertex gradient check ---")
        try:
            fd_grad = compute_fd_vertex_gradients(
                cam_pos, world_view, K, vp, inv_vp, W, H, fovx, fovy, min_t,
                vertices, indices, densities, base_colors, color_grads,
                loss_weights, eps=1e-3, use_slang=True,
            )
            fd_np = fd_grad.numpy()
            print(f"  FD d_vertices: abs_sum={np.abs(fd_np).sum():.6f}, "
                  f"range=[{fd_np.min():.6f}, {fd_np.max():.6f}]")

            for label, grads, has in [("VK", vk_grads, has_vk), ("Slang", slang_grads, has_slang)]:
                if not has:
                    continue
                g = grads['d_vertices'].detach().numpy()
                diff = np.abs(g - fd_np)
                scale = np.maximum(np.abs(fd_np), np.abs(g)).clip(min=1e-7)
                rel_diff = diff / scale
                n_bad = np.sum((diff >= 0.1) & (rel_diff >= 0.05))
                print(f"\n  {label} vs FD:")
                print(f"    Abs diff — mean={diff.mean():.6f}, max={diff.max():.6f}")
                print(f"    Rel diff — mean={rel_diff.mean():.4f}, max={rel_diff.max():.4f}")
                # Per-tet breakdown
                n_tets = indices.shape[0]
                for ti in range(n_tets):
                    v_start = indices[ti, 0].item()
                    v_end = v_start + 4
                    tet_diff = diff[v_start:v_end]
                    tet_g = g[v_start:v_end]
                    tet_fd = fd_np[v_start:v_end]
                    if tet_diff.max() > 0.01:
                        print(f"    Tet {ti} (verts {v_start}-{v_end-1}): max_diff={tet_diff.max():.4f}")
                        for vi in range(4):
                            if np.abs(tet_diff[vi]).max() > 0.01:
                                print(f"      v{vi} (idx={v_start+vi}): "
                                      f"{label}={tet_g[vi]}, FD={tet_fd[vi]}, diff={tet_diff[vi]}")

                if n_bad == 0:
                    print(f"    {label} vs FD: PASS")
                else:
                    print(f"    {label} vs FD: FAIL ({n_bad}/{diff.size} elements)")
        except Exception as e:
            import traceback
            print(f"  FD check failed: {e}")
            traceback.print_exc()

    return results


# ---------------------------------------------------------------------------
# 7. Main test
# ---------------------------------------------------------------------------

def test_slang_vs_vk():
    """Compare Slang interp vs rmeshvk for a synthetic scene."""
    W, H = 64, 64
    fovx = math.radians(60)
    fovy = math.radians(60)
    min_t = 0.1

    cam_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    look_at = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    world_view, K, vp, inv_vp, _ = make_camera_matrices(
        cam_pos, look_at, up, fovx, fovy, W, H)

    vertices, indices, densities, base_colors, color_grads = build_synthetic_scene(n_tets=4)

    # Cell values for Slang: [density, base_color(3), color_grad(3)]
    cell_values = torch.cat([
        densities.unsqueeze(1),
        base_colors,
        color_grads,
    ], dim=1)  # [T, 7]

    print(f"Scene: {vertices.shape[0]} vertices, {indices.shape[0]} tets")
    print(f"Camera: {W}x{H}, fov={math.degrees(fovx):.0f}°")
    print(f"VP matrix (passed to Rust, row-major):\n{vp}")

    # Verify: simulate what WGSL sees after mat4_from_flat (from_cols transposes)
    wgsl_vp = vp.T
    print(f"\nWGSL VP (after from_cols transpose):\n{wgsl_vp}")
    for ti in range(min(indices.shape[0], 2)):
        idx = indices[ti]
        for vi in range(4):
            v = vertices[idx[vi]]
            v_h = torch.tensor([v[0], v[1], v[2], 1.0])
            clip = wgsl_vp @ v_h
            ndc = clip[:3] / clip[3] if clip[3] != 0 else clip[:3]
            print(f"  Tet {ti} V{vi} pos={v.tolist()} -> clip.w={clip[3]:.4f}, ndc=({ndc[0]:.4f}, {ndc[1]:.4f}, {ndc[2]:.4f})")
    print()

    # --- Reference (CPU, Slang math) ---
    print("Rendering: CPU reference (Slang interp math)...")
    ref_image = render_scene_reference(
        cam_pos.numpy(), W, H, fovx, fovy,
        world_view.numpy(), K.numpy(),
        vertices.numpy(), indices.numpy(), densities.numpy(),
        base_colors.numpy(), color_grads.numpy(),
    )
    ref_alpha_sum = ref_image[:, :, 3].sum()
    print(f"  Reference: alpha_sum={ref_alpha_sum:.2f}, "
          f"RGB range=[{ref_image[:,:,:3].min():.4f}, {ref_image[:,:,:3].max():.4f}]")

    # --- rmeshvk ---
    has_vk = False
    vk_image = None
    try:
        print("Rendering: rmeshvk (wgpu tiled forward)...")
        vk_image = render_scene_vk(
            cam_pos, vp, inv_vp, W, H,
            vertices, indices, densities, base_colors, color_grads,
        )
        has_vk = True
        vk_alpha_sum = vk_image[:, :, 3].sum()
        print(f"  VK: alpha_sum={vk_alpha_sum:.2f}, "
              f"RGB range=[{vk_image[:,:,:3].min():.4f}, {vk_image[:,:,:3].max():.4f}]")
    except Exception as e:
        print(f"  rmeshvk not available: {e}")

    # --- Slang ---
    has_slang = False
    slang_image = None
    try:
        device = torch.device("cuda")
        print("Rendering: Slang interp...")
        slang_image = render_scene_slang(
            cam_pos.to(device), world_view.T.to(device), K.to(device),
            W, H, fovx, fovy, min_t,
            vertices.to(device), indices.to(device), cell_values.to(device),
        )
        has_slang = True
        slang_alpha_sum = slang_image[:, :, 3].sum()
        print(f"  Slang: alpha_sum={slang_alpha_sum:.2f}, "
              f"RGB range=[{slang_image[:,:,:3].min():.4f}, {slang_image[:,:,:3].max():.4f}]")
    except Exception as e:
        print(f"  Slang not available: {e}")

    # --- Comparisons ---
    print("\n" + "=" * 60)
    print("COMPARISONS")
    print("=" * 60)

    def compare(name_a, img_a, name_b, img_b):
        diff = np.abs(img_a - img_b)
        rgb_diff = diff[:, :, :3]
        alpha_diff = diff[:, :, 3]
        print(f"\n{name_a} vs {name_b}:")
        print(f"  RGB  — L1={rgb_diff.mean():.6f}, max={rgb_diff.max():.6f}")
        print(f"  Alpha — L1={alpha_diff.mean():.6f}, max={alpha_diff.max():.6f}")

        # Find worst pixel
        flat_idx = np.argmax(rgb_diff.sum(axis=2))
        py, px = divmod(flat_idx, W)
        print(f"  Worst pixel: ({px}, {py})")
        print(f"    {name_a}: RGB={img_a[py, px, :3]}, A={img_a[py, px, 3]:.6f}")
        print(f"    {name_b}: RGB={img_b[py, px, :3]}, A={img_b[py, px, 3]:.6f}")

        tol = 0.01
        if rgb_diff.max() > tol:
            n_bad = (rgb_diff > tol).sum()
            total = rgb_diff.size
            print(f"  FAIL: {n_bad}/{total} values exceed tolerance {tol}")
            return False
        else:
            print(f"  PASS: all within tolerance {tol}")
            return True

    results = {}

    if has_vk:
        results['ref_vs_vk'] = compare("Reference", ref_image, "VK", vk_image)

    if has_slang:
        results['ref_vs_slang'] = compare("Reference", ref_image, "Slang", slang_image)

    if has_vk and has_slang:
        results['slang_vs_vk'] = compare("Slang", slang_image, "VK", vk_image)

    # --- Backward gradient comparison ---
    backward_results = test_backward_gradients(
        W, H, fovx, fovy, min_t, cam_pos, look_at, up,
        vertices, indices, densities, base_colors, color_grads,
        world_view, K, vp, inv_vp,
    )
    results.update(backward_results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if not results:
        print("  No GPU renderers available — only CPU reference was computed.")
        print("  Build rmesh_wgpu (maturin develop) and/or install CUDA+slangtorch.")

    if not all_pass:
        sys.exit(1)


def test_single_tet_gradient():
    """Quick diagnostic: test backward vertex gradient for a single tet."""
    print("\n" + "=" * 60)
    print("SINGLE-TET GRADIENT DIAGNOSTIC")
    print("=" * 60)

    W, H = 64, 64
    fovx = math.radians(60)
    fovy = math.radians(60)
    min_t = 0.1

    cam_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    look_at_pt = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    world_view, K, vp, inv_vp, _ = make_camera_matrices(
        cam_pos, look_at_pt, up, fovx, fovy, W, H)

    vertices, indices, densities, base_colors, color_grads = build_synthetic_scene(n_tets=1)
    loss_weights = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float32)

    print(f"Scene: 1 tet, verts={vertices.tolist()}")

    try:
        vk_g = compute_vk_gradients(
            cam_pos, vp, inv_vp, W, H,
            vertices, indices, densities, base_colors, color_grads, loss_weights,
            return_debug_image=True)
        slang_g = compute_slang_gradients(
            cam_pos, world_view, K, W, H, fovx, fovy, min_t,
            vertices, indices, densities, base_colors, color_grads, loss_weights)
        fd_g = compute_fd_vertex_gradients(
            cam_pos, world_view, K, vp, inv_vp, W, H, fovx, fovy, min_t,
            vertices, indices, densities, base_colors, color_grads,
            loss_weights, eps=1e-3, use_slang=True)

        # Also compute CPU backward (implements VK formulas on CPU)
        wgsl_vp = vp.T.numpy()  # what WGSL sees
        wgsl_inv_vp = torch.inverse(vp.T).numpy()
        dv_cpu, cpu_pixel_debug = cpu_backward_vertex_gradients(
            cam_pos.numpy(), W, H, wgsl_vp, wgsl_inv_vp,
            vertices.numpy(), indices.numpy(), densities.numpy(),
            base_colors.numpy(), color_grads.numpy(), loss_weights.numpy(),
            return_per_pixel=True)

        dv_vk = vk_g['d_vertices'].detach().numpy()
        dv_sl = slang_g['d_vertices'].detach().numpy()
        dv_fd = fd_g.numpy()

        print(f"\n  VK   d_vertices:\n{dv_vk}")
        print(f"  CPU  d_vertices:\n{dv_cpu}")
        print(f"  Slang d_vertices:\n{dv_sl}")
        print(f"  FD   d_vertices:\n{dv_fd}")
        print(f"\n  VK-CPU diff (should be ~0 if WGSL matches formulas):\n{dv_vk - dv_cpu}")
        print(f"  CPU-FD diff (formula vs ground truth):\n{dv_cpu - dv_fd}")
        print(f"  VK-FD diff:\n{dv_vk - dv_fd}")
        print(f"  Slang-FD diff:\n{dv_sl - dv_fd}")
        print(f"\n  VK/FD ratio (where FD > 0.1):")
        mask = np.abs(dv_fd) > 0.1
        if mask.any():
            ratio = dv_vk[mask] / dv_fd[mask]
            print(f"    ratios: {ratio}")
            print(f"    mean ratio: {ratio.mean():.4f}")

        vk_fd_diff = np.abs(dv_vk - dv_fd).max()
        sl_fd_diff = np.abs(dv_sl - dv_fd).max()
        cpu_fd_diff = np.abs(dv_cpu - dv_fd).max()
        print(f"\n  VK vs FD: max_diff={vk_fd_diff:.6f}")
        print(f"  CPU vs FD: max_diff={cpu_fd_diff:.6f}")
        print(f"  Slang vs FD: max_diff={sl_fd_diff:.6f}")
        if vk_fd_diff < 0.5:
            print("  PASS: single tet VK matches FD")
        else:
            print("  FAIL: single tet VK gradient is WRONG")

        # Per-pixel debug removed (zero-init bug found and fixed)

    except Exception as e:
        import traceback
        print(f"  Single-tet test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_single_tet_gradient()
    test_slang_vs_vk()
