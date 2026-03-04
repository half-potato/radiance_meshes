"""Render utilities for the rmeshvk (wgpu) renderer.

Provides:
  - create_vk_renderer: builds an RMeshRenderer from the current model state
  - compute_vk_tensors: differentiable param extraction for autograd
  - render_vk: full render call matching the existing render() interface
"""

import torch
import numpy as np

from rmesh_wgpu import RMeshRenderer
from rmesh_wgpu.autograd import RMeshForward
from utils.topo_utils import calculate_circumcenters_torch
from utils.safe_math import safe_div

C0 = 0.28209479177387814


@torch.no_grad()
def create_vk_renderer(model, width, height):
    """Build an RMeshRenderer from current model state.

    Computes circumsphere data and extracts initial numpy arrays.
    """
    vertices = model.vertices  # [V, 3]
    indices = model.indices     # [T, 4]

    # Compute true circumspheres
    tets = vertices[indices]  # [T, 4, 3]
    circumcenter, radius = calculate_circumcenters_torch(tets.double())
    circumcenter = circumcenter.float()
    radius = radius.float()
    # circumdata: [T, 4] = [cx, cy, cz, r^2]
    circumdata = torch.cat([circumcenter, (radius ** 2).unsqueeze(-1)], dim=-1)

    # Get initial features for param shapes
    sh_coeffs, densities, color_grads = _compute_vk_tensors_detached(model, circumcenter, radius)

    renderer = RMeshRenderer(
        vertices.detach().cpu().numpy().ravel().astype(np.float32),
        indices.detach().cpu().numpy().ravel().astype(np.uint32),
        sh_coeffs.cpu().numpy().ravel().astype(np.float32),
        densities.cpu().numpy().ravel().astype(np.float32),
        color_grads.cpu().numpy().ravel().astype(np.float32),
        circumdata.detach().cpu().numpy().ravel().astype(np.float32),
        model.current_sh_deg,
        width,
        height,
    )
    return renderer


def _compute_vk_tensors_detached(model, circumcenter, radius):
    """Non-differentiable tensor extraction (for renderer init)."""
    vertices = model.vertices
    indices = model.indices
    nc = (model.current_sh_deg + 1) ** 2  # total SH coefficients per channel

    all_densities = []
    all_sh = []
    all_grads = []

    for start in range(0, indices.shape[0], model.chunk_size):
        end = min(start + model.chunk_size, indices.shape[0])
        cc, density, rgb, grd, sh, attr = model.compute_batch_features(
            vertices, indices, start, end, circumcenters=circumcenter)

        # Densities: [chunk, 1] -> [chunk]
        all_densities.append(density.squeeze(-1))

        # SH coefficients
        # DC: (rgb - 0.5) / C0 per channel -> [chunk, 3]
        dc = (rgb - 0.5) / C0  # [chunk, 3]

        if nc > 1:
            # Higher-order SH from backbone: reshape to [chunk, nc-1, 3]
            sh_higher = sh.reshape(-1, nc - 1, 3)
            # Combine DC + higher: [chunk, nc, 3]
            dc_expanded = dc.unsqueeze(1)  # [chunk, 1, 3]
            sh_all = torch.cat([dc_expanded, sh_higher], dim=1)
        else:
            sh_all = dc.unsqueeze(1)  # [chunk, 1, 3]

        # Pack per-tet: [R_0..R_{nc-1}, G_0..G_{nc-1}, B_0..B_{nc-1}]
        sh_packed = sh_all.permute(0, 2, 1).contiguous().reshape(-1, nc * 3)
        all_sh.append(sh_packed)

        # Color gradients: normed_grd = grd * rgb_min / radius
        tets_chunk = vertices[indices[start:end]]  # [chunk, 4, 3]
        cc_chunk = circumcenter[start:end]
        r_chunk = radius[start:end]

        rgb_min = rgb.reshape(-1, 3, 1).min(dim=1, keepdim=True).values  # [chunk, 1, 1]
        grd_scaled = grd.reshape(-1, 1, 3) * rgb_min  # [chunk, 1, 3]
        normed_grd = safe_div(grd_scaled, r_chunk.reshape(-1, 1, 1))  # [chunk, 1, 3]
        all_grads.append(normed_grd.reshape(-1, 3))

    densities = torch.cat(all_densities, dim=0)
    sh_coeffs = torch.cat(all_sh, dim=0)
    color_grads = torch.cat(all_grads, dim=0)
    return sh_coeffs, densities, color_grads


def compute_vk_tensors(model):
    """Differentiable tensor extraction for autograd.

    Returns (vertices, sh_coeffs, densities, color_grads) as flat tensors
    that retain grad for backward through the backbone.
    """
    vertices = model.vertices  # [V, 3]
    indices = model.indices    # [T, 4]
    nc = (model.current_sh_deg + 1) ** 2

    # Compute circumspheres (detached geometry for circumcenter computation)
    tets = vertices[indices].detach()  # [T, 4, 3]
    circumcenter, radius = calculate_circumcenters_torch(tets.double())
    circumcenter = circumcenter.float()
    radius = radius.float()

    all_densities = []
    all_sh = []
    all_grads = []

    for start in range(0, indices.shape[0], model.chunk_size):
        end = min(start + model.chunk_size, indices.shape[0])
        cc, density, rgb, grd, sh, attr = model.compute_batch_features(
            vertices, indices, start, end, circumcenters=circumcenter)

        # Densities: [chunk, 1] -> [chunk]
        all_densities.append(density.squeeze(-1))

        # SH DC: (rgb - 0.5) / C0
        dc = (rgb - 0.5) / C0  # [chunk, 3]
        if nc > 1:
            sh_higher = sh.reshape(-1, nc - 1, 3)
            sh_all = torch.cat([dc.unsqueeze(1), sh_higher], dim=1)  # [chunk, nc, 3]
        else:
            sh_all = dc.unsqueeze(1)

        # Pack: [R_0..R_{nc-1}, G_0..G_{nc-1}, B_0..B_{nc-1}]
        sh_packed = sh_all.permute(0, 2, 1).contiguous().reshape(-1, nc * 3)
        all_sh.append(sh_packed)

        # Color gradients
        r_chunk = radius[start:end]
        rgb_min = rgb.reshape(-1, 3, 1).min(dim=1, keepdim=True).values
        grd_scaled = grd.reshape(-1, 1, 3) * rgb_min
        normed_grd = safe_div(grd_scaled, r_chunk.reshape(-1, 1, 1))
        all_grads.append(normed_grd.reshape(-1, 3))

    densities = torch.cat(all_densities, dim=0)      # [T]
    sh_coeffs = torch.cat(all_sh, dim=0).reshape(-1)  # [T * nc * 3]
    color_grads = torch.cat(all_grads, dim=0).reshape(-1)  # [T * 3]

    return vertices.reshape(-1), sh_coeffs, densities, color_grads


def render_vk(camera, model, renderer):
    """Render using rmeshvk, returning a dict matching the Slang render() interface.

    Args:
        camera: Camera object with full_proj_transform, camera_center, etc.
        model: iNGPDW or FrozenTetModel
        renderer: RMeshRenderer instance

    Returns:
        dict with keys: render, alpha, distortion_loss, mask, xyzd, sh_reg, weight_square
    """
    device = model.device
    H = camera.image_height
    W = camera.image_width

    # Differentiable params
    vertices_flat, sh_coeffs, densities, color_grads = compute_vk_tensors(model)

    # Camera matrices
    vp = camera.full_proj_transform.to(device)  # [4, 4] row-major (VP^T)
    inv_vp = torch.inverse(vp)
    cam_pos = camera.camera_center.to(device)   # [3]

    # Forward render via autograd
    image_rgba = RMeshForward.apply(
        renderer, cam_pos, vp, inv_vp,
        vertices_flat, sh_coeffs, densities, color_grads
    )  # [H, W, 4] premultiplied RGBA

    # Use premultiplied RGB directly — avoids gradient explosion from dividing by small alpha.
    # Background pixels (alpha≈0) contribute black, which is correct for black-background GT.
    alpha = image_rgba[..., 3].clamp(min=0, max=1)
    rgb = image_rgba[..., :3].clamp(0, 1)

    # Apply gt_alpha_mask and convert to [3, H, W]
    gt_mask = camera.gt_alpha_mask.to(device)
    render_out = rgb.permute(2, 0, 1) * gt_mask  # [3, H, W]

    return {
        'render': render_out,
        'alpha': alpha,
        'distortion_loss': 0.0,
        'mask': torch.ones(model.indices.shape[0], device=device, dtype=torch.bool),
        'xyzd': torch.zeros(H, W, 4, device=device),
        'sh_reg': 0.0,
        'weight_square': torch.zeros(1, H, W, device=device),
    }
