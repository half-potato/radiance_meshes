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
from utils.graphics_utils import getProjectionMatrix


def make_vk_vp(camera, model, device):
    """Construct VP matrix with near/far planes encompassing the full scene.

    The default camera znear=0.5, zfar=10.0 clips most geometry when
    rendered via hardware rasterization. We recompute with scene-appropriate bounds.
    """
    # Compute scene extent from camera
    cam_pos = camera.camera_center.to(device)  # [3]
    verts = model.vertices  # [V, 3]
    dists = torch.norm(verts - cam_pos.unsqueeze(0), dim=1)
    zfar = dists.max().item() * 1.1  # 10% margin
    znear = max(0.01, dists.min().item() * 0.5)

    # Build new projection matrix with proper bounds
    proj = getProjectionMatrix(znear=znear, zfar=zfar,
                               fovX=camera.fovx, fovY=camera.fovy)
    # getProjectionMatrix has P[1,1] = +1/tanHalfFovY (OpenGL convention).
    # WGSL shader maps ndc→pixel as py = (1 - ndc_y)*H/2, so we need
    # P[1,1] = -1/tanHalfFovY to correctly map COLMAP's y-down camera to screen.
    proj[1, 1] = -proj[1, 1]
    proj = proj.transpose(0, 1).to(device)
    wvt = camera.world_view_transform.to(device)
    vp = wvt.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    inv_vp = torch.inverse(vp)
    return vp, inv_vp, cam_pos


@torch.no_grad()
def create_vk_renderer(model, camera, width, height):
    """Build an RMeshRenderer from current model state.

    Uses model.get_cell_values (same as Slang) for initial buffer data.
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

    # Get initial features via same path as Slang
    _, cell_values = model.get_cell_values(camera, all_circumcenters=circumcenter)
    # cell_values layout: [T, feat_dim] = [density, base_color(3), normed_grd(3), ...]
    densities = cell_values[:, 0]
    base_colors = cell_values[:, 1:4]
    color_grads = cell_values[:, 4:7]

    renderer = RMeshRenderer(
        vertices.detach().cpu().numpy().ravel().astype(np.float32),
        indices.detach().cpu().numpy().ravel().astype(np.uint32),
        base_colors.cpu().numpy().ravel().astype(np.float32),
        densities.cpu().numpy().ravel().astype(np.float32),
        color_grads.cpu().numpy().ravel().astype(np.float32),
        circumdata.detach().cpu().numpy().ravel().astype(np.float32),
        width,
        height,
    )
    return renderer


def compute_vk_tensors(model, camera):
    """Differentiable tensor extraction using the same cell_values path as Slang.

    Returns (vertices, base_colors, densities, color_grads) as flat tensors
    that retain grad for backward through the backbone.
    """
    _, cell_values = model.get_cell_values(camera)
    # cell_values layout: [T, feat_dim] = [density, base_color(3), normed_grd(3), ...]
    densities = cell_values[:, 0]                          # [T]
    base_colors = cell_values[:, 1:4].reshape(-1)          # [T * 3]
    color_grads = cell_values[:, 4:7].reshape(-1)          # [T * 3]
    vertices = model.vertices

    return vertices.reshape(-1), base_colors, densities, color_grads


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

    # Differentiable params — same cell_values as Slang (includes softplus on base_colors)
    vertices_flat, base_colors, densities, color_grads = compute_vk_tensors(model, camera)

    # Camera matrices with scene-appropriate near/far planes
    vp, inv_vp, cam_pos = make_vk_vp(camera, model, device)

    # Forward render via autograd
    image_rgba = RMeshForward.apply(
        renderer, cam_pos, vp, inv_vp,
        vertices_flat, base_colors, densities, color_grads
    )  # [H, W, 4] premultiplied RGBA

    # Use premultiplied RGB directly — avoids gradient explosion from dividing by small alpha.
    # Background pixels (alpha≈0) contribute black, which is correct for black-background GT.
    # Don't clamp RGB — hard clamp kills gradients for out-of-range pixels.
    alpha = image_rgba[..., 3].clamp(min=0, max=1)
    rgb = image_rgba[..., :3]

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
