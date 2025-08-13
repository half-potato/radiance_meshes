import numpy as np
import torch
from delaunay_rasterization.internal.alphablend_tiled_slang import render_constant_color
from delaunay_rasterization.internal.alphablend_tiled_slang_interp import AlphaBlendTiledRender as Render
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from data.camera import Camera

import matplotlib.pyplot as plt
cmap = plt.get_cmap("jet")

def render_debug(render_tensor, model, camera, density_multi=1):

    # Convert to RGB (NxMx3) using the colormap
    _, features = model.get_cell_values(camera)
    tet_grad_color = torch.zeros((features.shape[0], 4), device=features.device)
    if render_tensor.shape[1] == 1:
        tensor_min, tensor_max = render_tensor.min(), torch.quantile(render_tensor, 0.99)
        normalized_tensor = ((render_tensor - tensor_min) / (tensor_max - tensor_min)).clip(0, 1)
        normalized_tensor = torch.as_tensor(
            cmap(normalized_tensor.reshape(-1).cpu().numpy())).float().cuda()
    else:
        normalized_tensor = render_tensor
    tet_grad_color[:, :normalized_tensor.shape[1]] = normalized_tensor
    if render_tensor.shape[1] < 4:
        tet_grad_color[:, 3] = features[:, 0] * density_multi# * render_tensor.reshape(-1)
    render_pkg = render_constant_color(model.indices, model.vertices, None, camera, cell_values=tet_grad_color)

    image = render_pkg['render']
    image = image.permute(1, 2, 0)
    image = (image.detach().cpu().numpy() * 255).clip(min=0, max=255).astype(np.uint8)

    del render_pkg, render_tensor
    return image


def render(camera: Camera, model, cell_values=None, tile_size=16, min_t=0.1,
           scene_scaling=1, clip_multi=0, ray_jitter=None, glo=None,
           **kwargs):
    device = model.device
    if ray_jitter is None:
        ray_jitter = 0.5*torch.ones((camera.image_height, camera.image_width, 2), device=device)
    else:
        assert(ray_jitter.shape[0] == camera.image_height)
        assert(ray_jitter.shape[1] == camera.image_width)
        assert(ray_jitter.shape[2] == 2)
    vertices = model.vertices
    
    render_grid = RenderGrid(camera.image_height,
                             camera.image_width,
                             tile_height=tile_size,
                             tile_width=tile_size)
    tcam = dict(
        tile_height=tile_size,
        tile_width=tile_size,
        grid_height=render_grid.grid_height,
        grid_width=render_grid.grid_width,
        min_t=min_t,
        **camera.to_dict(device)
    )
    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _ = vertex_and_tile_shader(
        model.indices,
        vertices,
        tcam,
        render_grid)
    extras = {}
    if cell_values is None:
        cell_values = torch.zeros((mask.shape[0], model.feature_dim), device=circumcenter.device)
        if mask.sum() > 0 and model.mask_values:
            normed_cc, cell_values[mask] = model.get_cell_values(camera, mask, circumcenter[mask], glo=glo)
        else:
            normed_cc, cell_values = model.get_cell_values(camera, all_circumcenters=circumcenter, glo=glo)
        if clip_multi > 0 and not model.frozen:
            with torch.no_grad():
                tet_sens, sensitivity = topo_utils.compute_vertex_sensitivity(model.indices[mask],
                                                                            vertices, normed_cc, True)
                scaling = clip_multi*sensitivity.reshape(-1, 1).clip(min=1e-5)
            vertices = ClippedGradients.apply(vertices, scaling)

    image_rgb, distortion_img, tet_alive = Render.apply(
        sorted_tetra_idx,
        tile_ranges,
        model.indices,
        vertices,
        cell_values,
        render_grid,
        tcam,
        ray_jitter)
    alpha = image_rgb.permute(2,0,1)[3, ...]
    total_density = (distortion_img[:, :, 2]**2).clip(min=1e-6)
    distortion_loss = (((distortion_img[:, :, 0] - distortion_img[:, :, 1]) + distortion_img[:, :, 4]) / total_density).clip(min=0)

    
    render_pkg = {
        'render': image_rgb.permute(2,0,1)[:3, ...],
        'alpha': alpha,
        'distortion_img': distortion_img,
        'distortion_loss': distortion_loss.mean(),
        'visibility_filter': mask,
        'circumcenters': circumcenter,
        'density': cell_values[:, 0],
        'color': cell_values[:, 1:],
        'mask': mask,
        **extras
    }
    return render_pkg
