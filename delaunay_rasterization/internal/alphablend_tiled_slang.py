import torch
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from icecream import ic
import time
import math
from delaunay_rasterization.internal.util import recombine_tensors, split_tensors
from utils.safe_math import safe_div
from delaunay_rasterization.internal.slang.slang_modules import shader_manager

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def render_constant(camera, indices, vertices, cell_values=None, tile_size=4, min_t=0.1, ray_jitter=None, **kwargs):
    device = vertices.device
    if ray_jitter is None:
        ray_jitter = 0.5*torch.ones((camera.image_height, camera.image_width, 8, 2), device=device)
    else:
        assert(ray_jitter.shape[0] == camera.image_height)
        assert(ray_jitter.shape[1] == camera.image_width)
        assert(ray_jitter.shape[2] == 2)
    vertices = vertices
    
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
        indices,
        vertices,
        tcam,
        render_grid)
    extras = {}
    # if cell_values is None:
    #     cell_values = torch.zeros((mask.shape[0], model.feature_dim), device=circumcenter.device)
    #     if mask.sum() > 0 and model.mask_values:
    #         _, values = model.get_cell_values(camera, mask, circumcenter[mask])
    #         cell_values[mask] = values
    #     else:
    #         _, cell_values = model.get_cell_values(camera, all_circumcenters=circumcenter)

    image_rgb = AlphaBlendTiledRender.apply(
        sorted_tetra_idx,
        tile_ranges,
        indices,
        vertices,
        cell_values,
        render_grid,
        tcam,
        ray_jitter)
    alpha = image_rgb.permute(2,0,1)[-2, ...]
    image = image_rgb.permute(2,0,1)[:-2, ...] * camera.gt_alpha_mask.to(device)
    weight_square = image_rgb.permute(2,0,1)[-1:, ...]
    # latents = (safe_div(image, weight_square) / 3.5).unsqueeze(0).clip(min=-7, max=7)
    # latents = (image / 3.5).unsqueeze(0).clip(min=-7, max=7)
    
    render_pkg = {
        # 'latents': latents,
        'render': image,
        'alpha': alpha,
        'mask': mask,
        'weight_square': weight_square,
        **extras
    }
    return render_pkg



class AlphaBlendTiledRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                sorted_tetra_idx, tile_ranges,
                indices, vertices, rgbs, render_grid,
                tcam, ray_jitter,
                device="cuda"):
        output_img = torch.zeros((render_grid.image_height, 
                                  render_grid.image_width, (rgbs.shape[1]-1)+2), 
                                 device=device)
        n_contributors = torch.zeros((render_grid.image_height, 
                                      render_grid.image_width, 1),
                                     dtype=torch.int32, device=device)

        alpha_blend_tile_shader = shader_manager.get_alphablend(render_grid.tile_height, render_grid.tile_width, 0)
        st = time.time()
        splat_kernel_with_args = alpha_blend_tile_shader.splat_tiled(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=vertices,
            rgbs=rgbs,
            output_img=output_img,
            n_contributors=n_contributors,
            tcam=tcam,
            ray_jitter=ray_jitter,
        )
        splat_kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )
        # torch.cuda.synchronize()
        # ic("ab", time.time()-st)
        # ic(n_contributors.float().mean(), n_contributors.max())

        tensors = [
            sorted_tetra_idx, tile_ranges,
            indices, vertices, rgbs, 
            output_img, n_contributors, ray_jitter]
        non_tensor_data, tensor_data = split_tensors(tcam)
        ctx.save_for_backward(*tensors, *tensor_data)
        ctx.non_tensor_data = non_tensor_data
        ctx.len_tensors = len(tensors)

        ctx.render_grid = render_grid

        return output_img

    @staticmethod
    def backward(ctx, grad_output_img):
        (sorted_tetra_idx, tile_ranges,
            indices, vertices, rgbs, 
            output_img, n_contributors, ray_jitter) = ctx.saved_tensors[:ctx.len_tensors]
        tcam = recombine_tensors(ctx.non_tensor_data, ctx.saved_tensors[ctx.len_tensors:])
        render_grid = ctx.render_grid

        vertices_grad = torch.zeros_like(vertices)
        rgbs_grad = torch.zeros_like(rgbs)

        alpha_blend_tile_shader = shader_manager.get_alphablend(render_grid.tile_height, render_grid.tile_width, 0)

        st = time.time()
        kernel_with_args = alpha_blend_tile_shader.splat_tiled.bwd(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=(vertices, vertices_grad),
            rgbs=(rgbs, rgbs_grad),
            output_img=(output_img, grad_output_img),
            n_contributors=n_contributors,
            tcam=tcam,
            ray_jitter=ray_jitter)
        
        kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )
        
        return (None, None, None, vertices_grad, rgbs_grad, None, None, None)
