import torch
from delaunay_rasterization.internal.render_grid import RenderGrid
import delaunay_rasterization.internal.slang.slang_modules as slang_modules
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from icecream import ic
import time
from delaunay_rasterization.internal.util import recombine_tensors, split_tensors

class AlphaBlendTiledRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                sorted_tetra_idx, tile_ranges,
                indices, vertices, tet_density, render_grid,
                tcam, ray_jitter, device="cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distortion_img = torch.zeros((render_grid.image_height, 
                                  render_grid.image_width, 5), 
                                 device=device)
        output_img = torch.zeros((render_grid.image_height, 
                                  render_grid.image_width, 4), 
                                 device=device)
        n_contributors = torch.zeros((render_grid.image_height, 
                                      render_grid.image_width, 1),
                                     dtype=torch.int32, device=device)

        tet_alive = torch.zeros((indices.shape[0]), dtype=bool, device=device)
        assert (render_grid.tile_height, render_grid.tile_width) in slang_modules.alpha_blend_shaders_interp, (
            'Alpha Blend Shader was not compiled for this tile'
            f' {render_grid.tile_height}x{render_grid.tile_width} configuration, available configurations:'
            f' {slang_modules.alpha_blend_shaders_interp.keys()}'
        )

        alpha_blend_tile_shader = slang_modules.alpha_blend_shaders_interp[(render_grid.tile_height, render_grid.tile_width)]
        st = time.time()
        splat_kernel_with_args = alpha_blend_tile_shader.splat_tiled(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=vertices,
            tet_density=tet_density,
            output_img=output_img,
            distortion_img=distortion_img,
            n_contributors=n_contributors,
            tet_alive=tet_alive,
            tcam=tcam,
            ray_jitter=ray_jitter,
        )
        splat_kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )

        tensors = [
            sorted_tetra_idx, tile_ranges,
            indices, vertices, tet_density, 
            output_img, distortion_img, n_contributors,
            ray_jitter
        ]
        non_tensor_data, tensor_data = split_tensors(tcam)
        ctx.save_for_backward(*tensors, *tensor_data)
        ctx.non_tensor_data = non_tensor_data
        ctx.len_tensors = len(tensors)

        ctx.render_grid = render_grid

        return output_img, distortion_img, tet_alive

    @staticmethod
    def backward(ctx, grad_output_img, grad_distortion_img, grad_vert_alive):
        (sorted_tetra_idx, tile_ranges, 
         indices, vertices, tet_density,
         output_img, distortion_img, n_contributors,
            ray_jitter) = ctx.saved_tensors[:ctx.len_tensors]
        tcam = recombine_tensors(ctx.non_tensor_data, ctx.saved_tensors[ctx.len_tensors:])
        render_grid = ctx.render_grid

        vertices_grad = torch.zeros_like(vertices)
        tet_density_grad = torch.zeros_like(tet_density)

        assert (render_grid.tile_height, render_grid.tile_width) in slang_modules.alpha_blend_shaders_interp, (
            'Alpha Blend Shader was not compiled for this tile'
            f' {render_grid.tile_height}x{render_grid.tile_width} configuration, available configurations:'
            f' {slang_modules.alpha_blend_shaders_interp.keys()}'
        )

        alpha_blend_tile_shader = slang_modules.alpha_blend_shaders_interp[(render_grid.tile_height, render_grid.tile_width)]

        tet_alive = torch.zeros((indices.shape[0]), dtype=bool, device=vertices.device)
        st = time.time()
        kernel_with_args = alpha_blend_tile_shader.splat_tiled.bwd(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=(vertices, vertices_grad),
            tcam=tcam,
            tet_density=(tet_density, tet_density_grad),
            output_img=(output_img, grad_output_img),
            distortion_img=(distortion_img, grad_distortion_img),
            n_contributors=n_contributors,
            tet_alive=tet_alive,
            ray_jitter=ray_jitter)
        
        kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )

        return (None, None, None, vertices_grad, tet_density_grad, 
                None, None, None)
