import torch
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.slang.slang_modules import shader_manager
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from icecream import ic
import time
from delaunay_rasterization.internal.util import recombine_tensors, split_tensors

class AlphaBlendTiledRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                sorted_tetra_idx, tile_ranges,
                indices, vertices, cell_values, render_grid,
                tcam, ray_jitter, aux_dim, device="cuda"
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distortion_img = torch.zeros((render_grid.image_height, 
                                  render_grid.image_width, 5), 
                                 device=device)
        output_img = torch.zeros((render_grid.image_height, 
                                  render_grid.image_width, 4 + aux_dim), 
                                 device=device)
        xyzd_img = torch.zeros((render_grid.image_height, 
                                render_grid.image_width, 4), 
                                 device=device)
        n_contributors = torch.zeros((render_grid.image_height, 
                                      render_grid.image_width, 1),
                                     dtype=torch.int32, device=device)

        tet_alive = torch.zeros((indices.shape[0]), dtype=bool, device=device)

        assert(len(ray_jitter.shape) == 3)
        assert(ray_jitter.shape[0] == render_grid.image_height)
        assert(ray_jitter.shape[1] == render_grid.image_width)
        assert(ray_jitter.shape[2] == 2)

        alpha_blend_tile_shader = shader_manager.get_interp(render_grid.tile_height, render_grid.tile_width, aux_dim)
        st = time.time()
        splat_kernel_with_args = alpha_blend_tile_shader.splat_tiled(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=vertices,
            cell_values=cell_values,
            output_img=output_img,
            xyzd_img=xyzd_img,
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
            indices, vertices, cell_values, 
            output_img, xyzd_img, distortion_img, n_contributors,
            ray_jitter
        ]
        non_tensor_data, tensor_data = split_tensors(tcam)
        ctx.save_for_backward(*tensors, *tensor_data)
        ctx.non_tensor_data = non_tensor_data
        ctx.len_tensors = len(tensors)
        ctx.aux_dim = aux_dim

        ctx.render_grid = render_grid

        return output_img, xyzd_img, distortion_img, tet_alive

    @staticmethod
    def backward(ctx, grad_output_img, grad_xyzd_img, grad_distortion_img, grad_vert_alive):
        (sorted_tetra_idx, tile_ranges, 
         indices, vertices, cell_values,
         output_img, xyzd_img, distortion_img, n_contributors,
            ray_jitter) = ctx.saved_tensors[:ctx.len_tensors]
        tcam = recombine_tensors(ctx.non_tensor_data, ctx.saved_tensors[ctx.len_tensors:])
        render_grid = ctx.render_grid

        vertices_grad = torch.zeros_like(vertices)
        cell_values_grad = torch.zeros_like(cell_values)

        alpha_blend_tile_shader = shader_manager.get_interp(render_grid.tile_height, render_grid.tile_width, ctx.aux_dim)

        tet_alive = torch.zeros((indices.shape[0]), dtype=bool, device=vertices.device)
        st = time.time()
        kernel_with_args = alpha_blend_tile_shader.splat_tiled.bwd(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=(vertices, vertices_grad),
            tcam=tcam,
            cell_values=(cell_values, cell_values_grad),
            output_img=(output_img, grad_output_img),
            xyzd_img=(xyzd_img, grad_xyzd_img),
            distortion_img=(distortion_img, grad_distortion_img),
            n_contributors=n_contributors,
            tet_alive=tet_alive,
            ray_jitter=ray_jitter,
        )
        
        kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )

        return (None, None, None, vertices_grad, cell_values_grad, 
                None, None, None, None)
