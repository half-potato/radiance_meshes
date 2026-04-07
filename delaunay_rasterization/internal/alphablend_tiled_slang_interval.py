import torch
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.slang.slang_modules import shader_manager
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from delaunay_rasterization.internal.util import recombine_tensors, split_tensors


def ceil_div(x, y):
    return (x + y - 1) // y


def run_interval_generate(indices, vertices, vertex_normals, cell_values,
                          tcam, aux_dim, device="cuda"):
    """Run the interval generation compute prepass.

    Returns:
        interval_verts: [N, 5, 10] fan vertex data
        interval_tet_data: [N, 7 + aux_dim] per-tet data
        interval_meta: [N] uint32 num_sil_verts per tet
    """
    n_tets = indices.shape[0]

    interval_verts = torch.zeros((n_tets, 5, 10), device=device)
    interval_tet_data = torch.zeros((n_tets, 7 + aux_dim), device=device)
    interval_meta = torch.zeros((n_tets,), dtype=torch.int32, device=device)

    gen_shader = shader_manager.get_interval_generate(aux_dim)

    gen_shader.interval_generate(
        indices=indices,
        vertices=vertices,
        vertex_normals=vertex_normals,
        cell_values=cell_values,
        interval_verts=interval_verts,
        interval_tet_data=interval_tet_data,
        interval_meta=interval_meta,
        tcam=tcam,
    ).launchRaw(
        blockSize=(256, 1, 1),
        gridSize=(ceil_div(n_tets, 256), 1, 1)
    )

    return interval_verts, interval_tet_data, interval_meta


class AlphaBlendTiledRenderInterval(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                sorted_tetra_idx, tile_ranges,
                indices, vertices, vertex_normals, cell_values, render_grid,
                tcam, ray_jitter, aux_dim, device="cuda"
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Run interval generation prepass
        interval_verts, interval_tet_data, interval_meta = run_interval_generate(
            indices, vertices, vertex_normals, cell_values,
            tcam, aux_dim, device)

        # Allocate output tensors
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

        assert(len(ray_jitter.shape) == 3)
        assert(ray_jitter.shape[0] == render_grid.image_height)
        assert(ray_jitter.shape[1] == render_grid.image_width)
        assert(ray_jitter.shape[2] == 2)

        # Launch tile kernel
        interval_shader = shader_manager.get_interval(
            render_grid.tile_height, render_grid.tile_width, aux_dim)

        interval_shader.splat_tiled(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            interval_verts=interval_verts,
            interval_tet_data=interval_tet_data,
            interval_meta=interval_meta,
            output_img=output_img,
            xyzd_img=xyzd_img,
            distortion_img=distortion_img,
            n_contributors=n_contributors,
            ray_jitter=ray_jitter,
            tcam=tcam,
        ).launchRaw(
            blockSize=(render_grid.tile_width,
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width,
                      render_grid.grid_height, 1)
        )

        # Save for backward
        tensors = [
            sorted_tetra_idx, tile_ranges,
            indices, vertices, vertex_normals, cell_values,
            interval_verts, interval_tet_data,
            output_img, xyzd_img, distortion_img, n_contributors,
            ray_jitter
        ]
        non_tensor_data, tensor_data = split_tensors(tcam)
        ctx.save_for_backward(*tensors, *tensor_data)
        ctx.non_tensor_data = non_tensor_data
        ctx.len_tensors = len(tensors)
        ctx.aux_dim = aux_dim
        ctx.render_grid = render_grid
        ctx.interval_meta = interval_meta  # uint32, not a grad tensor

        return output_img, xyzd_img, distortion_img

    @staticmethod
    def backward(ctx, grad_output_img, grad_xyzd_img, grad_distortion_img):
        (sorted_tetra_idx, tile_ranges,
         indices, vertices, vertex_normals, cell_values,
         interval_verts, interval_tet_data,
         output_img, xyzd_img, distortion_img, n_contributors,
         ray_jitter) = ctx.saved_tensors[:ctx.len_tensors]
        tcam = recombine_tensors(ctx.non_tensor_data, ctx.saved_tensors[ctx.len_tensors:])
        render_grid = ctx.render_grid
        interval_meta = ctx.interval_meta

        # Gradient tensors for interval buffers
        interval_verts_grad = torch.zeros_like(interval_verts)
        interval_tet_data_grad = torch.zeros_like(interval_tet_data)

        # Launch backward tile kernel
        interval_shader = shader_manager.get_interval(
            render_grid.tile_height, render_grid.tile_width, ctx.aux_dim)

        interval_shader.splat_tiled.bwd(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            interval_verts=(interval_verts, interval_verts_grad),
            interval_tet_data=(interval_tet_data, interval_tet_data_grad),
            interval_meta=interval_meta,
            output_img=(output_img, grad_output_img),
            xyzd_img=(xyzd_img, grad_xyzd_img),
            distortion_img=(distortion_img, grad_distortion_img),
            n_contributors=n_contributors,
            ray_jitter=ray_jitter,
            tcam=tcam,
        ).launchRaw(
            blockSize=(render_grid.tile_width,
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width,
                      render_grid.grid_height, 1)
        )

        # Chain gradients through interval_generate backward
        vertices_grad = torch.zeros_like(vertices)
        vertex_normals_grad = torch.zeros_like(vertex_normals)
        cell_values_grad = torch.zeros_like(cell_values)

        n_tets = indices.shape[0]
        gen_shader = shader_manager.get_interval_generate(ctx.aux_dim)

        gen_shader.interval_generate.bwd(
            indices=indices,
            vertices=(vertices, vertices_grad),
            vertex_normals=(vertex_normals, vertex_normals_grad),
            cell_values=(cell_values, cell_values_grad),
            interval_verts=(interval_verts, interval_verts_grad),
            interval_tet_data=(interval_tet_data, interval_tet_data_grad),
            interval_meta=interval_meta,
            tcam=tcam,
        ).launchRaw(
            blockSize=(256, 1, 1),
            gridSize=(ceil_div(n_tets, 256), 1, 1)
        )

        return (None, None, None, vertices_grad, vertex_normals_grad, cell_values_grad,
                None, None, None, None)
