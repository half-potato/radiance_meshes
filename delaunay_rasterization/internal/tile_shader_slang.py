import torch
import delaunay_rasterization.internal.slang.slang_modules as slang_modules
import math
from delaunay_rasterization.internal.sort_by_keys import sort_by_keys_cub
from icecream import ic
from delaunay_rasterization.internal.util import recombine_tensors, split_tensors

def augment(v):
    return torch.cat([v, torch.ones_like(v[:, :1])], dim=-1)

def augmentT(v):
    return torch.cat([v, torch.ones_like(v[:1])], dim=0)

def point2image(vertices, viewmat, projection_matrix, cam_pos, eps=torch.finfo(torch.float32).eps):
    cam_space_homo = viewmat @ augment(vertices).T
    cam_space_nohomo = cam_space_homo[:2] / (cam_space_homo[2:3].abs() + 1e-10)
    pixel_space = projection_matrix @ augmentT(cam_space_nohomo)
    inv_distance  = 1 / torch.clip(cam_space_homo[2:3].T, eps, None)
    return torch.cat([pixel_space[:2].T, inv_distance], dim=1)

def ceil_div(x, y):
    return (x + y - 1) // y

def vertex_and_tile_shader(indices,
                           vertices,
                           cam,
                           render_grid):
    n_tetra = indices.shape[0]
    tiles_touched, rect_tile_space, vs_tetra, circumcenter = VertexShader.apply(
        indices, 
        vertices,
        cam,
        render_grid)
    # ic((vs_tetra[:, 1] == 1).sum())

    with torch.no_grad():
        index_buffer_offset = torch.cumsum(tiles_touched, dim=0, dtype=tiles_touched.dtype)
        total_size_index_buffer = index_buffer_offset[-1]
        unsorted_keys = torch.zeros((total_size_index_buffer,), 
                                    device="cuda", 
                                    dtype=torch.int64)
        unsorted_tetra_idx = torch.zeros((total_size_index_buffer,), 
                                         device="cuda", 
                                         dtype=torch.int32)

        slang_modules.tile_shader.generate_keys(xyz_vs=vs_tetra,
                                                rect_tile_space=rect_tile_space,
                                                index_buffer_offset=index_buffer_offset,
                                                out_unsorted_keys=unsorted_keys,
                                                out_unsorted_gauss_idx=unsorted_tetra_idx,
                                                grid_height=render_grid.grid_height,
                                                grid_width=render_grid.grid_width).launchRaw(
              blockSize=(256, 1, 1),
              gridSize=(ceil_div(n_tetra, 256), 1, 1)
        )

        highest_tile_id_msb = (render_grid.grid_width*render_grid.grid_height).bit_length()
        sorted_keys, sorted_tetra_idx = sort_by_keys_cub.sort_by_keys(
            unsorted_keys, unsorted_tetra_idx, highest_tile_id_msb)

        tile_ranges = torch.zeros((render_grid.grid_height*render_grid.grid_width, 2), 
                                  device="cuda",
                                  dtype=torch.int32)
        if total_size_index_buffer > 0:
            slang_modules.tile_shader.compute_tile_ranges(sorted_keys=sorted_keys,
                                                        out_tile_ranges=tile_ranges).launchRaw(
                    blockSize=(256, 1, 1),
                    gridSize=(ceil_div(total_size_index_buffer, 256).item(), 1, 1)
            )

    mask = tiles_touched > 0
    return sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, rect_tile_space


class VertexShader(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, vertices, tcam, render_grid, device="cuda"):
        assert not torch.isnan(vertices).any(), "Tensor contains NaN values!"
        n_tetra = indices.shape[0]
        tiles_touched = torch.zeros((n_tetra), 
                                    device="cuda", 
                                    dtype=torch.int32)
        rect_tile_space = torch.zeros((n_tetra, 4), 
                                      device="cuda", 
                                      dtype=torch.int32)
        
        vs_tetra = torch.zeros((n_tetra, 3),
                               device="cuda",
                               dtype=torch.float)
        circumcenter = torch.zeros((n_tetra, 3),
                                   device="cuda",
                                   dtype=torch.float)
        
        slang_modules.vertex_shader.vertex_shader(indices=indices,
                                                  vertices=vertices,
                                                  out_tiles_touched=tiles_touched,
                                                  out_rect_tile_space=rect_tile_space,
                                                  out_vs=vs_tetra,
                                                  out_circumcenter=circumcenter,
                                                  tcam=tcam).launchRaw(
                blockSize=(256, 1, 1),
                gridSize=(ceil_div(n_tetra, 256), 1, 1)
        )

        tensors = [
            indices, vertices,
            tiles_touched, rect_tile_space, vs_tetra, circumcenter
        ]
        non_tensor_data, tensor_data = split_tensors(tcam)
        ctx.non_tensor_data = non_tensor_data
        ctx.len_tensors = len(tensors)
        ctx.save_for_backward(*tensors, *tensor_data)
        ctx.render_grid = render_grid

        return tiles_touched, rect_tile_space, vs_tetra, circumcenter
    
    @staticmethod
    def backward(ctx, grad_tiles_touched, grad_rect_tile_space, grad_vs_tetra, grad_circumcenter):
        (indices, vertices,
            tiles_touched, rect_tile_space, vs_tetra, circumcenter) = ctx.saved_tensors[:ctx.len_tensors]
        tcam = recombine_tensors(ctx.non_tensor_data, ctx.saved_tensors[ctx.len_tensors:])
        render_grid = ctx.render_grid

        n_tetra = indices.shape[0]

        grad_indices = torch.zeros_like(indices)
        grad_vertices = torch.zeros_like(vertices)

        slang_modules.vertex_shader.vertex_shader.bwd(indices=indices,
                                                      vertices=(vertices, grad_vertices),
                                                      tcam=tcam,
                                                      out_tiles_touched=tiles_touched,
                                                      out_rect_tile_space=rect_tile_space,
                                                      out_vs=(vs_tetra, grad_vs_tetra),
                                                      out_circumcenter=(circumcenter, grad_circumcenter),
                                                      ).launchRaw(
                blockSize=(256, 1, 1),
                gridSize=(ceil_div(n_tetra, 256), 1, 1)
        )
        return grad_indices, grad_vertices, None, None
