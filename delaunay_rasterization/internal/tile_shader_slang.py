import torch
import delaunay_rasterization.internal.slang.slang_modules as slang_modules
import math
from delaunay_rasterization.internal.sort_by_keys import sort_by_keys_cub
from icecream import ic

def ceil_div(x, y):
    return (x + y - 1) // y

def vertex_and_tile_shader(indices,
                           vertices,
                           densitites,
                           world_view_transform,
                           K,
                           cam_pos,
                           fovy,
                           fovx,
                           render_grid):
    """
    Vertex and Tile Shader for 3D Tetrahedra Rasterization.

    Args:
      indices: Tensor with tetrahedra indices [N, 4].
      vertices: Tensor with world-space coordinates of vertices [M, 3].
      world_view_transform: The World to View-Space Camera transformation.
      K: The View to Screen-Space(Projection) Matrix.
      cam_pos: The camera position.
      fovy: The vertical Field of View in radians.
      fovx: The horizontal Field of View in radians.
      render_grid: Describes the resolution of the image and the tiling resolution.
   
    Returns:
      sorted_tetra_idx: A list of indices that describe the sorted order for rendering tetrahedra. [M, 1]
      tile_ranges: Describes the range of tetrahedra in the sorted_tetra_idx for each tile. [T, 2]
      radii: The radius of the bounding sphere of each tetrahedron. [N, 1]
      vs_tetra: Tensor with view-space coordinates of tetrahedra centroids [N, 3].
      circumcenter: Tensor with the circumcenter of each tetrahedron in view-space [N, 3].
    """
    n_tetra = indices.shape[0]
    tiles_touched, rect_tile_space, vs_tetra, circumcenter = VertexShader.apply(
        indices, 
        vertices,
        densitites,
        world_view_transform,
        K,
        cam_pos,
        fovy,
        fovx,
        render_grid)

    inds, = torch.where(vs_tetra[:, 1] == 1)
    if len(inds) > 0:
        min_tet_z = vs_tetra[inds[0], 2].clone()
        z_val = vs_tetra[:, 2].clone()

        behind_mask = z_val < min_tet_z
        tiles_touched[behind_mask] = 0
        rect_tile_space[behind_mask, :] = 0
        vs_tetra[:, 2] -= min_tet_z
        vs_tetra[behind_mask, 2] = 0

    with torch.no_grad():
        # w = rect_tile_space[:, 2] - rect_tile_space[:, 0]
        # h = rect_tile_space[:, 3] - rect_tile_space[:, 1]
        # tiles_touched = w * h
        index_buffer_offset = torch.cumsum(tiles_touched, dim=0, dtype=tiles_touched.dtype)
        total_size_index_buffer = index_buffer_offset[-1]
        unsorted_keys = torch.zeros((total_size_index_buffer,), 
                                    device="cuda", 
                                    dtype=torch.int64)
        unsorted_tetra_idx = torch.zeros((total_size_index_buffer,), 
                                         device="cuda", 
                                         dtype=torch.int32)
        # ic(total_size_index_buffer, tiles_touched.max(), index_buffer_offset.max(), tiles_touched.long().sum())
        # must be positive for key sort
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
        torch.cuda.synchronize()
        sorted_keys, sorted_tetra_idx = sort_by_keys_cub.sort_by_keys(
            unsorted_keys, unsorted_tetra_idx, highest_tile_id_msb)

        torch.cuda.synchronize()
        tile_ranges = torch.zeros((render_grid.grid_height*render_grid.grid_width, 2), 
                                  device="cuda",
                                  dtype=torch.int32)
        slang_modules.tile_shader.compute_tile_ranges(sorted_keys=sorted_keys,
                                                      out_tile_ranges=tile_ranges).launchRaw(
                blockSize=(256, 1, 1),
                gridSize=(ceil_div(total_size_index_buffer, 256).item(), 1, 1)
        )
        torch.cuda.synchronize()
        # ic(tlen.min(), tlen.max(), tlen.float().mean())
        # if tile_ranges[-1, 1] != total_size_index_buffer:
        #     tlen = tile_ranges[:, 1] - tile_ranges[:, 0]
        #     inds = torch.arange(sorted_keys.shape[0], device=sorted_keys.device)
        #     ic(highest_tile_id_msb, (inds == sorted_keys.argsort()).all(), sorted_keys.shape, unsorted_keys.shape)
        #     ic(render_grid.grid_height*render_grid.grid_width, sorted_keys[1] >> 32, sorted_keys[0] >> 32, sorted_keys[-2] >> 32, sorted_keys[-1] >> 32)
        #     ic(sorted_keys.shape, tile_ranges[-1], total_size_index_buffer, sorted_keys.min(), sorted_keys.max())
        #     ic(tile_ranges)
        #     # tile_ranges[sorted_keys[-1] >> 32, 1] = total_size_index_buffer
        #     ic(rect_tile_space.max())
        #     last_tile_used = (sorted_keys[-1] >> 32)
        #     tile_ranges[last_tile_used, 1] = total_size_index_buffer
        #     # ic((torch.arange(test_ids.shape[0], device=test_ids.device) == test_ids).all(), torch.arange(test_ids.shape[0]), test_ids)
        #     # ic(sorted_keys >> 32)
        #     # ic(tile_ranges.shape, sorted_keys.shape, tile_ranges[-1], total_size_index_buffer)
        #     # # ic(grid_size, grid_size*256)
        #     print("issue with tile ranges")

    mask = tiles_touched > 0
    return sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, rect_tile_space


class VertexShader(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                indices, vertices, densitites,
                world_view_transform, K, cam_pos,
                fovy, fovx,
                render_grid, device="cuda"):
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
                                                  densities=densitites,
                                                  world_view_transform=world_view_transform,
                                                  K=K,
                                                  cam_pos=cam_pos,
                                                  out_tiles_touched=tiles_touched,
                                                  out_rect_tile_space=rect_tile_space,
                                                  out_vs=vs_tetra,
                                                  out_circumcenter=circumcenter,
                                                  fovy=fovy,
                                                  fovx=fovx,
                                                  image_height=render_grid.image_height,
                                                  image_width=render_grid.image_width,
                                                  grid_height=render_grid.grid_height,
                                                  grid_width=render_grid.grid_width,
                                                  tile_height=render_grid.tile_height,
                                                  tile_width=render_grid.tile_width).launchRaw(
                blockSize=(256, 1, 1),
                gridSize=(ceil_div(n_tetra, 256), 1, 1)
        )

        ctx.save_for_backward(indices, vertices, world_view_transform, K, cam_pos,
                              tiles_touched, rect_tile_space, vs_tetra, circumcenter, densitites)
        ctx.render_grid = render_grid
        ctx.fovy = fovy
        ctx.fovx = fovx

        return tiles_touched, rect_tile_space, vs_tetra, circumcenter
    
    @staticmethod
    def backward(ctx, grad_tiles_touched, grad_rect_tile_space, grad_vs_tetra, grad_circumcenter):
        (indices, vertices, world_view_transform, K, cam_pos,
         tiles_touched, rect_tile_space, vs_tetra, circumcenter, densitites) = ctx.saved_tensors
        render_grid = ctx.render_grid
        fovy = ctx.fovy
        fovx = ctx.fovx

        n_tetra = indices.shape[0]

        grad_indices = torch.zeros_like(indices)
        grad_vertices = torch.zeros_like(vertices)

        slang_modules.vertex_shader.vertex_shader.bwd(indices=indices,
                                                      vertices=(vertices, grad_vertices),
                                                      densities=densitites,
                                                      world_view_transform=world_view_transform,
                                                      K=K,
                                                      cam_pos=cam_pos,
                                                      out_tiles_touched=tiles_touched,
                                                      out_rect_tile_space=rect_tile_space,
                                                      out_vs=(vs_tetra, grad_vs_tetra),
                                                      out_circumcenter=(circumcenter, grad_circumcenter),
                                                      fovy=fovy,
                                                      fovx=fovx,
                                                      image_height=render_grid.image_height,
                                                      image_width=render_grid.image_width,
                                                      grid_height=render_grid.grid_height,
                                                      grid_width=render_grid.grid_width,
                                                      tile_height=render_grid.tile_height,
                                                      tile_width=render_grid.tile_width).launchRaw(
                blockSize=(256, 1, 1),
                gridSize=(ceil_div(n_tetra, 256), 1, 1)
        )
        return grad_indices, grad_vertices, None, None, None, None, None
