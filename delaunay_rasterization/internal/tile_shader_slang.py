import torch
import delaunay_rasterization.internal.slang.slang_modules as slang_modules
import math
from delaunay_rasterization.internal.sort_by_keys import sort_by_keys_cub
from icecream import ic

def vertex_and_tile_shader(indices,
                           vertices,
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
    tiles_touched, rect_tile_space, radii, vs_tetra, circumcenter = VertexShader.apply(indices, 
                                                                                       vertices,
                                                                                       world_view_transform,
                                                                                       K,
                                                                                       cam_pos,
                                                                                       fovy,
                                                                                       fovx,
                                                                                       render_grid)

    # ic(tiles_touched.float().mean(), tiles_touched.max(), (tiles_touched.max() == tiles_touched).sum())
    # ic((vs_tetra[(tiles_touched.max() == tiles_touched), 2] > 0).sum())
    # ic((vs_tetra[(tiles_touched.max() == tiles_touched), 2] < 0).sum())
    # ic(tiles_touched, rect_tile_space)
    with torch.no_grad():
        mask = tiles_touched > 0
        index_buffer_offset = torch.cumsum(tiles_touched, dim=0, dtype=tiles_touched.dtype)
        total_size_index_buffer = index_buffer_offset[-1]
        unsorted_keys = torch.zeros((total_size_index_buffer,), 
                                    device="cuda", 
                                    dtype=torch.int64)
        unsorted_tetra_idx = torch.zeros((total_size_index_buffer,), 
                                         device="cuda", 
                                         dtype=torch.int32)
        # must be positive for key sort
        biased_xyz_vs = vs_tetra - vs_tetra.min(dim=0, keepdim=True).values
        slang_modules.tile_shader.generate_keys(xyz_vs=biased_xyz_vs,
                                                rect_tile_space=rect_tile_space,
                                                index_buffer_offset=index_buffer_offset,
                                                out_unsorted_keys=unsorted_keys,
                                                out_unsorted_gauss_idx=unsorted_tetra_idx,
                                                grid_height=render_grid.grid_height,
                                                grid_width=render_grid.grid_width).launchRaw(
              blockSize=(256, 1, 1),
              gridSize=(math.ceil(n_tetra/256), 1, 1)
        )    

        highest_tile_id_msb = (render_grid.grid_width*render_grid.grid_height).bit_length()
        sorted_keys, sorted_tetra_idx = sort_by_keys_cub.sort_by_keys(unsorted_keys, unsorted_tetra_idx, highest_tile_id_msb)

        tile_ranges = torch.zeros((render_grid.grid_height*render_grid.grid_width, 2), 
                                  device="cuda",
                                  dtype=torch.int32)
        slang_modules.tile_shader.compute_tile_ranges(sorted_keys=sorted_keys,
                                                      out_tile_ranges=tile_ranges).launchRaw(
                blockSize=(256, 1, 1),
                gridSize=(max(math.ceil(total_size_index_buffer/256), 1), 1, 1)
        )

    return sorted_tetra_idx, tile_ranges, radii, vs_tetra, circumcenter, mask


class VertexShader(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                indices, vertices,
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
        radii = torch.zeros((n_tetra),
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
                                                  world_view_transform=world_view_transform,
                                                  K=K,
                                                  cam_pos=cam_pos,
                                                  out_tiles_touched=tiles_touched,
                                                  out_rect_tile_space=rect_tile_space,
                                                  out_radii=radii,
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
                gridSize=(math.ceil(n_tetra/256), 1, 1)
        )

        ctx.save_for_backward(indices, vertices, world_view_transform, K, cam_pos,
                              tiles_touched, rect_tile_space, radii, vs_tetra, circumcenter)
        ctx.render_grid = render_grid
        ctx.fovy = fovy
        ctx.fovx = fovx

        return tiles_touched, rect_tile_space, radii, vs_tetra, circumcenter
    
    @staticmethod
    def backward(ctx, grad_tiles_touched, grad_rect_tile_space, grad_radii, grad_vs_tetra, grad_circumcenter):
        (indices, vertices, world_view_transform, K, cam_pos,
         tiles_touched, rect_tile_space, radii, vs_tetra, circumcenter) = ctx.saved_tensors
        render_grid = ctx.render_grid
        fovy = ctx.fovy
        fovx = ctx.fovx

        n_tetra = indices.shape[0]

        grad_indices = torch.zeros_like(indices)
        grad_vertices = torch.zeros_like(vertices)

        slang_modules.vertex_shader.vertex_shader.bwd(indices=indices,
                                                      vertices=(vertices, grad_vertices),
                                                      world_view_transform=world_view_transform,
                                                      K=K,
                                                      cam_pos=cam_pos,
                                                      out_tiles_touched=tiles_touched,
                                                      out_rect_tile_space=rect_tile_space,
                                                      out_radii=radii,
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
                gridSize=(math.ceil(n_tetra/256), 1, 1)
        )
        return grad_indices, grad_vertices, None, None, None, None, None, None
