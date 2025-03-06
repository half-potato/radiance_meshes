import torch
import delaunay_rasterization.internal.slang.slang_modules as slang_modules
import math
from delaunay_rasterization.internal.sort_by_keys import sort_by_keys_cub
from icecream import ic

def augment(v):
    return torch.cat([v, torch.ones_like(v[:, :1])], dim=-1)

def augmentT(v):
    return torch.cat([v, torch.ones_like(v[:1])], dim=0)

def point2image(vertices, viewmat, projection_matrix, cam_pos, eps=torch.finfo(torch.float32).eps):
    cam_space_homo = viewmat @ augment(vertices).T
    cam_space_nohomo = cam_space_homo[:2] / (cam_space_homo[2:3].abs() + 1e-10)
    pixel_space = projection_matrix @ augmentT(cam_space_nohomo)
    inv_distance  = 1 / torch.clip(cam_space_homo[2:3].T, eps, None)
    # inv_distance  = 1 / torch.sqrt(
    #     torch.clip(torch.sum((vertices - cam_pos.reshape(1, 3))**2, dim=1, keepdim=True), eps, None)
    # )
    return torch.cat([pixel_space[:2].T, inv_distance], dim=1)

def ceil_div(x, y):
    return (x + y - 1) // y

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
    tiles_touched, rect_tile_space, vs_tetra, circumcenter, tet_area = VertexShader.apply(
        indices, 
        vertices,
        world_view_transform,
        K,
        cam_pos,
        fovy,
        fovx,
        render_grid)

    with torch.no_grad():
        index_buffer_offset = torch.cumsum(tiles_touched, dim=0, dtype=tiles_touched.dtype)
        total_size_index_buffer = index_buffer_offset[-1]
        unsorted_keys = torch.zeros((total_size_index_buffer,), 
                                    device="cuda", 
                                    dtype=torch.int64)
        unsorted_tetra_idx = torch.zeros((total_size_index_buffer,), 
                                         device="cuda", 
                                         dtype=torch.int32)
        # slang_modules.tile_shader.generate_keys_smart(xyz_vs=vs_tetra,
        #                                         vertices=vertices,
        #                                         indices=indices,
        #                                         rect_tile_space=rect_tile_space,
        #                                         index_buffer_offset=index_buffer_offset,
        #                                         out_unsorted_keys=unsorted_keys,
        #                                         out_unsorted_gauss_idx=unsorted_tetra_idx,
        #                                         grid_height=render_grid.grid_height,
        #                                         grid_width=render_grid.grid_width,
        #                                         fovy=fovy,
        #                                         fovx=fovx,
        #                                         world_view_transform=world_view_transform,
        #                                         K=K,
        #                                         cam_pos=cam_pos,
        #                                         tiles_touched=tiles_touched,
        #                                         image_height=render_grid.image_height,
        #                                         image_width=render_grid.image_width,
        #                                         tile_height=render_grid.tile_height,
        #                                         tile_width=render_grid.tile_width).launchRaw(
        #       blockSize=(256, 1, 1),
        #       gridSize=(ceil_div(n_tetra, 256), 1, 1)
        # )

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
    return sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, rect_tile_space, tet_area


class VertexShader(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                indices, vertices,
                world_view_transform, K, cam_pos,
                fovy, fovx,
                render_grid, device="cuda"):
        assert not torch.isnan(vertices).any(), "Tensor contains NaN values!"
        n_tetra = indices.shape[0]
        tiles_touched = torch.zeros((n_tetra), 
                                    device="cuda", 
                                    dtype=torch.int32)
        rect_tile_space = torch.zeros((n_tetra, 4), 
                                      device="cuda", 
                                      dtype=torch.int32)
        tet_area = torch.ones((n_tetra), 
                                device="cuda", 
                                dtype=torch.float)
        
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
                                                  out_tet_area=tet_area,
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
                              tiles_touched, rect_tile_space, vs_tetra, circumcenter, tet_area)
        ctx.render_grid = render_grid
        ctx.fovy = fovy
        ctx.fovx = fovx

        return tiles_touched, rect_tile_space, vs_tetra, circumcenter, tet_area
    
    @staticmethod
    def backward(ctx, grad_tiles_touched, grad_rect_tile_space, grad_vs_tetra, grad_circumcenter):
        (indices, vertices, world_view_transform, K, cam_pos,
         tiles_touched, rect_tile_space, vs_tetra, circumcenter, tet_area) = ctx.saved_tensors
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
                                                      out_tet_area=tet_area,
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
        return grad_indices, grad_vertices, None, None, None, None
