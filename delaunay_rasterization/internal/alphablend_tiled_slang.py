import torch
from delaunay_rasterization.internal.render_grid import RenderGrid
import delaunay_rasterization.internal.slang.slang_modules as slang_modules
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from icecream import ic
import time

def render_alpha_blend_tiles_slang_raw(indices, vertices,
                                       rgbs_fn,
                                       world_view_transform, K, cam_pos,
                                       fovy, fovx, height, width, cell_values=None, tile_size=16):
    torch.cuda.synchronize()
    st = time.time()
    
    render_grid = RenderGrid(height,
                             width,
                             tile_height=tile_size,
                             tile_width=tile_size)
    sorted_tetra_idx, tile_ranges, radii, vs_tetra, circumcenter, mask = vertex_and_tile_shader(indices,
                                                                                 vertices,
                                                                                 world_view_transform,
                                                                                 K,
                                                                                 cam_pos,
                                                                                 fovy,
                                                                                 fovx,
                                                                                 render_grid)
   
    # print(vs_tetra)
    # retain_grad fails if called with torch.no_grad() under evaluation
    try:
        vs_tetra.retain_grad()
    except:
        pass
    # ic(circumcenter)
    # torch.cuda.synchronize()
    # dt1 = (time.time() - st)
    if cell_values is None:
        rgbs = torch.zeros((circumcenter.shape[0], 4), device=circumcenter.device)
        rgbs[mask] = rgbs_fn(circumcenter[mask])
    else:
        rgbs = cell_values

    # torch.cuda.synchronize()
    # st = time.time()
    image_rgb = AlphaBlendTiledRender.apply(
        sorted_tetra_idx,
        tile_ranges,
        indices,
        vertices,
        rgbs,
        render_grid,
        world_view_transform,
        K,
        cam_pos,
        fovy,
        fovx)
    # torch.cuda.synchronize()
    # dt2 = (time.time() - st)
    # print(dt1, dt2, 1/(dt1+dt2))
    
    render_pkg = {
        'render': image_rgb.permute(2,0,1)[:3, ...],
        'viewspace_points': vs_tetra,
        'visibility_filter': radii > 0,
        'radii': radii,
        'circumcenters': circumcenter,
        'rgbs': rgbs,
    }

    return render_pkg


class AlphaBlendTiledRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                sorted_tetra_idx, tile_ranges,
                indices, vertices, rgbs, render_grid,
                world_view_transform, K, cam_pos,
                fovy, fovx, device="cuda"):
        output_img = torch.zeros((render_grid.image_height, 
                                  render_grid.image_width, 4), 
                                 device=device)
        n_contributors = torch.zeros((render_grid.image_height, 
                                      render_grid.image_width, 1),
                                     dtype=torch.int32, device=device)

        assert (render_grid.tile_height, render_grid.tile_width) in slang_modules.alpha_blend_shaders, (
            'Alpha Blend Shader was not compiled for this tile'
            f' {render_grid.tile_height}x{render_grid.tile_width} configuration, available configurations:'
            f' {slang_modules.alpha_blend_shaders.keys()}'
        )

        alpha_blend_tile_shader = slang_modules.alpha_blend_shaders[(render_grid.tile_height, render_grid.tile_width)]
        splat_kernel_with_args = alpha_blend_tile_shader.splat_tiled(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=vertices,
            rgbs=rgbs,
            output_img=output_img,
            n_contributors=n_contributors,
            image_height=render_grid.image_height,
            image_width=render_grid.image_width,
            grid_height=render_grid.grid_height,
            grid_width=render_grid.grid_width,
            world_view_transform=world_view_transform,
            K=K,
            cam_pos=cam_pos,
            fovy=fovy,
            fovx=fovx,
            tile_height=render_grid.tile_height,
            tile_width=render_grid.tile_width
        )
        splat_kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )

        ctx.save_for_backward(sorted_tetra_idx, tile_ranges,
                              indices, vertices, rgbs, 
                              output_img, n_contributors,
                              world_view_transform, K, cam_pos)
        ctx.render_grid = render_grid
        ctx.fovy = fovy
        ctx.fovx = fovx

        return output_img

    @staticmethod
    def backward(ctx, grad_output_img):
        (sorted_tetra_idx, tile_ranges, 
         indices, vertices, rgbs, 
         output_img, n_contributors,
         world_view_transform, K, cam_pos) = ctx.saved_tensors
        render_grid = ctx.render_grid
        fovy = ctx.fovy
        fovx = ctx.fovx

        indices_grad = torch.zeros_like(indices)
        vertices_grad = torch.zeros_like(vertices)
        rgbs_grad = torch.zeros_like(rgbs)

        assert (render_grid.tile_height, render_grid.tile_width) in slang_modules.alpha_blend_shaders, (
            'Alpha Blend Shader was not compiled for this tile'
            f' {render_grid.tile_height}x{render_grid.tile_width} configuration, available configurations:'
            f' {slang_modules.alpha_blend_shaders.keys()}'
        )

        alpha_blend_tile_shader = slang_modules.alpha_blend_shaders[(render_grid.tile_height, render_grid.tile_width)]

        kernel_with_args = alpha_blend_tile_shader.splat_tiled.bwd(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=(vertices, vertices_grad),
            rgbs=(rgbs, rgbs_grad),
            output_img=(output_img, grad_output_img),
            n_contributors=n_contributors,
            grid_height=render_grid.grid_height,
            grid_width=render_grid.grid_width,
            image_height=render_grid.image_height,
            image_width=render_grid.image_width,
            world_view_transform=world_view_transform,
            K=K,
            cam_pos=cam_pos,
            fovy=fovy,
            fovx=fovx,
            tile_height=render_grid.tile_height,
            tile_width=render_grid.tile_width)
        
        kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )
        
        return (None, None, indices_grad, vertices_grad, rgbs_grad, 
                None, None, None, None, None, None, None)
