import torch
from delaunay_rasterization.internal.render_grid import RenderGrid
import delaunay_rasterization.internal.slang.slang_modules as slang_modules
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from icecream import ic
import time
import math
from delaunay_rasterization.internal.util import recombine_tensors, split_tensors

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def render_constant_color(indices, vertices,
                                       rgbs_fn,
                                       camera,
                                       cell_values=None, tile_size=16, min_t=0.1):
    device = indices.device
    
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
    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _, tet_area = vertex_and_tile_shader(
        indices,
        vertices,
        tcam,
        render_grid)
   
    # torch.cuda.synchronize()
    # ic("vt", time.time()-st)
    # retain_grad fails if called with torch.no_grad() under evaluation
    try:
        vs_tetra.retain_grad()
    except:
        pass
    # ic(circumcenter)
    # torch.cuda.synchronize()
    # dt1 = (time.time() - st)
    if cell_values is None:
        rgbs = torch.zeros((mask.shape[0], 4), device=circumcenter.device)
        rgbs[mask] = rgbs_fn(circumcenter[mask])
    else:
        rgbs = cell_values
    rgbs = cell_values

    # torch.cuda.synchronize()
    # st = time.time()
    # tet_vertices = vertices[indices]
    image_rgb, distortion_img = AlphaBlendTiledRender.apply(
        sorted_tetra_idx,
        tile_ranges,
        indices,
        vertices,
        rgbs,
        render_grid,
        tcam,
        100,
        -0.1)
    # torch.cuda.synchronize()
    
    render_pkg = {
        'render': image_rgb.permute(2,0,1)[:3, ...],
        'viewspace_points': vs_tetra,
        'visibility_filter': mask,
        'circumcenters': circumcenter,
        'rgbs': rgbs,
        'mask': mask,
        'tet_area': tet_area,
    }

    return render_pkg


class AlphaBlendTiledRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                sorted_tetra_idx, tile_ranges,
                indices, vertices, rgbs, render_grid,
                tcam, pre_multi, ladder_p, 
                device="cuda"):
        distortion_img = torch.zeros((render_grid.image_height, 
                                  render_grid.image_width, 4), 
                                 device=device)
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
        st = time.time()
        splat_kernel_with_args = alpha_blend_tile_shader.splat_tiled(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=vertices,
            rgbs=rgbs,
            output_img=output_img,
            distortion_img=distortion_img,
            n_contributors=n_contributors,
            tcam=tcam,
            pre_multi=pre_multi,
            ladder_p=ladder_p,
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
            output_img, distortion_img, n_contributors]
        non_tensor_data, tensor_data = split_tensors(tcam)
        ctx.save_for_backward(*tensors, *tensor_data)
        ctx.non_tensor_data = non_tensor_data
        ctx.len_tensors = len(tensors)

        ctx.render_grid = render_grid
        ctx.ladder_p = ladder_p
        ctx.pre_multi = pre_multi

        return output_img, distortion_img

    @staticmethod
    def backward(ctx, grad_output_img, grad_distortion_img):
        (sorted_tetra_idx, tile_ranges,
            indices, vertices, rgbs, 
            output_img, distortion_img, n_contributors) = ctx.saved_tensors[:ctx.len_tensors]
        tcam = recombine_tensors(ctx.non_tensor_data, ctx.saved_tensors[ctx.len_tensors:])
        render_grid = ctx.render_grid
        ladder_p = ctx.ladder_p
        pre_multi = ctx.pre_multi

        vertices_grad = torch.zeros_like(vertices)
        rgbs_grad = torch.zeros_like(rgbs)

        assert (render_grid.tile_height, render_grid.tile_width) in slang_modules.alpha_blend_shaders, (
            'Alpha Blend Shader was not compiled for this tile'
            f' {render_grid.tile_height}x{render_grid.tile_width} configuration, available configurations:'
            f' {slang_modules.alpha_blend_shaders.keys()}'
        )

        alpha_blend_tile_shader = slang_modules.alpha_blend_shaders[(render_grid.tile_height, render_grid.tile_width)]

        st = time.time()
        kernel_with_args = alpha_blend_tile_shader.splat_tiled.bwd(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=(vertices, vertices_grad),
            rgbs=(rgbs, rgbs_grad),
            output_img=(output_img, grad_output_img),
            distortion_img=(distortion_img, grad_distortion_img),
            n_contributors=n_contributors,
            tcam=tcam,
            pre_multi=pre_multi,
            ladder_p=ladder_p)
        # ic(rgbs, rgbs_grad)
        
        kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )
        # torch.cuda.synchronize()
        # ic("abb", time.time()-st)
        
        return (None, None, None, vertices_grad, rgbs_grad, 
                None, None, None, None)
