import torch
from delaunay_rasterization.internal.render_grid import RenderGrid
import delaunay_rasterization.internal.slang.slang_modules as slang_modules
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from icecream import ic
import time
from utils.train_util import fov2focal
from data.camera import Camera
from utils.ssim import ssim

def render_err(gt_image, camera: Camera, model, tile_size=16, min_t=0.1, lambda_ssim=0.2, **kwargs):
    device = model.device
    indices = model.indices
    vertices = model.vertices
    fy = fov2focal(camera.fovy, camera.image_height)
    fx = fov2focal(camera.fovx, camera.image_width)
    K = torch.tensor([
    [fx, 0, camera.image_width/2],
    [0, fy, camera.image_height/2],
    [0, 0, 1],
    ]).to(device)
    world_view_transform = camera.world_view_transform.T.to(device)

    cam_pos = camera.world_view_transform.T.inverse()[:3, 3].to(device)
    fovy = camera.fovy
    fovx = camera.fovx
    torch.cuda.synchronize()
    assert(indices.device == vertices.device)
    assert(indices.device == world_view_transform.device)
    assert(indices.device == K.device)
    assert(indices.device == cam_pos.device)
    st = time.time()
    device = indices.device
    pre_multi=500
    ladder_p=-0.1 
    render_grid = RenderGrid(camera.image_height,
                             camera.image_width,
                             tile_height=tile_size,
                             tile_width=tile_size)
    st = time.time()
    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _, tet_area = vertex_and_tile_shader(
        indices,
        vertices,
        world_view_transform,
        K,
        cam_pos,
        fovy,
        fovx,
        render_grid)
   
    # torch.cuda.synchronize()
    # ic("vt", time.time()-st)
    # retain_grad fails if called with torch.no_grad() under evaluation
    try:
        vs_tetra.retain_grad()
    except:
        pass
    # cell_values = torch.zeros((mask.shape[0], 4), device=circumcenter.device)
    # cell_values[mask] = model.get_cell_values(camera, mask)
    vertex_color, cell_values = model.get_cell_values(camera)

    # torch.cuda.synchronize()
    # st = time.time()
    tet_vertices = vertices[indices]
    distortion_img = torch.zeros((render_grid.image_height, 
                                render_grid.image_width, 4), 
                                device=device)
    output_img = torch.zeros((render_grid.image_height, 
                                render_grid.image_width, 4), 
                                device=device)
    n_contributors = torch.zeros((render_grid.image_height, 
                                    render_grid.image_width, 1),
                                    dtype=torch.int32, device=device)
    tet_err = torch.zeros((tet_vertices.shape[0]), dtype=torch.float, device=device)
    tet_count = torch.zeros((tet_vertices.shape[0]), dtype=torch.int32, device=device)

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
        vertex_color=vertex_color,
        tet_density=cell_values,
        output_img=output_img,
        distortion_img=distortion_img,
        n_contributors=n_contributors,
        image_height=render_grid.image_height,
        image_width=render_grid.image_width,
        grid_height=render_grid.grid_height,
        grid_width=render_grid.grid_width,
        world_view_transform=world_view_transform,
        K=K,
        cam_pos=cam_pos,
        pre_multi=pre_multi,
        ladder_p=ladder_p,
        min_t=min_t,
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
    torch.cuda.synchronize()
    alpha = 1-output_img.permute(2,0,1)[3, ...]
    render_img = output_img.permute(2,0,1)[:3, ...].clip(min=0, max=1)
    l2_err = ((render_img - gt_image)**2).mean(dim=0)
    ssim_err = 1-ssim(render_img, gt_image).mean(dim=0)
    pixel_err = ((1-lambda_ssim) * l2_err + lambda_ssim * ssim_err).contiguous()
    assert(pixel_err.shape[0] == render_grid.image_height)
    assert(pixel_err.shape[1] == render_grid.image_width)
    alpha_blend_tile_shader.calc_tet_err(
        sorted_gauss_idx=sorted_tetra_idx,
        tile_ranges=tile_ranges,
        indices=indices,
        vertices=vertices,
        vertex_color=vertex_color,
        tet_density=cell_values,
        output_img=output_img,
        pixel_err=pixel_err,
        tet_err=tet_err,
        tet_count=tet_count,
        n_contributors=n_contributors,
        image_height=render_grid.image_height,
        image_width=render_grid.image_width,
        grid_height=render_grid.grid_height,
        grid_width=render_grid.grid_width,
        world_view_transform=world_view_transform,
        K=K,
        cam_pos=cam_pos,
        min_t=min_t,
        fovy=fovy,
        fovx=fovx,
        tile_height=render_grid.tile_height,
        tile_width=render_grid.tile_width
    ).launchRaw(
        blockSize=(render_grid.tile_width, 
                    render_grid.tile_height, 1),
        gridSize=(render_grid.grid_width, 
                    render_grid.grid_height, 1)
    )
    torch.cuda.synchronize()
    tet_err = tet_err.clip(max=pixel_err.max())
    # ic((tet_area > 2).float().mean(), tet_area.mean())
    mask = tet_area > 1
    # tet_err = torch.where(mask, tet_err, 0)

    return tet_err, dict(
        mask = mask,
        alpha = alpha,
        tet_area = tet_area,
        tet_count = tet_count,
        pixel_err = pixel_err,
        ssim_err = ssim_err,
        l2_err = l2_err,
        render_img = render_img
    )
