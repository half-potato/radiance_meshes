import torch
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.slang.slang_modules import shader_manager
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from icecream import ic
import time
from delaunay_rasterization.internal.util import recombine_tensors, split_tensors
from utils.safe_math import safe_div


def render(camera, model, face_values=None, tile_size=4, min_t=0.1,
           scene_scaling=1, clip_multi=0, ray_jitter=None,
           **kwargs):
    device = model.device
    if ray_jitter is None:
        ray_jitter = 0.5*torch.ones((camera.image_height, camera.image_width, 2), device=device)
    else:
        assert(ray_jitter.shape[0] == camera.image_height)
        assert(ray_jitter.shape[1] == camera.image_width)
        assert(ray_jitter.shape[2] == 2)
    vertices = model.vertices
    
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
    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _ = vertex_and_tile_shader(
        model.indices,
        vertices,
        tcam,
        render_grid)
    extras = {}
    sh_reg = 0
    if face_values is None:
        face_values = model.get_face_values(camera)
        # face_values = torch.zeros((mask.shape[0], model.feature_dim), device=circumcenter.device)
        # if mask.sum() > 0 and model.mask_values:
        #     values = model.get_face_values(camera, mask)
        #     face_values[mask] = values
        # else:
        #     face_values = model.get_face_values(camera)

    image_rgb, xyzd_img, distortion_img, tet_alive = AlphaBlendTiledRender.apply(
        sorted_tetra_idx,
        tile_ranges,
        model.indices,
        model.tet_face_ids,
        vertices,
        face_values,
        render_grid,
        tcam,
        ray_jitter,
        model.additional_attr)
    alpha = 1-image_rgb.permute(2,0,1)[3, ...].exp()
    total_density = (distortion_img[:, :, 2]**2).clip(min=1e-6)
    distortion_loss = safe_div(((distortion_img[:, :, 0] - distortion_img[:, :, 1]) + distortion_img[:, :, 4]), total_density).clip(min=0)

    # unrotate the xyz part of the xyzd_img
    rotated = xyzd_img[..., :3].reshape(-1, 3) @ camera.world_view_transform[:3, :3].to(device)
    rxyzd_img = torch.cat([rotated.reshape(xyzd_img[..., :3].shape), xyzd_img[..., 3:]], dim=-1)
    
    render_pkg = {
        'aux': image_rgb.permute(2,0,1)[4:, ...] * camera.gt_alpha_mask.to(device),
        'render': image_rgb.permute(2,0,1)[:3, ...] * camera.gt_alpha_mask.to(device),
        'alpha': alpha,
        'distortion_loss': distortion_loss.mean(),
        'mask': mask,
        'xyzd': rxyzd_img,
        'sh_reg': sh_reg,
        'weight_square': image_rgb.permute(2,0,1)[4:5, ...],
        "face_values": face_values,
        **extras
    }
    return render_pkg



class AlphaBlendTiledRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                sorted_tetra_idx, tile_ranges,
                indices, tet_face_ids, vertices, face_values, render_grid,
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

        alpha_blend_tile_shader = shader_manager.get_flux(render_grid.tile_height, render_grid.tile_width, aux_dim)
        st = time.time()
        splat_kernel_with_args = alpha_blend_tile_shader.splat_tiled(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            tet_face_ids=tet_face_ids,
            vertices=vertices,
            face_values=face_values,
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
            indices, tet_face_ids, vertices, face_values, 
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
         indices, tet_face_ids, vertices, face_values,
         output_img, xyzd_img, distortion_img, n_contributors,
            ray_jitter) = ctx.saved_tensors[:ctx.len_tensors]
        tcam = recombine_tensors(ctx.non_tensor_data, ctx.saved_tensors[ctx.len_tensors:])
        render_grid = ctx.render_grid

        vertices_grad = torch.zeros_like(vertices)
        face_values_grad = torch.zeros_like(face_values)

        alpha_blend_tile_shader = shader_manager.get_flux(render_grid.tile_height, render_grid.tile_width, ctx.aux_dim)

        tet_alive = torch.zeros((indices.shape[0]), dtype=bool, device=vertices.device)
        st = time.time()
        kernel_with_args = alpha_blend_tile_shader.splat_tiled.bwd(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            tet_face_ids=tet_face_ids,
            vertices=(vertices, vertices_grad),
            tcam=tcam,
            face_values=(face_values, face_values_grad),
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

        return (None, None, None, None, vertices_grad, face_values_grad, 
                None, None, None, None)
