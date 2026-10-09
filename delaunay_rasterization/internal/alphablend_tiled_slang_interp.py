import os
import torch
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.slang.slang_modules import shader_manager
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from icecream import ic
import time
from delaunay_rasterization.internal.util import recombine_tensors, split_tensors


def _use_cuda_bwd(tcam):
    """CUDA backward is opt-in (USE_CUDA_BWD env or module flag) and only
    supports the tuned 4x4 tile (16-thread half-warp) that the warp reduction
    assumes. Falls back to Slang otherwise so the A/B path always works."""
    if not (os.environ.get("USE_CUDA_BWD", "0") == "1" or _CUDA_BWD_FLAG[0]):
        return False
    return int(tcam["tile_height"]) * int(tcam["tile_width"]) == 16


_CUDA_BWD_FLAG = [False]  # in-process override (tests flip this)

# One-shot slot for the per-pixel distortion anchor (H, W, 3): (t_hat, tau,
# gamma) in ray-t units. Set via set_dist_anchor() immediately before
# AlphaBlendTiledRender.apply; consumed by the next forward. Kept out of
# .apply's argument list on purpose: the anchor is non-differentiable, and
# routing it through autograd would force every caller to change arity.
_DIST_ANCHOR = [None]


def set_dist_anchor(anchor):
    _DIST_ANCHOR[0] = anchor


def _cuda_backward(sorted_tetra_idx, tile_ranges, indices, vertices, vertex_normals,
                   cell_values, output_img, xyzd_img, distortion_img, n_contributors,
                   ray_jitter, grad_output_img, grad_xyzd_img, grad_distortion_img,
                   tcam, render_grid):
    from delaunay_rasterization.internal.cuda import alphablend_bwd_cuda as _cu
    vertices_grad = torch.zeros_like(vertices)
    vertex_normals_grad = torch.zeros_like(vertex_normals)
    cell_values_grad = torch.zeros_like(cell_values)
    _cu.alphablend_backward(
        sorted_tetra_idx.int().contiguous(),
        tile_ranges.int().contiguous(),
        indices.int().contiguous(),
        vertices.contiguous(), vertex_normals.contiguous(), cell_values.contiguous(),
        output_img.contiguous(), xyzd_img.contiguous(), distortion_img.contiguous(),
        n_contributors.int().contiguous(), ray_jitter.contiguous(),
        grad_output_img.contiguous(), grad_xyzd_img.contiguous(), grad_distortion_img.contiguous(),
        vertices_grad, vertex_normals_grad, cell_values_grad,
        tcam["world_view_transform"].float().contiguous(),
        tcam["K"].float().contiguous(),
        tcam["cam_pos"].float().contiguous(),
        tcam["distortion_params"].float().contiguous(),
        int(tcam["camera_type"]), float(tcam["min_t"]),
        int(render_grid.image_height), int(render_grid.image_width),
        int(render_grid.grid_width), int(render_grid.grid_height),
        int(tcam["tile_width"]), int(tcam["tile_height"]),
    )
    return vertices_grad, vertex_normals_grad, cell_values_grad

class AlphaBlendTiledRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                sorted_tetra_idx, tile_ranges,
                indices, vertices, vertex_normals, cell_values, render_grid,
                tcam, ray_jitter, aux_dim, device="cuda", shared_ray_flash=False
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distortion_img = torch.zeros((render_grid.image_height,
                                  render_grid.image_width, 5),
                                 device=device)
        output_img = torch.zeros((render_grid.image_height,
                                  render_grid.image_width, 4 + aux_dim),
                                 device=device)
        xyzd_img = torch.zeros((render_grid.image_height,
                                render_grid.image_width, 4),
                                 device=device)
        # Forward-only median-depth output: depth at which transmittance T
        # crosses 0.5 along the ray. Written from inside alpha_blend as a
        # non-Diff TensorView (no gradient flows through it).
        median_depth_img = torch.zeros((render_grid.image_height,
                                        render_grid.image_width, 1),
                                       device=device)
        n_contributors = torch.zeros((render_grid.image_height,
                                      render_grid.image_width, 1),
                                     dtype=torch.int32, device=device)

        tet_alive = torch.zeros((indices.shape[0]), dtype=bool, device=device)

        dist_anchor = _DIST_ANCHOR[0]
        _DIST_ANCHOR[0] = None  # one-shot
        anchor_active = dist_anchor is not None
        if dist_anchor is None:
            dist_anchor = torch.zeros((render_grid.image_height,
                                       render_grid.image_width, 3),
                                      device=device)
        assert dist_anchor.shape == (render_grid.image_height,
                                     render_grid.image_width, 3), dist_anchor.shape
        dist_anchor = dist_anchor.detach().float().contiguous()

        assert(len(ray_jitter.shape) == 3)
        assert(ray_jitter.shape[0] == render_grid.image_height)
        assert(ray_jitter.shape[1] == render_grid.image_width)
        assert(ray_jitter.shape[2] == 2)

        # Immutable selectors belong to this autograd invocation, including backward.
        if shared_ray_flash:
            selectors = shared_ray_flash if isinstance(shared_ray_flash, tuple) else (0, 0)
            alpha_blend_tile_shader = shader_manager.get_shared_ray_flash(
                render_grid.tile_height, render_grid.tile_width, aux_dim, *selectors)
        else:
            alpha_blend_tile_shader = shader_manager.get_interp(
                render_grid.tile_height, render_grid.tile_width, aux_dim)
        ctx.shader = alpha_blend_tile_shader
        ctx.shared_ray_flash = shared_ray_flash
        st = time.time()
        splat_kernel_with_args = alpha_blend_tile_shader.splat_tiled(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=vertices,
            vertex_normals=vertex_normals,
            cell_values=cell_values,
            output_img=output_img,
            xyzd_img=xyzd_img,
            distortion_img=distortion_img,
            n_contributors=n_contributors,
            tet_alive=tet_alive,
            median_depth_img=median_depth_img,
            dist_anchor=dist_anchor,
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
            indices, vertices, vertex_normals, cell_values,
            output_img, xyzd_img, distortion_img, n_contributors,
            ray_jitter, median_depth_img, dist_anchor
        ]
        non_tensor_data, tensor_data = split_tensors(tcam)
        ctx.save_for_backward(*tensors, *tensor_data)
        ctx.non_tensor_data = non_tensor_data
        ctx.len_tensors = len(tensors)
        ctx.aux_dim = aux_dim
        ctx.anchor_active = anchor_active

        ctx.render_grid = render_grid

        return output_img, xyzd_img, distortion_img, tet_alive, median_depth_img

    @staticmethod
    def backward(ctx, grad_output_img, grad_xyzd_img, grad_distortion_img,
                 grad_vert_alive, grad_median_depth_img):
        (sorted_tetra_idx, tile_ranges,
         indices, vertices, vertex_normals, cell_values,
         output_img, xyzd_img, distortion_img, n_contributors,
            ray_jitter, median_depth_img, dist_anchor) = ctx.saved_tensors[:ctx.len_tensors]
        tcam = recombine_tensors(ctx.non_tensor_data, ctx.saved_tensors[ctx.len_tensors:])
        render_grid = ctx.render_grid

        if _use_cuda_bwd(tcam) and not ctx.anchor_active and not ctx.shared_ray_flash:
            vertices_grad, vertex_normals_grad, cell_values_grad = _cuda_backward(
                sorted_tetra_idx, tile_ranges, indices, vertices, vertex_normals,
                cell_values, output_img, xyzd_img, distortion_img, n_contributors,
                ray_jitter, grad_output_img, grad_xyzd_img, grad_distortion_img,
                tcam, render_grid)
            return (None, None, None, vertices_grad, vertex_normals_grad, cell_values_grad,
                    None, None, None, None, None, None)[:len(ctx.needs_input_grad)]

        vertices_grad = torch.zeros_like(vertices)
        vertex_normals_grad = torch.zeros_like(vertex_normals)
        cell_values_grad = torch.zeros_like(cell_values)

        alpha_blend_tile_shader = ctx.shader

        tet_alive = torch.zeros((indices.shape[0]), dtype=bool, device=vertices.device)
        st = time.time()
        kernel_with_args = alpha_blend_tile_shader.splat_tiled.bwd(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=(vertices, vertices_grad),
            vertex_normals=(vertex_normals, vertex_normals_grad),
            tcam=tcam,
            cell_values=(cell_values, cell_values_grad),
            output_img=(output_img, grad_output_img),
            xyzd_img=(xyzd_img, grad_xyzd_img),
            distortion_img=(distortion_img, grad_distortion_img),
            n_contributors=n_contributors,
            tet_alive=tet_alive,
            median_depth_img=median_depth_img,
            dist_anchor=dist_anchor,
            ray_jitter=ray_jitter,
        )

        kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width,
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width,
                      render_grid.grid_height, 1)
        )

        return (None, None, None, vertices_grad, vertex_normals_grad, cell_values_grad,
                None, None, None, None, None, None)[:len(ctx.needs_input_grad)]
