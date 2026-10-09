import torch

from delaunay_rasterization.internal.slang.slang_modules import shader_manager


# cell_values columns expected by the shader: density(1) + base_rgb_v0(3) +
# color_grad(3) + aux(AUX_DIM). Anything past the first 7 columns is treated
# as auxiliary channels and forwarded into aux_img with alpha-weighted blend.
_CELL_BASE = 7
# aux_img columns: normal(3) + depth(1) + entropy(4) + depth2(1) + aux(AUX_DIM).
# depth and depth2 are alpha-premultiplied front-to-back composites of E[z]
# and E[z^2] along the ray, used by the volumetric VSM Chebyshev shadow path.
_AUX_BASE = 9
_AUX_DEPTH_COL = 3
_AUX_DEPTH2_COL = 8


def _ceil_div(a, b):
    return (a + b - 1) // b


def get_degenerate_tet_mask(vertices: torch.Tensor,
                            indices: torch.Tensor,
                            eps: float = 1e-12) -> torch.Tensor:
    """Boolean [T] mask flagging tets whose signed volume is below ``eps``."""
    v = vertices[indices]
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    e3 = v[:, 3] - v[:, 0]
    vol6 = (torch.cross(e1, e2, dim=-1) * e3).sum(dim=-1)
    return vol6.abs() < (6.0 * eps)


def compute_thin_tet_mask(vertices: torch.Tensor,
                          indices: torch.Tensor,
                          scene_scale: float,
                          altitude_ratio: float = 1e-3) -> torch.Tensor:
    """Boolean [T] mask flagging slivers by minimum altitude.

    Min altitude = 3·V / max_face_area. Captures tets where one or more
    faces is grazingly thin -- the case where the adjacency walker's
    4-face plane intersection becomes ambiguous and latches a bogus exit.
    Flagged tets get routed through ``eval_thin_tet`` in the kernel
    (single-plane intersection on the face most-aligned-with-ray).
    """
    v = vertices[indices.long()]                       # [T, 4, 3]
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    e3 = v[:, 3] - v[:, 0]
    vol = (torch.cross(e1, e2, dim=-1) * e3).sum(-1).abs() / 6.0  # [T]

    # Face j is opposite vertex j.
    def _area(a, b, c):
        return 0.5 * torch.norm(torch.cross(b - a, c - a, dim=-1), dim=-1)
    a0 = _area(v[:, 1], v[:, 2], v[:, 3])
    a1 = _area(v[:, 0], v[:, 2], v[:, 3])
    a2 = _area(v[:, 0], v[:, 1], v[:, 3])
    a3 = _area(v[:, 0], v[:, 1], v[:, 2])
    max_face = torch.maximum(torch.maximum(a0, a1), torch.maximum(a2, a3))
    min_alt = 3.0 * vol / max_face.clamp_min(1e-30)
    return min_alt < (float(altitude_ratio) * float(scene_scale))


class TetrahedralRayTrace(torch.autograd.Function):
    """Adjacency-walking ray tracer over a Delaunay tet mesh.

    Mirrors the WGSL ``raytrace_compute`` algorithm but is purely walk-based:
    the camera-containing tet is provided externally (via dtlookup) so there
    is no BVH. Forward only for now -- backward raises NotImplementedError.

    Returns:
        output_img: [N, 4] -- (r, g, b, alpha)
        aux_img:    [N, 9 + AUX_DIM] --
            (normal.xyz, depth, entropy[4], depth2, aux...)
            where depth = sum_segments T_j * (w0*t_exit + w1*t_enter) and
            depth2 = sum_segments T_j * (w0*t_exit^2 + w1*t_enter^2). With
            output_img[:, 3] = 1 - T_final = total alpha, these supply
            volumetric (E[z], E[z^2], alpha) for a Chebyshev shadow bound.
    """

    @staticmethod
    def forward(ctx,
                rays: torch.Tensor,            # [N, 6]
                indices: torch.Tensor,         # [T, 4] int32
                vertices: torch.Tensor,        # [V, 3]
                cell_values: torch.Tensor,     # [T, 7+AUX]
                tet_adj: torch.Tensor,         # [T, 4] int32
                start_tet_ids: torch.Tensor,   # [N] int32
                min_t: float,
                max_iters: int,
                t_far=1e30,                    # float OR [N] tensor
                return_n_steps: bool = False,
                return_t_saturate: bool = False,
                saturate_log_t: float = -3.0,  # default ⇒ alpha=0.95
                thin_mask: "torch.Tensor | None" = None):  # [T] int32 ⇒ skip 4-face intersection
        assert rays.is_cuda and rays.shape[1] == 6, \
            f"rays must be [N, 6] cuda tensor, got {tuple(rays.shape)}"
        assert cell_values.shape[0] == indices.shape[0], \
            "cell_values must have one row per tet (including empties)"
        assert cell_values.shape[1] >= _CELL_BASE, \
            f"cell_values needs at least {_CELL_BASE} channels"
        assert tet_adj.shape == indices.shape, \
            "tet_adj must match indices shape"
        assert start_tet_ids.shape[0] == rays.shape[0], \
            "one start_tet per ray"

        device = rays.device
        N = rays.shape[0]
        T = indices.shape[0]
        aux_dim = int(cell_values.shape[1] - _CELL_BASE)

        rays_c = rays.contiguous().float()
        indices_c = indices.contiguous().to(torch.int32)
        vertices_c = vertices.contiguous().float()
        cell_values_c = cell_values.contiguous().float()
        tet_adj_c = tet_adj.contiguous().to(torch.int32)
        start_tet_c = start_tet_ids.contiguous().to(torch.int32)

        if thin_mask is None:
            thin_mask_c = torch.zeros((T,), device=device, dtype=torch.int32)
        else:
            assert thin_mask.shape == (T,), \
                f"thin_mask must be shape ({T},), got {tuple(thin_mask.shape)}"
            thin_mask_c = thin_mask.contiguous().to(
                device=device, dtype=torch.int32)

        if torch.is_tensor(t_far):
            assert t_far.shape == (N,), \
                f"t_far tensor must be shape ({N},), got {tuple(t_far.shape)}"
            t_far_t = t_far.contiguous().to(device=device, dtype=torch.float32)
        else:
            t_far_t = torch.full((N,), float(t_far),
                                 device=device, dtype=torch.float32)

        output_img = torch.zeros((N, 4), device=device, dtype=torch.float32)
        aux_img = torch.zeros((N, _AUX_BASE + aux_dim),
                              device=device, dtype=torch.float32)
        n_steps = torch.zeros((N,), device=device, dtype=torch.int32)
        t_saturate = torch.full((N,), -1.0, device=device, dtype=torch.float32)

        shader = shader_manager.get_raytrace(aux_dim)
        block = 64
        grid = _ceil_div(N, block)
        shader.ray_trace(
            rays=rays_c,
            indices=indices_c,
            vertices=vertices_c,
            cell_values=cell_values_c,
            tet_adj=tet_adj_c,
            start_tet=start_tet_c,
            t_far=t_far_t,
            thin_mask=thin_mask_c,
            output_img=output_img,
            aux_img=aux_img,
            n_steps=n_steps,
            t_saturate=t_saturate,
            min_t=float(min_t),
            max_iters=int(max_iters),
            saturate_log_t=float(saturate_log_t),
        ).launchRaw(blockSize=(block, 1, 1), gridSize=(grid, 1, 1))

        extras = []
        if return_n_steps:
            extras.append(n_steps)
        if return_t_saturate:
            extras.append(t_saturate)
        if extras:
            return (output_img, aux_img, *extras)
        return output_img, aux_img

    @staticmethod
    def backward(ctx, *grads):
        raise NotImplementedError(
            "TetrahedralRayTrace backward is not implemented yet; "
            "wrap the call in torch.no_grad() for now.")
