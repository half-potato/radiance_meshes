import torch
from torch import nn
from typing import Optional, Tuple
import gc
import numpy as np
from pathlib import Path
import open3d as o3d

from gdel3d import Del
from scipy.spatial import Delaunay

from utils.topo_utils import (
    tet_volumes, calculate_circumcenters_torch,
    fibonacci_spiral_on_sphere, get_tet_adjacency,
)
from utils.model_util import activate_output, pre_calc_cell_values, offset_normalize, RGB2SH
from utils.safe_math import safe_exp
from utils.train_util import get_expon_lr_func, SpikingLR
from utils import optim
from utils.args import Args
from models.base_model import BaseModel


# ---------------------------------------------------------------------------
# Vertex-to-tet adjacency (from experiments/test_field_transfer.py)
# ---------------------------------------------------------------------------

def build_v2t(indices: torch.Tensor, n_verts: int):
    """Build padded vertex->tet adjacency table.

    Returns:
        v2t: (V, max_valence) int64, padded with -1
        valence: (V,) int64, actual count per vertex
    """
    T = indices.shape[0]
    device = indices.device
    flat_vidx = indices.long().reshape(-1)
    flat_tidx = torch.arange(T, device=device).repeat_interleave(4)

    sort_order = torch.argsort(flat_vidx)
    sorted_vidx = flat_vidx[sort_order]
    sorted_tidx = flat_tidx[sort_order]

    valence = torch.bincount(flat_vidx, minlength=n_verts)
    offsets = torch.zeros(n_verts + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(valence, dim=0)
    max_val = valence.max().item()

    group_starts = offsets[sorted_vidx]
    local_pos = torch.arange(len(sorted_vidx), device=device) - group_starts

    v2t = torch.full((n_verts, max_val), -1, dtype=torch.long, device=device)
    v2t[sorted_vidx, local_pos] = sorted_tidx.long()

    return v2t, valence


def _min_edge_length(vertices, indices):
    """Compute minimum edge length per tet. indices: (T, 4), returns (T,)."""
    p = vertices[indices.long()]  # (T, 4, 3)
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    el = torch.stack([(p[:, a] - p[:, b]).norm(dim=-1) for a, b in edges])
    return el.min(dim=0).values


def compute_transfer_weights(
    v2t: torch.Tensor,              # (V, max_val) padded with -1
    new_indices: torch.Tensor,       # (T_new, 4) int
    old_indices: torch.Tensor,       # (T_old, 4) int
    old_cc: torch.Tensor,            # (T_old, 3)
    new_cc: torch.Tensor,            # (T_new, 3)
    vertex_positions: torch.Tensor,  # (V, 3)
    vert_chunk: int = 50_000,
):
    """For each new tet, compute blending weights across 4 candidate old tets.

    Two-stage approach:
    1. Per-vertex: find the nearest old tet (by CC distance to the vertex)
    2. Per-new-tet: compute weights based on vertex overlap + inverse CC distance

    Returns:
        cands: (T_new, 4) long — candidate old tet indices
        weights: (T_new, 4) float — normalized blending weights
        density_scale: (T_new, 4) float — per-candidate edge-length ratio
    """
    V = vertex_positions.shape[0]
    device = new_indices.device

    # Stage 1: per-vertex best old tet
    vert_best = torch.zeros(V, dtype=torch.long, device=device)
    for start in range(0, V, vert_chunk):
        end = min(start + vert_chunk, V)
        v_pos = vertex_positions[start:end]
        v_cands = v2t[start:end]
        valid = v_cands >= 0
        safe = v_cands.clamp(min=0)
        dists = (old_cc[safe] - v_pos.unsqueeze(1)).pow(2).sum(-1)
        dists[~valid] = float("inf")
        best_local = dists.argmin(dim=1)
        vert_best[start:end] = safe.gather(1, best_local.unsqueeze(1)).squeeze(1)

    del v2t

    # Stage 2: candidates + weights
    cands = vert_best[new_indices.long()]               # (T_new, 4)

    # Vertex overlap
    cand_verts = old_indices[cands].long()               # (T_new, 4, 4)
    new_verts = new_indices.long().unsqueeze(1)           # (T_new, 1, 4)
    matches = (cand_verts.unsqueeze(-1) == new_verts.unsqueeze(2))  # (T_new, 4, 4, 4)
    overlap = matches.any(dim=-1).sum(dim=-1).float()    # (T_new, 4)

    # CC distance
    cand_cc = old_cc[cands]                             # (T_new, 4, 3)
    cc_dist_sq = (cand_cc - new_cc.unsqueeze(1)).pow(2).sum(-1)  # (T_new, 4)

    # Weight: overlap bonus * inverse distance kernel
    # Tets with more shared vertices get exponentially more weight
    overlap_weight = torch.exp(overlap * 2.0)  # e^0=1, e^2≈7, e^4≈55, e^6≈403, e^8≈2981
    dist_weight = 1.0 / (cc_dist_sq + 1e-8)
    raw_weights = overlap_weight * dist_weight
    weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)  # (T_new, 4) normalized

    # Per-candidate density scale (edge-length ratio)
    new_el = _min_edge_length(vertex_positions, new_indices)  # (T_new,)
    old_el = torch.stack([
        _min_edge_length(vertex_positions, old_indices[cands[:, i]]) for i in range(4)
    ], dim=1)  # (T_new, 4)
    density_scale = (old_el / new_el.unsqueeze(1).clamp(min=1e-8)).clamp(min=0.1, max=10.0)

    return cands, weights, density_scale


# ===========================================================================
# SimpleModel
# ===========================================================================

class SimpleModel(BaseModel):
    """Per-tet parameter model with adjacency-based attribute transfer.

    Like FrozenTetModel but with a working update_triangulation() that
    transfers attributes through retriangulation using adj_grad.
    """

    def __init__(
        self,
        int_vertices: torch.Tensor,
        ext_vertices: torch.Tensor,
        indices: torch.Tensor,
        density: torch.Tensor,
        rgb: torch.Tensor,
        gradient: torch.Tensor,
        sh: torch.Tensor,
        center: torch.Tensor,
        scene_scaling: torch.Tensor | float,
        *,
        max_sh_deg: int = 2,
        chunk_size: int = 408_576,
        density_offset: float = -3,
        **kwargs,
    ) -> None:
        super().__init__()

        # Geometry
        self.interior_vertices = nn.Parameter(int_vertices.cuda(), requires_grad=True)
        self.register_buffer("ext_vertices", ext_vertices.cuda())
        self.register_buffer("indices", indices.int())
        self.empty_indices = torch.empty((0, 4), dtype=indices.dtype, device='cuda')
        self.register_buffer("center", center.reshape(1, 3))
        self.register_buffer("scene_scaling", torch.as_tensor(scene_scaling))

        # Per-tet learnable parameters
        self.density = nn.Parameter(density, requires_grad=True)
        self.gradient = nn.Parameter(gradient, requires_grad=True)
        self.rgb = nn.Parameter(rgb, requires_grad=True)
        self.sh = nn.Parameter(sh.half(), requires_grad=True)

        # Config
        self.max_sh_deg = max_sh_deg
        self.current_sh_deg = max_sh_deg
        self.chunk_size = chunk_size
        self.device = self.density.device
        self.sh_dim = ((1 + max_sh_deg) ** 2 - 1) * 3
        self.density_offset = density_offset

        self.mask_values = False
        self.frozen = False
        self.linear = False
        self.feature_dim = 7
        self.additional_attr = 0

    def sh_up(self):
        self.current_sh_deg = min(self.max_sh_deg, self.current_sh_deg + 1)

    @property
    def vertices(self) -> torch.Tensor:
        return torch.cat([self.interior_vertices, self.ext_vertices], dim=0)

    @property
    def num_int_verts(self):
        return self.interior_vertices.shape[0]

    def compute_batch_features(
        self,
        vertices: torch.Tensor,
        indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        circumcenters: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if circumcenters is None:
            circumcenter = pre_calc_cell_values(vertices, indices)
        else:
            circumcenter = circumcenters

        if mask is not None:
            density = self.density[mask]
            grd = self.gradient[mask]
            rgb = self.rgb[mask]
            sh = self.sh[mask]
        else:
            density = self.density
            grd = self.gradient
            rgb = self.rgb
            sh = self.sh

        sh_dim = (self.max_sh_deg + 1) ** 2 - 1
        attr = torch.empty((density.shape[0], 0), device=grd.device)
        if sh_dim == 0:
            sh_out = torch.empty((density.shape[0], 0, 3), device=grd.device, dtype=sh.dtype)
        else:
            sh_out = sh.reshape(-1, sh_dim, 3)
        return circumcenter, density, rgb, grd, sh_out, attr

    def compute_features(self, offset=False):
        vertices = self.vertices
        indices = self.indices
        circumcenters, density, rgb, grd, sh, attr = self.compute_batch_features(
            vertices, indices
        )
        tets = vertices[indices]
        if offset:
            base_color_v0_raw, normed_grd = offset_normalize(
                rgb, grd, circumcenters, tets
            )
            return circumcenters, density, rgb, normed_grd, sh
        else:
            return circumcenters, density, rgb, grd, sh

    def get_cell_values(
        self,
        camera,
        mask: Optional[torch.Tensor] = None,
        all_circumcenters: Optional[torch.Tensor] = None,
        radii: Optional[torch.Tensor] = None,
    ):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices
        cc, density, rgb, grd, sh, attr = self.compute_batch_features(
            vertices, indices, mask, circumcenters=all_circumcenters
        )
        cell_output = activate_output(
            camera.camera_center.to(self.device),
            density, rgb, grd,
            sh,
            attr,
            indices,
            cc,
            vertices,
            self.current_sh_deg,
            self.max_sh_deg,
        )
        return sh, cell_output

    def compute_adjacency(self):
        vols = tet_volumes(self.vertices[self.indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            self.indices[reverse_mask] = self.indices[reverse_mask][:, [1, 0, 2, 3]]
        self.faces, self.side_index = get_tet_adjacency(self.indices)

    def calc_tet_density(self):
        _, densities, _, _, _, _ = self.compute_batch_features(
            self.vertices, self.indices
        )
        return densities.reshape(-1)

    @torch.no_grad()
    def update_triangulation(
        self,
        high_precision=False,
        density_threshold=0.0,
        alpha_threshold=0.0,
    ):
        """Recompute Delaunay triangulation only (geometry).

        Does NOT modify per-tet parameters — the optimizer handles that.

        Returns:
            (cands, weights, density_scale, new_indices, new_cc, old_cc) or None
        """
        torch.cuda.empty_cache()

        old_indices = self.indices.clone()
        verts = self.vertices
        old_n_verts = verts.shape[0]

        old_cc, _ = calculate_circumcenters_torch(
            verts[old_indices.long()].double()
        )
        old_cc = old_cc.float()

        # Build v2t adjacency on old mesh
        v2t, _ = build_v2t(old_indices, old_n_verts)

        # Retriangulate
        if high_precision:
            indices_np = Delaunay(verts.detach().cpu().numpy()).simplices.astype(
                np.int32
            )
        else:
            v = Del(verts.shape[0])
            indices_np, prev = v.compute(verts.detach().cpu().double())
            indices_np = indices_np.clone().numpy()
            valid_mask = (indices_np >= 0) & (indices_np < verts.shape[0])
            indices_np = indices_np[valid_mask.all(axis=1)]
            del prev

        # Ensure positive volumes
        new_indices = torch.as_tensor(indices_np).cuda()
        vols = tet_volumes(verts[new_indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            new_indices[reverse_mask] = new_indices[reverse_mask][:, [1, 0, 2, 3]]

        new_cc, _ = calculate_circumcenters_torch(verts[new_indices.long()].double())
        new_cc = new_cc.float()

        # Compute transfer weights across 4 candidates per new tet
        cands, weights, density_scale = compute_transfer_weights(
            v2t, new_indices, old_indices, old_cc, new_cc, verts)

        torch.cuda.empty_cache()
        return cands, weights, density_scale, new_indices, new_cc, old_cc, old_indices

    @staticmethod
    def init_from_pcd(
        point_cloud,
        cameras,
        device,
        max_sh_deg=2,
        voxel_size=0.00,
        density_offset=-3,
        **kwargs,
    ):
        torch.manual_seed(2)

        ccenters = torch.stack(
            [c.camera_center.reshape(3) for c in cameras], dim=0
        ).to(device)
        center = ccenters.mean(dim=0)
        scaling = torch.linalg.norm(
            ccenters - center.reshape(1, 3), dim=1, ord=torch.inf
        ).max()
        print(f"Scene scaling: {scaling}. Center: {center}")

        vertices = torch.as_tensor(point_cloud.points).float()

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(vertices.numpy())
        if voxel_size > 0:
            o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)

        vertices = torch.as_tensor(np.asarray(o3d_pcd.points)).float()
        vertices = vertices + torch.randn(*vertices.shape) * 1e-3

        pcd_scaling = torch.linalg.norm(
            vertices - center.cpu().reshape(1, 3), dim=1, ord=2
        ).max()
        new_radius = pcd_scaling.cpu().item()

        num_ext = 1000
        ext_vertices = (
            fibonacci_spiral_on_sphere(num_ext, new_radius, device="cpu")
            + center.reshape(1, 3).cpu()
        )

        # Concatenate exterior into interior (same as Model.init_from_pcd)
        vertices = torch.cat([vertices, ext_vertices], dim=0)
        ext_vertices = torch.empty((0, 3))

        int_vertices = vertices.to(device)
        ext_verts = ext_vertices.to(device)

        # Initial Delaunay
        all_verts = torch.cat([int_vertices, ext_verts], dim=0)
        v = Del(all_verts.shape[0])
        indices_np, prev = v.compute(all_verts.detach().cpu().double())
        indices_np = indices_np.clone().numpy()
        valid_mask = (indices_np >= 0) & (indices_np < all_verts.shape[0])
        indices_np = indices_np[valid_mask.all(axis=1)]
        del prev
        indices = torch.as_tensor(indices_np).to(device)
        vols = tet_volumes(all_verts[indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            indices[reverse_mask] = indices[reverse_mask][:, [1, 0, 2, 3]]

        T = indices.shape[0]
        sh_dim = ((1 + max_sh_deg) ** 2 - 1) * 3

        # Initialize per-tet parameters
        density = safe_exp(
            torch.full((T, 1), density_offset, device=device)
        )
        rgb = torch.full((T, 3), 0.5, device=device)
        gradient = torch.zeros((T, 1, 3), device=device)
        sh = torch.zeros((T, sh_dim // 3, 3), device=device)

        model = SimpleModel(
            int_vertices=int_vertices,
            ext_vertices=ext_verts,
            indices=indices,
            density=density,
            rgb=rgb,
            gradient=gradient,
            sh=sh,
            center=center,
            scene_scaling=scaling,
            max_sh_deg=max_sh_deg,
            density_offset=density_offset,
        )
        return model

    @staticmethod
    def load_ckpt(path: Path, device):
        ckpt_path = path / "ckpt.pth"
        config_path = path / "config.json"
        config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)

        int_vertices = ckpt["interior_vertices"]
        ext_vertices = ckpt["ext_vertices"]
        indices = ckpt["indices"]
        if "empty_indices" in ckpt:
            del ckpt["empty_indices"]

        density = ckpt["density"]
        rgb = ckpt["rgb"]
        gradient = ckpt["gradient"]
        sh = ckpt["sh"]
        center = ckpt["center"]
        scene_scaling = ckpt["scene_scaling"]

        model = SimpleModel(
            int_vertices=int_vertices.to(device),
            ext_vertices=ext_vertices.to(device),
            indices=indices.to(device),
            density=density.to(device),
            rgb=rgb.to(device),
            gradient=gradient.to(device),
            sh=sh.to(device),
            center=center.to(device),
            scene_scaling=scene_scaling.to(device),
            max_sh_deg=config.max_sh_deg,
        )
        model.load_state_dict(ckpt)
        model.min_t = config.min_t
        return model


# ===========================================================================
# SimpleOptimizer
# ===========================================================================

class SimpleOptimizer:
    """Optimizer for SimpleModel. Mirrors FrozenTetOptimizer but with
    working update_triangulation that remaps optimizer state."""

    def __init__(
        self,
        model: SimpleModel,
        *,
        freeze_lr: float = 1e-3,
        final_freeze_lr: float = 1e-4,
        lr_delay_multi=1e-8,
        lr_delay=0,
        vertices_lr: float = 4e-4,
        final_vertices_lr: float = 4e-7,
        vert_lr_delay: int = 500,
        vertices_lr_delay_multi: float = 0.01,
        freeze_start: int = 15000,
        iterations: int = 30000,
        spike_duration: int = 20,
        densify_interval: int = 500,
        densify_end: int = 15000,
        densify_start: int = 2000,
        split_std: float = 0.5,
        **kwargs,
    ) -> None:
        self.model = model
        self.split_std = split_std

        self.optim = optim.CustomAdam([
            {"params": [model.density], "lr": freeze_lr, "name": "density"},
            {"params": [model.rgb], "lr": freeze_lr, "name": "color"},
            {"params": [model.gradient], "lr": freeze_lr, "name": "gradient"},
        ])
        self.sh_optim = optim.CustomAdam([
            {"params": [model.sh], "lr": freeze_lr, "name": "sh"},
        ], eps=1e-4)
        self.vert_lr_multi = float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam([
            {
                "params": [model.interior_vertices],
                "lr": self.vert_lr_multi * vertices_lr,
                "name": "interior_vertices",
            },
        ])

        self.freeze_start = freeze_start
        self.scheduler = get_expon_lr_func(
            lr_init=freeze_lr,
            lr_final=final_freeze_lr,
            lr_delay_mult=lr_delay_multi,
            max_steps=iterations,
            lr_delay_steps=lr_delay,
        )

        self.vertex_lr = self.vert_lr_multi * vertices_lr
        base_vertex_scheduler = get_expon_lr_func(
            lr_init=self.vertex_lr,
            lr_final=self.vert_lr_multi * final_vertices_lr,
            lr_delay_mult=vertices_lr_delay_multi,
            max_steps=iterations,
            lr_delay_steps=vert_lr_delay,
        )
        self.vertex_scheduler_args = SpikingLR(
            spike_duration,
            freeze_start,
            base_vertex_scheduler,
            densify_start,
            densify_interval,
            densify_end,
            self.vertex_lr,
            self.vertex_lr,
        )

        # Alias for compatibility
        self.net_optim = self.optim

    # --- optimizer steps ---
    def step(self):
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()

    def main_step(self):
        self.step()

    def main_zero_grad(self):
        self.zero_grad()

    def update_learning_rate(self, iteration):
        for param_group in self.optim.param_groups:
            lr = self.scheduler(iteration)
            param_group["lr"] = lr

        for param_group in self.sh_optim.param_groups:
            lr = self.scheduler(iteration)
            param_group["lr"] = lr

        for param_group in self.vertex_optim.param_groups:
            if param_group["name"] == "interior_vertices":
                lr = self.vertex_scheduler_args(iteration)
                self.vertex_lr = lr
                param_group["lr"] = lr

    def regularizer(self, render_pkg, lambda_weight_decay=0, **kwargs):
        return 0.0

    def _rebuild_optim(self):
        """Rebuild per-tet optimizers from current model parameters.
        This fully resets Adam state (step counter + momentum)."""
        lr = self.optim.param_groups[0]["lr"]
        sh_lr = self.sh_optim.param_groups[0]["lr"]
        self.optim = optim.CustomAdam([
            {"params": [self.model.density], "lr": lr, "name": "density"},
            {"params": [self.model.rgb], "lr": lr, "name": "color"},
            {"params": [self.model.gradient], "lr": lr, "name": "gradient"},
        ])
        self.sh_optim = optim.CustomAdam([
            {"params": [self.model.sh], "lr": sh_lr, "name": "sh"},
        ], eps=1e-4)
        self.net_optim = self.optim

    @staticmethod
    def _save_adam_state(custom_adam):
        """Save Adam state (exp_avg, exp_avg_sq, step) for all param groups."""
        saved = {}
        for group in custom_adam.param_groups:
            s = custom_adam.optimizer.state.get(group['params'][0])
            if s and 'exp_avg' in s:
                saved[group['name']] = {
                    'exp_avg': s['exp_avg'].clone(),
                    'exp_avg_sq': s['exp_avg_sq'].clone(),
                    'step': s['step'].clone(),
                }
        return saved

    @staticmethod
    def _blend_old_values(old_tensor, cands, weights):
        """Weighted blend of old tensor values across 4 candidates.
        old_tensor: (T_old, ...), cands: (T_new, 4), weights: (T_new, 4)
        Returns: (T_new, ...)
        """
        # Gather candidates: (T_new, 4, ...)
        gathered = old_tensor[cands]
        # Expand weights to match tensor dims
        w = weights
        for _ in range(old_tensor.dim() - 1):
            w = w.unsqueeze(-1)
        return (gathered * w).sum(dim=1)

    @staticmethod
    def _restore_adam_state_blended(custom_adam, saved, cands, weights, cull_mask=None):
        """Restore Adam state using weighted blend across candidates."""
        for group in custom_adam.param_groups:
            name = group['name']
            if name not in saved:
                continue
            param = group['params'][0]
            old = saved[name]

            # Weighted blend of momentum across candidates
            w = weights
            ea = old['exp_avg']
            ea_sq = old['exp_avg_sq']

            # Expand weights for broadcasting
            w_exp = w
            for _ in range(ea.dim() - 1):
                w_exp = w_exp.unsqueeze(-1)

            new_exp_avg = (ea[cands] * w_exp).sum(dim=1)
            new_exp_avg_sq = (ea_sq[cands] * w_exp).sum(dim=1)

            if cull_mask is not None:
                new_exp_avg = new_exp_avg[cull_mask]
                new_exp_avg_sq = new_exp_avg_sq[cull_mask]

            custom_adam.optimizer.state[param] = {
                'step': old['step'].clone(),
                'exp_avg': new_exp_avg,
                'exp_avg_sq': new_exp_avg_sq,
            }

    # --- triangulation update with optimizer state transfer ---
    def update_triangulation(self, density_threshold=0.0, alpha_threshold=0.0, **kwargs):
        # Save Adam state before retriangulation
        old_optim_state = self._save_adam_state(self.optim)
        old_sh_state = self._save_adam_state(self.sh_optim)

        result = self.model.update_triangulation(
            density_threshold=density_threshold,
            alpha_threshold=alpha_threshold,
            **kwargs,
        )
        if result is None:
            return
        cands, weights, density_scale, new_indices, new_cc, old_cc, old_indices = result

        # Save old sums for conservation
        old_density_sum = self.model.density.data.sum()
        old_rgb_sum = self.model.rgb.data.sum(dim=0)  # per-channel (3,)

        # Weighted blend of parameter values across candidates
        # Scale density per-candidate by edge-length ratio before blending
        old_density = self.model.density.data
        scaled_density = old_density[cands] * density_scale.unsqueeze(-1)
        w_density = weights.unsqueeze(-1)
        new_density = (scaled_density * w_density).sum(dim=1)

        new_rgb = self._blend_old_values(self.model.rgb.data, cands, weights)
        new_gradient = self._blend_old_values(self.model.gradient.data, cands, weights)
        new_sh = self._blend_old_values(self.model.sh.data, cands, weights)

        # Conservation: scale so global sums are preserved
        new_density *= (old_density_sum / new_density.sum().clamp(min=1e-8))
        new_rgb *= (old_rgb_sum / new_rgb.sum(dim=0).clamp(min=1e-8))

        self.model.density = nn.Parameter(new_density.contiguous().requires_grad_(True))
        self.model.rgb = nn.Parameter(new_rgb.contiguous().requires_grad_(True))
        self.model.gradient = nn.Parameter(new_gradient.contiguous().requires_grad_(True))
        self.model.sh = nn.Parameter(new_sh.contiguous().requires_grad_(True))

        # Update model indices
        self.model.indices = new_indices.int()

        # Cull low-density tets
        cull_mask = None
        if density_threshold > 0 or alpha_threshold > 0:
            tet_density = self.model.calc_tet_density()
            tet_alpha = self.model.calc_tet_alpha(mode="min", density=tet_density)
            cull_mask = (tet_density > density_threshold) | (tet_alpha > alpha_threshold)
            self.model.empty_indices = self.model.indices[~cull_mask]
            self.model.indices = self.model.indices[cull_mask]
            self.model.density = nn.Parameter(self.model.density.data[cull_mask].contiguous().requires_grad_(True))
            self.model.rgb = nn.Parameter(self.model.rgb.data[cull_mask].contiguous().requires_grad_(True))
            self.model.gradient = nn.Parameter(self.model.gradient.data[cull_mask].contiguous().requires_grad_(True))
            self.model.sh = nn.Parameter(self.model.sh.data[cull_mask].contiguous().requires_grad_(True))
        else:
            self.model.empty_indices = torch.empty(
                (0, 4), dtype=self.model.indices.dtype, device="cuda"
            )

        # Rebuild optimizers then restore blended Adam state
        self._rebuild_optim()
        if old_optim_state:
            self._restore_adam_state_blended(
                self.optim, old_optim_state, cands, weights, cull_mask)
        if old_sh_state:
            self._restore_adam_state_blended(
                self.sh_optim, old_sh_state, cands, weights, cull_mask)

        self.model.device = self.model.density.device
        torch.cuda.empty_cache()

    # --- densification ---
    def add_points(self, new_verts: torch.Tensor, raw_verts=False):
        self.model.interior_vertices = self.vertex_optim.cat_tensors_to_optimizer(
            dict(interior_vertices=new_verts)
        )["interior_vertices"]
        # Retriangulate and transfer per-tet state via optimizer
        self.update_triangulation()

    def remove_points(self, keep_mask: torch.Tensor):
        keep_mask = keep_mask[: self.model.interior_vertices.shape[0]]
        self.model.interior_vertices = self.vertex_optim.prune_optimizer(keep_mask)[
            "interior_vertices"
        ]
        self.update_triangulation()

    @torch.no_grad()
    def split(self, split_point, **kwargs):
        self.add_points(split_point)

    def clip_grad_norm_(self, max_norm):
        torch.nn.utils.clip_grad_norm_(self.model.density, max_norm)
        torch.nn.utils.clip_grad_norm_(self.model.rgb, max_norm)
        torch.nn.utils.clip_grad_norm_(self.model.gradient, max_norm)
        torch.nn.utils.clip_grad_norm_(self.model.sh, max_norm)
