import torch
from torch import nn
from typing import Optional, Tuple
import gc
import tinyplypy
import numpy as np
from pathlib import Path
import open3d as o3d

from gdel3d import Del
from utils.topo_utils import tet_volumes
from utils.model_util import activate_output
from utils import optim
from utils.model_util import *
from utils.safe_math import safe_log, safe_exp
from utils.train_util import get_expon_lr_func, SpikingLR
from utils import mesh_util
from utils.args import Args
from models.base_model import BaseModel
from dtlookup import lookup_inds


class FrozenTetModel(BaseModel):
    """Minimal field representation with *fixed* tetrahedral geometry.

    The iNGP backbone is entirely removed; instead, each tetrahedron stores its
    own learnable parameters (density, RGB base, per-channel linear gradient,
    and spherical–harmonics coefficients).  The public interface matches the
    original `Model` class so existing rendering and loss code can be reused
    unmodified.
    """

    # ---------------------------------------------------------------------
    # constructor
    # ---------------------------------------------------------------------
    def __init__(
        self,
        int_vertices: torch.Tensor,          # (N_int, 3)
        ext_vertices: torch.Tensor,          # (N_ext, 3)
        indices: torch.Tensor,               # (T, 4)
        empty_indices: torch.Tensor,               # (T, 4)
        density: torch.Tensor,               # (T, 1)
        rgb: torch.Tensor,                   # (T, 3)
        gradient: torch.Tensor,              # (T, 3, 3)
        sh: torch.Tensor,                    # (T, ((max_sh_deg+1)**2-1), 3)
        center: torch.Tensor,                # (1, 3)
        scene_scaling: torch.Tensor | float,
        *,
        max_sh_deg: int = 2,
        chunk_size: int = 408_576,
        **kwargs
    ) -> None:
        super().__init__()

        # geometry ----------------------------------------------------------------
        self.interior_vertices = nn.Parameter(int_vertices.cuda(), requires_grad=True)          # immutable
        self.register_buffer("ext_vertices", ext_vertices.cuda())
        self.register_buffer("indices", indices.int())
        self.empty_indices = indices
        self.register_buffer("center", center.reshape(1, 3))
        self.register_buffer("scene_scaling", torch.as_tensor(scene_scaling))

        # learnable per‑tet parameters -------------------------------------------
        # self.density   = nn.Parameter(safe_log(density))    # (T, 1)
        # self.gradient  = nn.Parameter(torch.atanh(gradient.clip(min=-0.99, max=0.99)))   # (T, 3, 3)
        self.density   = nn.Parameter(density, requires_grad=True)    # (T, 1)
        self.gradient  = nn.Parameter(gradient, requires_grad=True)   # (T, 3, 3)
        self.rgb       = nn.Parameter(rgb, requires_grad=True)        # (T, 3)
        self.sh        = nn.Parameter(sh.half(), requires_grad=True)         # (T, SH, 3)

        # misc --------------------------------------------------------------------
        self.max_sh_deg      = max_sh_deg
        self.chunk_size      = chunk_size
        self.device          = self.density.device
        self.sh_dim = ((1+max_sh_deg)**2-1)*3

        self.mask_values = False
        self.frozen = False
        self.linear = False
        self.feature_dim = 7

    def sh_up(self):
        pass

    @staticmethod
    def load_ckpt(path: Path, device):
        """Load a checkpoint from a directory containing ckpt.pth and alldata.json.

        Parameters
        ----------
        path : Path
            Directory containing the checkpoint files
        device : torch.device
            Device to load the model onto

        Returns
        -------
        FrozenTetModel
            The loaded model
        """
        ckpt_path = path / "ckpt.pth"
        config_path = path / "config.json"
        config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)
        
        # Extract required parameters from checkpoint
        int_vertices = ckpt['interior_vertices']
        ext_vertices = ckpt['ext_vertices']
        indices = ckpt['indices']
        if 'empty_indices' in ckpt:
            empty_indices = ckpt['empty_indices']
            del ckpt['empty_indices']
        else:
            empty_indices = torch.empty((0, 4), dtype=indices.dtype, device=indices.device)

        density = ckpt['density']
        rgb = ckpt['rgb']
        gradient = ckpt['gradient']
        sh = ckpt['sh']
        center = ckpt['center']
        scene_scaling = ckpt['scene_scaling']
        
        print(f"Loaded {int_vertices.shape[0]} internal vertices")
        
        # Create model instance
        model = FrozenTetModel(
            int_vertices=int_vertices.to(device),
            ext_vertices=ext_vertices.to(device),
            indices=indices.to(device),
            empty_indices=empty_indices.to(device),
            density=density.to(device),
            rgb=rgb.to(device),
            gradient=gradient.to(device),
            sh=sh.to(device),
            center=center.to(device),
            scene_scaling=scene_scaling.to(device),
            max_sh_deg=config.max_sh_deg,
            chunk_size=config.chunk_size if hasattr(config, 'chunk_size') else 408_576,
        )

        
        # Load state dict to ensure all parameters are properly loaded
        model.load_state_dict(ckpt)
        model.min_t = config.min_t
        
        return model

    # ------------------------------------------------------------------
    # convenience properties
    # ------------------------------------------------------------------
    @property
    def vertices(self) -> torch.Tensor:
        """Concatenated vertex tensor (internal + exterior)."""
        return torch.cat([self.interior_vertices, self.ext_vertices], dim=0)


    # ------------------------------------------------------------------
    # core helper (network‑free)
    # ------------------------------------------------------------------
    def compute_batch_features(
        self,
        vertices: torch.Tensor,
        indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        circumcenters: Optional[torch.Tensor] = None,
        **kwargs
    ):

        if circumcenters is None:
            circumcenter = pre_calc_cell_values(
                vertices, indices
            )
        else:
            circumcenter = circumcenters

        if mask is not None:
            density  = self.density[mask]
            grd      = self.gradient[mask]
            rgb      = self.rgb[mask]
            sh       = self.sh[mask]
        else:
            density  = self.density
            grd      = self.gradient
            rgb      = self.rgb
            sh       = self.sh

        sh_dim = (self.max_sh_deg+1)**2 - 1
        attr = torch.empty((sh.shape[0], 0), device=grd.device)
        return circumcenter, density, rgb, grd, sh.reshape(-1, sh_dim, 3), attr

    def compute_features(self, offset=False):
        vertices = self.vertices
        indices = self.indices
        circumcenters, density, rgb, grd, sh, attr = self.compute_batch_features(vertices, indices)
        tets = vertices[indices]
        if offset:
            base_color_v0_raw, normed_grd = offset_normalize(rgb, grd, circumcenters, tets)
            return circumcenters, density, rgb, normed_grd, sh
        else:
            return circumcenters, density, rgb, grd, sh

    # ------------------------------------------------------------------
    # public renderer interface
    # ------------------------------------------------------------------
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
            sh.reshape(-1, (self.max_sh_deg+1)**2 - 1, 3),
            attr,
            indices,
            cc, vertices,
            self.max_sh_deg, self.max_sh_deg,
        )
        return sh, cell_output

    @torch.no_grad()
    def update_triangulation(self, old_vertices, old_indices, high_precision=False, density_threshold=0.0, alpha_threshold=0.0):
        return
        torch.cuda.empty_cache()
        verts = self.vertices
        if high_precision:
            indices_np = Delaunay(verts.detach().cpu().numpy()).simplices.astype(np.int32)
            # self.indices = torch.tensor(indices_np, device=verts.device).int().cuda()
        else:
            v = Del(verts.shape[0])
            indices_np, prev = v.compute(verts.detach().cpu().double())
            indices_np = indices_np.numpy()
            indices_np = indices_np[(indices_np < verts.shape[0]).all(axis=1)]
            del prev
        

        # Ensure volume is positive
        indices = torch.as_tensor(indices_np).cuda()
        vols = tet_volumes(verts[indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            indices[reverse_mask] = indices[reverse_mask][:, [1, 0, 2, 3]]

        new_centroids = verts[indices].mean(dim=1)
        new_inds = lookup_inds(old_indices, old_vertices, new_centroids).clip(min=0)

        # Cull tets with low density
        self.full_indices = indices.clone()
        self.indices = indices
        return new_inds

    @property
    def num_int_verts(self):
        return self.interior_vertices.shape[0]

    def calc_tet_density(self):
        densities = []
        verts = self.vertices
        _, densities, _, _, _, _ = self.compute_batch_features(verts, self.indices)
        return densities.reshape(-1)



# =============================================================================
# 3.  OPTIMISER FOR FROZEN MODEL                                             |
# =============================================================================

class FrozenTetOptimizer:
    """Lightweight optimiser tailored to `FrozenTetModel`.

    * Only the per‑tet tensors are trainable (density, rgb, gradient, SH).
    * No vertex updates, no encoding, no backbone network.
    * Learning rate scheduling is optional – defaults to constant lrs.
    """

    def __init__(
        self,
        model: FrozenTetModel,
        *,
        freeze_lr:   float = 1e-3,
        final_freeze_lr:   float = 1e-4,
        lr_delay_multi=1e-8,
        lr_delay=0,
        vertices_lr: float=4e-4,
        final_vertices_lr: float=4e-7,
        vert_lr_delay: int = 500,
        vertices_lr_delay_multi: float=0.01,
        freeze_start: int = 15000,
        iterations: int = 30000,
        spike_duration: int = 20,
        densify_interval: int = 500,
        densify_end: int = 15000,
        densify_start: int = 2000,
        split_std: float = 0.5,
        **kwargs
    ) -> None:
        self.model = model
        self.split_std = split_std

        # ------------------------------------------------------------------
        # single optimiser with four parameter groups
        # ------------------------------------------------------------------
        self.optim = optim.CustomAdam([
        # self.optim = torch.optim.Adam([
            {"params": [model.density],  "lr": freeze_lr,  "name": "density"},
            {"params": [model.rgb],      "lr": freeze_lr,    "name": "color"},
            {"params": [model.gradient], "lr": freeze_lr, "name": "gradient"},
        ])
        self.sh_optim = optim.CustomAdam([
            {"params": [model.sh],       "lr": freeze_lr,       "name": "sh"},
        ], eps=1e-6)
        self.vert_lr_multi = float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.interior_vertices], "lr": self.vert_lr_multi*vertices_lr, "name": "interior_vertices"},
        ])
        self.freeze_start = freeze_start
        self.scheduler = get_expon_lr_func(lr_init=freeze_lr,
                                           lr_final=final_freeze_lr,
                                           lr_delay_mult=lr_delay_multi,
                                           max_steps=iterations,
                                           lr_delay_steps=lr_delay)

        self.vertex_lr = self.vert_lr_multi*vertices_lr
        base_vertex_scheduler = get_expon_lr_func(lr_init=self.vertex_lr,
                                                lr_final=self.vert_lr_multi*final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_multi,
                                                max_steps=iterations,
                                                lr_delay_steps=vert_lr_delay)

        self.vertex_scheduler_args = base_vertex_scheduler
        self.vertex_scheduler_args = SpikingLR(
            spike_duration, freeze_start, base_vertex_scheduler,
            densify_start, densify_interval, densify_end,
            self.vertex_lr, self.vertex_lr)
            # self.vertex_lr, self.vertex_lr)

        # alias for external training scripts that expected these names
        self.net_optim   = self.optim

    # ------------------------------------------------------------------
    # public helpers ----------------------------------------------------
    # ------------------------------------------------------------------
    def step(self):
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()

    # compatibility shims ------------------------------------------------
    def main_step(self):
        self.step()

    def main_zero_grad(self):
        self.zero_grad()

    def update_learning_rate(self, iteration):
        # ''' Learning rate scheduling per step '''
        # self.iteration = iteration
        # for param_group in self.optim.param_groups:
        #     lr = self.scheduler(iteration)
        #     param_group['lr'] = lr
        #
        # for param_group in self.sh_optim.param_groups:
        #     lr = self.scheduler(iteration)
        #     param_group['lr'] = lr
        # for param_group in self.vertex_optim.param_groups:
        #     if param_group["name"] == "interior_vertices":
        #         lr = self.vertex_scheduler_args(iteration)
        #         self.vertex_lr = lr
        #         param_group['lr'] = lr
        pass

    # ------------------------------------------------------------------
    # regularisers ------------------------------------------------------
    # ------------------------------------------------------------------
    def regularizer(self, render_pkg, lambda_weight_decay, **kwargs):
        # wd_loss = self.weight_decay * sum((p ** 2).mean() for p in [
        #     self.model.density, self.model.rgb, self.model.gradient, self.model.sh
        # ])

        # if self.lambda_density > 0:
        #     density = self.model.density.squeeze(-1)
        #     density_loss = density.mean()
        # else:
        #     density_loss = 0.0
        #
        # if self.lambda_tv > 0:
        #     # simple TV on densities as example
        #     diff = (self.model.density[self.pairs[:, 0]] - self.model.density[self.pairs[:, 1]]).abs()
        #     tv_loss = (self.face_area * diff).sum() / self.face_area.sum()
        # else:
        #     tv_loss = 0.0

        return 0.0

    def add_points(self, new_verts: torch.Tensor, raw_verts=False):
        self.model.interior_vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            interior_vertices = new_verts
        ))['interior_vertices']
        self.model.update_triangulation()

    @torch.no_grad()
    def split(self, clone_indices, split_point, split_mode, split_std, **kwargs):
        device = self.model.device
        clone_vertices = self.model.vertices[clone_indices]

        if split_mode == "circumcenter":
            circumcenters, radius = calculate_circumcenters_torch(clone_vertices)
            radius = radius.reshape(-1, 1)
            circumcenters = circumcenters.reshape(-1, 3)
            sphere_loc = sample_uniform_in_sphere(circumcenters.shape[0], 3).to(device)
            r = torch.randn((clone_indices.shape[0], 1), device=self.model.device)
            r[r.abs() < 1e-2] = 1e-2
            sampled_radius = (r * self.split_std + 1) * radius
            new_vertex_location = l2_normalize_th(sphere_loc) * sampled_radius + circumcenters
        elif split_mode == "barycenter":
            barycentric_weights = 0.25*torch.ones((clone_indices.shape[0], clone_indices.shape[1], 1), device=device).clip(min=0.01, max=0.99)
            new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights).sum(dim=1)
        elif split_mode == "barycentric":
            barycentric = torch.rand((clone_indices.shape[0], clone_indices.shape[1], 1), device=device).clip(min=0.01, max=0.99)
            barycentric_weights = barycentric / (1e-3+barycentric.sum(dim=1, keepdim=True))
            new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights).sum(dim=1)
        elif split_mode == "split_point":
            _, radius = calculate_circumcenters_torch(self.model.vertices[clone_indices])
            split_point += (split_std * radius.reshape(-1, 1)).clip(min=1e-3, max=3) * torch.randn(*split_point.shape, device=self.model.device)
            new_vertex_location = split_point
            # new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights.unsqueeze(-1)).sum(dim=1)
        elif split_mode == "split_point_c":
            barycentric_weights = calc_barycentric(split_point, clone_vertices).clip(min=0)
            barycentric_weights = barycentric_weights / (1e-3+barycentric_weights.sum(dim=1, keepdim=True))
            barycentric_weights += 1e-4*torch.randn(*barycentric_weights.shape, device=self.model.device)
            new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise Exception(f"Split mode: {split_mode} not supported")
        self.add_points(new_vertex_location)

    def update_triangulation(self, **kwargs):
        return
        new_inds = self.model.update_triangulation(**kwargs)
        tensors = self.optim.tensor_index(
            new_inds, ['density', 'color', 'gradient'])
        self.model.density, self.model.rgb, self.model.gradient = tensors['density'], tensors['color'], tensors['gradient'], 
        self.model.sh = self.sh_optim.tensor_index(new_inds, ['sh'])['sh']

def bake_from_model(base_model, mask, chunk_size: int = 408_576) -> FrozenTetModel:
    """Convert an existing neural‑field `Model` into a parameter‑only
    `FrozenTetModel`.  All per‑tet features are *evaluated once* through the
    network and stored explicitly so that no backbone is needed afterwards."""
    device = base_model.device

    vertices_full = base_model.vertices.detach()
    int_vertices  = vertices_full[: base_model.num_int_verts]
    ext_vertices  = base_model.ext_vertices.detach()
    # full_mask[full_mask] = mask.cpu()
    indices       = base_model.indices.detach()
    empty_indices       = base_model.empty_indices.detach()
    max_sh_deg = base_model.max_sh_deg
    center = base_model.center
    scene_scaling = base_model.scene_scaling

    d_list, rgb_list, grd_list, sh_list = [], [], [], []
    for start in range(0, indices.shape[0], chunk_size):
        end = min(start + chunk_size, indices.shape[0])
        _, density, rgb, grd, sh, attr = base_model.compute_batch_features(
            vertices_full, indices, start, end
        )
        d_list.append(density.detach().cpu())
        rgb_list.append(rgb.detach().cpu())
        grd_list.append(grd.detach().cpu())
        sh_list.append(sh.detach().cpu())

    _offload_model_to_cpu(base_model)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    density  = torch.cat(d_list, 0).float()
    rgb      = torch.cat(rgb_list, 0).float()
    gradient = torch.cat(grd_list, 0).float()
    sh       = torch.cat(sh_list, 0)

    return FrozenTetModel(
        int_vertices=int_vertices.to(device),
        ext_vertices=ext_vertices.to(device),
        indices=indices.to(device),
        empty_indices=empty_indices,
        density=density.to(device),
        rgb=rgb.to(device),
        gradient=gradient.to(device),
        sh=sh.to(device),
        center=center.detach().to(device),
        scene_scaling=scene_scaling.detach().to(device),
        max_sh_deg=max_sh_deg,
        chunk_size=chunk_size,
    )


def _offload_model_to_cpu(model: nn.Module):
    """Move every parameter & buffer to CPU and drop gradients to free GPU VRAM."""
    if model is None:
        return
    for p in model.parameters(recurse=True):
        p.grad = None
        p.data = p.data.cpu()
    for b in model.buffers(recurse=True):
        b.data = b.data.cpu()
    torch.cuda.empty_cache()

def freeze_model(
    base_model,
    mask,
    args,
    chunk_size: int = 408_576,
    **kwargs
) -> Tuple[FrozenTetModel, FrozenTetOptimizer]:
    """Utility wrapper to *freeze* a trained neural‑field `Model`, produce the
    corresponding `FrozenTetModel`, and return a ready‑to‑use
    `FrozenTetOptimizer` so training can continue seamlessly.

    Returns
    -------
    FrozenTetModel
        Parameter‑only representation of the field.
    FrozenTetOptimizer
        Optimiser bound to the frozen model.
    """
    print("Freezing model")
    frozen_model = bake_from_model(base_model, mask, chunk_size=chunk_size)

    frozen_optim = FrozenTetOptimizer(
        frozen_model,
        **args.as_dict()
    )

    # free GPU memory used by the big backbone (optional but handy)
    _offload_model_to_cpu(base_model)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    return frozen_model, frozen_optim
