import torch
from torch import nn
from typing import Optional, Tuple
import gc
import tinyplypy
import numpy as np
from pathlib import Path

from data.camera import Camera
from utils.topo_utils import (
    build_tv_struct, max_density_contrast
)
from utils.model_util import activate_output
from utils import optim
from utils.model_util import *
from utils.safe_math import safe_log, safe_exp
from utils.train_util import get_expon_lr_func, SpikingLR
from utils import mesh_util
from utils.args import Args
from models.base_model import BaseModel
from sh_slang.eval_sh import eval_sh


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
        indices: torch.Tensor,               # (T, 4)
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
        self.register_buffer("interior_vertices", int_vertices)          # immutable
        self.register_buffer("indices", indices.int())
        self.register_buffer("center", center.reshape(1, 3))
        self.register_buffer("scene_scaling", torch.as_tensor(scene_scaling))

        # learnable per‑tet parameters -------------------------------------------
        # self.density   = nn.Parameter(safe_log(density))    # (T, 1)
        # self.gradient  = nn.Parameter(torch.atanh(gradient.clip(min=-0.99, max=0.99)))   # (T, 3, 3)
        self.density   = nn.Parameter(density, requires_grad=True)    # (T, 1)
        self.gradient  = nn.Parameter(gradient, requires_grad=True)   # (T, 3, 3)
        self.rgb       = nn.Parameter(rgb, requires_grad=True)        # (T, 3)
        self.sh        = nn.Parameter(sh, requires_grad=True)         # (T, SH, 3)

        # misc --------------------------------------------------------------------
        self.max_sh_deg      = max_sh_deg
        self.chunk_size      = chunk_size
        self.device          = self.density.device

        self.mask_values = True
        self.frozen = True
        self.linear = False
        self.feature_dim = 7

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
        indices = ckpt['indices']
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
            indices=indices.to(device),
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
        model.min_t = model.scene_scaling * config.base_min_t
        
        return model

    @property
    def vertices(self) -> torch.Tensor:
        """Concatenated vertex tensor (internal + exterior)."""
        return self.interior_vertices


    def compute_features(self):
        vertices = self.vertices
        indices = self.indices
        tets = vertices[indices]
        circumcenters, radius = calculate_circumcenters_torch(tets.double())
        return circumcenters, self.density, self.rgb, safe_div(self.gradient, radius.reshape(-1, 1, 1)), self.sh

    def get_cell_values(self, camera: Camera, mask=None,
                        all_circumcenters=None, glo=None):
        cam_center = camera.camera_center.to(self.device)
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices
        tets = vertices[indices]
        if all_circumcenters is None:
            all_circumcenters, radius = calculate_circumcenters_torch(tets.double())

        rgb = self.rgb if mask is None else self.rgb[mask]
        density = self.density if mask is None else self.density[mask]
        grd = self.gradient if mask is None else self.gradient[mask]
        sh = self.sh if mask is None else self.sh[mask]

        tet_color_raw = eval_sh(
            tets.mean(dim=1).detach(),
            RGB2SH(rgb),
            sh.reshape(-1, (self.max_sh_deg+1)**2 - 1, 3),
            cam_center,
            self.max_sh_deg).float()
        cell_output = activate_output(cam_center, tet_color_raw,
            density, grd,
            all_circumcenters,
            tets)

        normalized = (all_circumcenters - self.center) / self.scene_scaling
        return normalized, cell_output

    # ------------------------------------------------------------------
    # geometry is frozen
    # ------------------------------------------------------------------
    def update_triangulation(self, *_, **__):
        return None

    @property
    def num_int_verts(self):
        return self.interior_vertices.shape[0]

    def sh_up(self):
        pass

    def calc_tet_density(self):
        densities = []
        verts = self.vertices
        _, _, densities, _, _, _ = self.compute_batch_features(verts, self.indices)
        return densities.reshape(-1)

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
        weight_decay: float = 1e-10,
        lambda_tv:    float = 0.0,
        lambda_density: float = 0.0,
        lr_delay_multi=1e-8,
        lr_delay=0,
        freeze_start: int = 15000,
        iterations: int = 30000,
        **kwargs
    ) -> None:
        self.model = model
        self.weight_decay   = weight_decay
        self.lambda_tv      = lambda_tv
        self.lambda_density = lambda_density

        # ------------------------------------------------------------------
        # single optimiser with four parameter groups
        # ------------------------------------------------------------------
        # self.optim = torch.optim.RMSprop([
        self.optim = torch.optim.Adam([
            {"params": [model.density],  "lr": freeze_lr,  "name": "density"},
            {"params": [model.rgb],      "lr": freeze_lr,    "name": "color"},
            {"params": [model.gradient], "lr": freeze_lr, "name": "gradient"},
        ])
        # self.sh_optim = torch.optim.RMSprop([
        self.sh_optim = torch.optim.Adam([
            {"params": [model.sh],       "lr": freeze_lr / 20,       "name": "sh"},
        ])#, eps=1e-6)
        self.freeze_start = freeze_start
        self.scheduler = get_expon_lr_func(lr_init=freeze_lr,
                                           lr_final=final_freeze_lr,
                                           lr_delay_mult=lr_delay_multi,
                                           max_steps=iterations - self.freeze_start,
                                           lr_delay_steps=lr_delay)

        # alias for external training scripts that expected these names
        self.net_optim   = self.optim
        self.vertex_optim = None  # geometry is frozen

    def update_triangulation(self, *_, **__):
        return None

    def step(self):
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()

    def main_step(self):
        self.step()

    def main_zero_grad(self):
        self.zero_grad()

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self.iteration = iteration
        for param_group in self.optim.param_groups:
            lr = self.scheduler(iteration - self.freeze_start)
            param_group['lr'] = lr

    @torch.no_grad()
    def split(self, clone_indices, split_point, split_mode):
        print("Split called on frozen optimizer")

    def regularizer(self, *_):
        if self.lambda_tv > 0:
            # simple TV on densities as example
            diff = (self.model.density[self.pairs[:, 0]] - self.model.density[self.pairs[:, 1]]).abs()
            tv_loss = (self.face_area * diff).sum() / self.face_area.sum()
        else:
            tv_loss = 0.0

        return self.lambda_tv * tv_loss

def bake_from_model(base_model, mask, chunk_size: int = 408_576) -> FrozenTetModel:
    """Convert an existing neural‑field `Model` into a parameter‑only
    `FrozenTetModel`.  All per‑tet features are *evaluated once* through the
    network and stored explicitly so that no backbone is needed afterwards."""
    device = base_model.device

    vertices_full = base_model.vertices.detach()
    int_vertices  = vertices_full[: base_model.num_int_verts]
    indices       = base_model.indices[mask].detach()

    d_list, rgb_list, grd_list, sh_list = [], [], [], []
    for start in range(0, indices.shape[0], chunk_size):
        end = min(start + chunk_size, indices.shape[0])
        _, _, density, rgb, grd, sh = base_model.compute_batch_features(
            vertices_full, indices, start, end
        )
        d_list.append(density)
        rgb_list.append(rgb)
        grd_list.append(grd)
        sh_list.append(sh)

    density  = torch.cat(d_list, 0)
    rgb      = torch.cat(rgb_list, 0)
    gradient = torch.cat(grd_list, 0)
    sh       = torch.cat(sh_list, 0)

    density, rgb, gradient, sh = (x.clone().detach() for x in (density, rgb, gradient, sh))

    return FrozenTetModel(
        int_vertices=int_vertices.to(device),
        indices=indices.to(device),
        density=density.to(device),
        rgb=rgb.to(device),
        gradient=gradient.to(device),
        sh=sh.to(device),
        center=base_model.center.detach().to(device),
        scene_scaling=base_model.scene_scaling.detach().to(device),
        max_sh_deg=base_model.max_sh_deg,
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
