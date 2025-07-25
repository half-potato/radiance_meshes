import torch
from utils.contraction import contract_mean_std
from utils.contraction import contract_points, inv_contract_points
from models.base_model import BaseModel
from muon import SingleDeviceMuonWithAuxAdam
import math
from data.camera import Camera
from torch import nn
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint
import gc

from utils.topo_utils import calculate_circumcenters_torch
from utils.contraction import contract_mean_std
from utils.contraction import contract_points, inv_contract_points

from utils.train_util import get_expon_lr_func
from pathlib import Path
import numpy as np
from utils.args import Args
from utils.model_util import *


class FrozenTetModel(BaseModel):
    def __init__(self,
                 int_vertices: torch.Tensor,
                 ext_vertices: torch.Tensor,
                 indices: torch.Tensor,
                 center: torch.Tensor,
                 features: torch.Tensor,
                 scene_scaling: float,
                 max_sh_deg=2,
                 glo_dim=0,
                 **kwargs):
        super().__init__()
        self.device = int_vertices.device
        self.max_sh_deg = max_sh_deg
        self.dir_offset = torch.tensor([
            [0, 0],
            [math.pi, 0],
        ], device=self.device)
        sh_dim = ((1+max_sh_deg)**2-1)*3
        self.backbone = torch.compile(Heads(
            features.shape[1],
            sh_dim,
            glo_dim=glo_dim,
            **kwargs)).to(self.device)
        self.features = nn.Parameter(features, requires_grad=True)
        self.default_glo = None if glo_dim == 0 else torch.zeros((1, glo_dim), device=self.device)
        self.chunk_size = 408576
        self.mask_values = True
        self.frozen = True
        self.alpha = 0
        self.linear = False
        self.feature_dim = 7

        self.register_buffer("interior_vertices", int_vertices)          # immutable
        self.register_buffer("ext_vertices", ext_vertices)
        self.register_buffer('indices', indices.to(self.device))
        self.register_buffer('center', center.reshape(1, 3))
        self.register_buffer('scene_scaling', torch.tensor(float(scene_scaling), device=self.device))
        self.update_triangulation()

    def get_circumcenters(self):
        circumcenter =  pre_calc_cell_values(
            self.vertices, self.indices, self.center, self.scene_scaling)
        return circumcenter

    def get_cell_values(self, camera: Camera, mask=None,
                        all_circumcenters=None, glo=None):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices
        glo = glo if glo is not None else self.default_glo

        outputs = []
        normed_cc = []
        start = 0
        features = self.features[mask] if mask is not None else self.features
        cam_center = camera.camera_center.to(self.device)
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])
            circumcenters, normalized, density, rgb, grd, sh = self.compute_batch_features(
                vertices, indices, start, end, features=features, circumcenters=all_circumcenters, glo=glo)
            tets = vertices[indices[start:end]]
            tet_color_raw = eval_sh(
                tets.mean(dim=1).detach(),
                RGB2SH(rgb),
                sh.reshape(-1, (self.max_sh_deg+1)**2 - 1, 3).half(),
                cam_center,
                self.max_sh_deg).float()
            dvrgbs = activate_output(cam_center, tet_color_raw,
                                     density, grd,
                                     circumcenters,
                                     tets)
            normed_cc.append(normalized)
            outputs.append(dvrgbs)
        features = torch.cat(outputs, dim=0)
        normed_cc = torch.cat(normed_cc, dim=0)
        return normed_cc, features

    def compute_batch_features(self, vertices, indices, start, end, features=None, circumcenters=None, glo=None):
        if circumcenters is None:
            tets = vertices[indices[start:end]]
            circumcenter, radius = calculate_circumcenters_torch(tets.double())
        else:
            circumcenter = circumcenters[start:end]
        if self.training:
            circumcenter += self.alpha*torch.rand_like(circumcenter)
        normalized = (circumcenter - self.center) / self.scene_scaling

        glo = glo if glo is not None else self.default_glo

        if features is not None:
            output = checkpoint(self.backbone, features[start:end], glo, use_reentrant=True)
        else:
            output = checkpoint(self.backbone, self.features[start:end], glo, use_reentrant=True)
        return circumcenter, normalized, *output

    def compute_features(self):
        vertices = self.vertices
        indices = self.indices
        features = self.features
        cs, ds, rs, gs, ss = [], [], [], [], []
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])

            circumcenters, _, density, rgb, grd, sh = self.compute_batch_features(vertices, indices, start, end, features=features, glo=self.default_glo)
            tets = vertices[indices[start:end]]
            cs.append(circumcenters)
            ds.append(density)
            ss.append(sh)
            rs.append(rgb)
            gs.append(grd)
        cs = torch.cat(cs, dim=0)
        ds = torch.cat(ds, dim=0)
        rs = torch.cat(rs, dim=0)
        gs = torch.cat(gs, dim=0)
        ss = torch.cat(ss, dim=0)
        return cs, ds, rs, gs, ss


    @staticmethod
    def load_ckpt(path: Path, device, **overrides):
        ckpt_path = path / "ckpt.pth"
        config_path = path / "config.json"
        config = Args.load_from_json(str(config_path))
        for k, v in overrides.items():
            config[k] = v
        ckpt = torch.load(ckpt_path)
        vertices = ckpt['interior_vertices']
        indices = ckpt["indices"]  # shape (N,4)
        print(f"Loaded {vertices.shape[0]} vertices")
        ext_vertices = ckpt['ext_vertices']
        model = FrozenTetModel(
            int_vertices=vertices.to(device),
            ext_vertices=ext_vertices,
            features=ckpt['features'],
            indices=indices,
            center=ckpt['center'],
            scene_scaling=ckpt['scene_scaling'],
            **config.as_dict())
        model.load_state_dict(ckpt)
        # model.min_t = model.scene_scaling * config.base_min_t
        model.min_t = config.base_min_t
        model.indices = torch.as_tensor(indices).cuda()
        return model

    def calc_tet_density(self):
        densities = []
        verts = self.vertices
        for start in range(0, self.indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.indices.shape[0])
            
            _, _, density, _, _, _ = self.compute_batch_features(verts, self.indices, start, end, glo=self.default_glo)

            densities.append(density.reshape(-1))
        return torch.cat(densities)

    def inv_contract(self, points):
        return inv_contract_points(points) * self.scene_scaling + self.center

    def contract(self, points):
        return contract_points((points - self.center) / self.scene_scaling)

    @property
    def vertices(self):
        verts = self.interior_vertices
        return torch.cat([verts, self.ext_vertices])

    def sh_up(self):
        pass

    @torch.no_grad()
    def update_triangulation(self, high_precision=False, density_threshold=0.0, alpha_threshold=0.0):
        pass

    def __len__(self):
        return self.vertices.shape[0]


class FrozenTetOptimizer:
    def __init__(self,
                 model: FrozenTetModel,
                 feature_lr: float=1e-3,
                 final_feature_lr: float=1e-4,
                 fnetwork_lr: float=1e-3,
                 final_fnetwork_lr: float=1e-3,

                 weight_decay=1e-10,
                 lr_delay: int = 500,
                 lambda_tv: float = 0.0,
                 lambda_density: float = 0.0,

                 glo_net_decay: float = 0,
                 glo_network_lr: float = 1e-3,

                 freeze_start: int = 15000,
                 iterations: int = 30000,

                 **kwargs):
        self.weight_decay = weight_decay
        self.lambda_tv = lambda_tv
        self.lambda_density = lambda_density
        def process(body, lr, weight_decay=0):
            hidden_weights = [p for p in body.parameters() if p.ndim >= 2]
            hidden_gains_biases = [p for p in body.parameters() if p.ndim < 2]
            a = dict(
                params=hidden_weights,
                use_muon = True,
                momentum=0.95,
                lr=lr,
                weight_decay=weight_decay,
            )
            b = dict(
                params=hidden_gains_biases,
                use_muon = False,
                betas=(0.9, 0.999),
                eps=1e-15,
                weight_decay=weight_decay,
            )
            return [a, b]
        glo_p = process(model.backbone.glo_net, glo_network_lr, weight_decay=glo_net_decay) if model.backbone.glo_dim > 0 else []
        self.net_optim = SingleDeviceMuonWithAuxAdam(
            process(model.backbone.density_net, fnetwork_lr) + \
            process(model.backbone.color_net, fnetwork_lr) + \
            process(model.backbone.gradient_net, fnetwork_lr) + \
            process(model.backbone.sh_net, fnetwork_lr) + \
            glo_p
        )
        self.feature_optim = torch.optim.Adam([
            {"params": [model.features],       "lr": feature_lr,       "name": "sh"},
        ])
        self.sh_optim = None
        self.model = model
        self.freeze_start = freeze_start
        self.net_scheduler = get_expon_lr_func(lr_init=fnetwork_lr,
                                                lr_final=final_fnetwork_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=iterations - self.freeze_start)

        self.feature_scheduler = get_expon_lr_func(lr_init=feature_lr,
                                                lr_final=final_feature_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=iterations - self.freeze_start)
        self.iteration = 0

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self.iteration = iteration
        for param_group in self.net_optim.param_groups:
            lr = self.net_scheduler(iteration - self.freeze_start)
            param_group['lr'] = lr
        for param_group in self.feature_optim.param_groups:
            lr = self.feature_scheduler(iteration - self.freeze_start)
            param_group['lr'] = lr

    def step(self):
        self.net_optim.step()
        self.feature_optim.step()

    def zero_grad(self):
        self.net_optim.zero_grad()
        self.feature_optim.zero_grad()

    # compatibility shims ------------------------------------------------
    def main_step(self):
        self.step()

    def main_zero_grad(self):
        self.zero_grad()

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

    def update_triangulation(self, *_, **__):
        return None


def bake_from_model(base_model, args, chunk_size: int = 408_576) -> FrozenTetModel:
    """Convert an existing neural‑field `Model` into a parameter‑only
    `FrozenTetModel`.  All per‑tet features are *evaluated once* through the
    network and stored explicitly so that no backbone is needed afterwards."""
    device = base_model.device

    vertices = base_model.vertices.detach()
    int_vertices  = vertices[: base_model.num_int_verts]
    ext_vertices  = base_model.ext_vertices.detach()
    indices       = base_model.indices.detach()

    features = []
    for start in range(0, indices.shape[0], chunk_size):
        end = min(start + chunk_size, indices.shape[0])

        tets = vertices[indices[start:end]]
        circumcenter, radius = calculate_circumcenters_torch(tets.double())
        normalized = (circumcenter - base_model.center) / base_model.scene_scaling
        radius = torch.linalg.norm(circumcenter - vertices[indices[start:end, 0]], dim=-1)
        cv, cr = contract_mean_std(normalized, radius / base_model.scene_scaling)
        x = (cv/2 + 1)/2
        output = base_model.backbone.encode(x, cr)
        features.append(output)

    features  = torch.cat(features, 0)

    fmodel = FrozenTetModel(
        int_vertices=int_vertices.to(device),
        ext_vertices=ext_vertices.to(device),
        indices=indices.to(device),
        features=features.to(device),
        center=base_model.center.detach().to(device),
        scene_scaling=base_model.scene_scaling.detach().to(device),
        density_offset=base_model.density_offset,
        max_sh_deg=base_model.max_sh_deg,
        chunk_size=chunk_size,
        sh_hidden_dim=base_model.backbone.sh_hidden_dim,
        hidden_dim=base_model.backbone.hidden_dim,
        glo_dim=base_model.glo_dim,
    )
    fmodel.backbone.load_state_dict(base_model.backbone.state_dict(), strict=False)
    return fmodel

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


@torch.no_grad()
def freeze_model(
    base_model,
    args,
    chunk_size: int = 408_576,
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
    frozen_model = bake_from_model(base_model, args, chunk_size=chunk_size)

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
