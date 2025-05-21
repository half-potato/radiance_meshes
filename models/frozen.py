import torch
from torch import nn
from typing import Optional, Tuple
import gc

# --- project utilities ------------------------------------------------------
from data.camera import Camera
from utils.topo_utils import (
    build_tv_struct,             # for TV regulariser
)
from utils.model_util import activate_output
from utils import optim          # project's CustomAdam wrapper
from utils.model_util import *

# =============================================================================
# 1.  FROZEN TET MODEL (no neural network)                                    |
# =============================================================================

class FrozenTetModel(nn.Module):
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
        density: torch.Tensor,               # (T, 1)
        rgb: torch.Tensor,                   # (T, 3)
        gradient: torch.Tensor,              # (T, 3, 3)
        sh: torch.Tensor,                    # (T, ((max_sh_deg+1)**2-1), 3)
        center: torch.Tensor,                # (1, 3)
        scene_scaling: torch.Tensor | float,
        *,
        density_offset: float = -1.0,
        current_sh_deg: int = 2,
        max_sh_deg: int = 2,
        chunk_size: int = 408_576,
    ) -> None:
        super().__init__()

        # geometry ----------------------------------------------------------------
        self.register_buffer("contracted_vertices", int_vertices)          # immutable
        self.register_buffer("ext_vertices", ext_vertices)
        self.register_buffer("indices", indices.int())
        self.register_buffer("center", center.reshape(1, 3))
        self.register_buffer("scene_scaling", torch.as_tensor(scene_scaling))

        # learnable per‑tet parameters -------------------------------------------
        self.density   = nn.Parameter(density)    # (T, 1)
        self.rgb       = nn.Parameter(rgb)        # (T, 3)
        self.gradient  = nn.Parameter(gradient)   # (T, 3, 3)
        self.sh        = nn.Parameter(sh)         # (T, SH, 3)

        # misc --------------------------------------------------------------------
        self.density_offset  = density_offset
        self.current_sh_deg  = current_sh_deg
        self.max_sh_deg      = max_sh_deg
        self.chunk_size      = chunk_size
        self.device          = self.density.device

        # flag to inform external code that vertices are frozen
        self.contract_vertices = False
        self.mask_values = False
        self.frozen = True

    # ------------------------------------------------------------------
    # convenience properties
    # ------------------------------------------------------------------
    @property
    def vertices(self) -> torch.Tensor:
        """Concatenated vertex tensor (internal + exterior)."""
        return torch.cat([self.contracted_vertices, self.ext_vertices], dim=0)


    # ------------------------------------------------------------------
    # core helper (network‑free)
    # ------------------------------------------------------------------
    def compute_batch_features(
        self,
        vertices: torch.Tensor,
        indices: torch.Tensor,
        start: int,
        end: int,
        *,
        circumcenters: Optional[torch.Tensor] = None,
    ):
        if circumcenters is None:
            circumcenter, cv, cr, normalized = pre_calc_cell_values(
                vertices, indices[start:end], self.center, self.scene_scaling
            )
        else:
            circumcenter = circumcenters[start:end]
            normalized = (circumcenter - self.center) / self.scene_scaling

        density  = self.density[start:end]
        rgb      = self.rgb[start:end]
        grd      = self.gradient[start:end]
        sh       = self.sh[start:end]

        return circumcenter, normalized, density, rgb, grd.expand(-1, 3, 3), sh

    # ------------------------------------------------------------------
    # public renderer interface
    # ------------------------------------------------------------------
    def get_cell_values(
        self,
        camera: Camera,
        mask: Optional[torch.Tensor] = None,
        all_circumcenters: Optional[torch.Tensor] = None,
        radii: Optional[torch.Tensor] = None,
    ):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices

        normed_cc_list, out_list = [], []
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])
            cc, normalized, density, rgb, grd, sh = self.compute_batch_features(
                vertices, indices, start, end, circumcenters=all_circumcenters
            )
            cell_output = activate_output(
                camera.camera_center.to(self.device),
                density, rgb, grd, sh, indices[start:end],
                cc, vertices,
                self.current_sh_deg, self.max_sh_deg,
                self.density_offset,
            )
            normed_cc_list.append(normalized)
            out_list.append(cell_output)
        return torch.cat(normed_cc_list, 0), torch.cat(out_list, 0)

    # ------------------------------------------------------------------
    # geometry is frozen
    # ------------------------------------------------------------------
    def update_triangulation(self, *_, **__):
        return None

    def __len__(self):
        return self.vertices.shape[0]


    @property
    def num_int_verts(self):
        return self.contracted_vertices.shape[0]

    def get_circumcenters(self):
        circumcenter, cv, cr, normalized =  pre_calc_cell_values(
            self.vertices, self.indices, self.center, self.scene_scaling)
        return cv

    def calc_vert_alpha(self):
        tet_alphas = self.calc_tet_alpha()
        vertex_alpha = torch.zeros((self.vertices.shape[0],), device=self.device)
        indices = self.indices.long()

        reduce_type = "amax"
        vertex_alpha.scatter_reduce_(dim=0, index=indices[..., 0], src=tet_alphas, reduce=reduce_type)
        vertex_alpha.scatter_reduce_(dim=0, index=indices[..., 1], src=tet_alphas, reduce=reduce_type)
        vertex_alpha.scatter_reduce_(dim=0, index=indices[..., 2], src=tet_alphas, reduce=reduce_type)
        vertex_alpha.scatter_reduce_(dim=0, index=indices[..., 3], src=tet_alphas, reduce=reduce_type)
        return vertex_alpha

    def calc_tet_area(self):
        verts = self.vertices
        v0, v1, v2, v3 = verts[self.indices].unbind(dim=1)
        mat = torch.stack([v1-v0, v2-v0, v3-v0], dim=-1)
        return torch.det(mat).abs() / 6.0

    def calc_vert_density(self):
        verts = self.vertices
        vertex_density = torch.zeros((verts.shape[0],), device=self.device)
        indices = self.indices.long()
        for start in range(0, self.indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.indices.shape[0])
            
            _, _, density, _, _, _ = self.compute_batch_features(verts, self.indices, start, end)

            density = density.reshape(-1)
            indices_chunk = indices[start:end]
            reduce_type = "amax"
            vertex_density.scatter_reduce_(dim=0, index=indices_chunk[..., 0], src=density, reduce=reduce_type)
            vertex_density.scatter_reduce_(dim=0, index=indices_chunk[..., 1], src=density, reduce=reduce_type)
            vertex_density.scatter_reduce_(dim=0, index=indices_chunk[..., 2], src=density, reduce=reduce_type)
            vertex_density.scatter_reduce_(dim=0, index=indices_chunk[..., 3], src=density, reduce=reduce_type)
        return vertex_density

    def calc_tet_alpha(self):
        alpha_list = []
        start = 0
        
        verts = self.vertices
        for start in range(0, self.indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.indices.shape[0])
            
            _, _, density, _, _, _ = self.compute_batch_features(verts, self.indices, start, end)

            indices_chunk = self.indices[start:end]
            v0, v1, v2, v3 = verts[indices_chunk[:, 0]], verts[indices_chunk[:, 1]], verts[indices_chunk[:, 2]], verts[indices_chunk[:, 3]]
            
            edge_lengths = torch.stack([
                torch.norm(v0 - v1, dim=1), torch.norm(v0 - v2, dim=1), torch.norm(v0 - v3, dim=1),
                torch.norm(v1 - v2, dim=1), torch.norm(v1 - v3, dim=1), torch.norm(v2 - v3, dim=1)
            ], dim=0).max(dim=0)[0]
            
            # Compute the maximum possible alpha using the largest edge length
            alpha = 1 - torch.exp(-density.reshape(-1) * edge_lengths.reshape(-1))
            alpha_list.append(alpha)
            del edge_lengths, density
        
        alphas = torch.cat(alpha_list, dim=0)
        return alphas

    def tet_variability(self):
        vertices = self.vertices
        indices = self.indices
        tet_var = torch.zeros((indices.shape[0]), device=vertices.device)
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])

            circumcenters, _, density, rgb, grd, sh = self.compute_batch_features(vertices, indices, start, end)
            vcolors = compute_vertex_colors_from_field(
                vertices[indices[start:end]].detach(), rgb.float(), grd.float(),
                circumcenters.float().detach())
            vcolors = torch.nn.functional.softplus(vcolors, beta=10)
            tet_var[start:end] = vcolors.std(dim=1).mean(dim=1)
            # tet_var[start:end] = torch.linalg.norm(clipped_gradients, dim=-1).mean(dim=-1)
        return tet_var

    def calc_tet_density(self):
        densities = []
        verts = self.vertices
        for start in range(0, self.indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.indices.shape[0])
            
            _, _, density, _, _, _ = self.compute_batch_features(verts, self.indices, start, end)

            densities.append(density.reshape(-1))
        return torch.cat(densities)

    @torch.no_grad
    def extract_mesh(self, path, density_threshold=0.5):
        path.mkdir(exist_ok=True, parents=True)
        mask = self.calc_tet_density() > density_threshold
        verts = self.vertices
        meshes = mesh_util.extract_meshes(verts.detach().cpu().numpy(), self.indices[mask])
        for i, mesh in enumerate(meshes):
            F = mesh['face']['vertex_indices']
            if F > 1000:
                mpath = path / f"{i}.ply"
                print(f"Saving #F:{F} to {mpath}")
                tinyplypy.write_ply(str(mpath), mesh, is_binary=False)


# =============================================================================
# 2.  BAKING UTILITY                                                         |
# =============================================================================

@torch.no_grad()
def bake_from_model(base_model: "Model", *, detach: bool = True, chunk_size: int = 408_576) -> FrozenTetModel:
    """Convert an existing neural‑field `Model` into a parameter‑only
    `FrozenTetModel`.  All per‑tet features are *evaluated once* through the
    network and stored explicitly so that no backbone is needed afterwards."""
    device = base_model.device

    vertices_full = base_model.vertices.detach() if detach else base_model.vertices
    int_vertices  = vertices_full[: base_model.num_int_verts]
    ext_vertices  = base_model.ext_vertices.detach() if detach else base_model.ext_vertices
    indices       = base_model.indices.detach() if detach else base_model.indices

    d_list, rgb_list, grd_list, sh_list = [], [], [], []
    for start in range(0, indices.shape[0], chunk_size):
        end = min(start + chunk_size, indices.shape[0])
        _, _, density, rgb, grd, sh = base_model.compute_batch_features(
            vertices_full, indices, start, end
        )
        d_list.append(density)
        rgb_list.append(rgb)
        grd_list.append(grd[:, 0:1])
        sh_list.append(sh)

    density  = torch.cat(d_list, 0)
    rgb      = torch.cat(rgb_list, 0)
    gradient = torch.cat(grd_list, 0)
    sh       = torch.cat(sh_list, 0)

    if detach:
        density, rgb, gradient, sh = (x.clone().detach() for x in (density, rgb, gradient, sh))

    return FrozenTetModel(
        int_vertices=int_vertices.to(device),
        ext_vertices=ext_vertices.to(device),
        indices=indices.to(device),
        density=density.to(device),
        rgb=rgb.to(device),
        gradient=gradient.to(device),
        sh=sh.to(device),
        center=base_model.center.detach().to(device),
        scene_scaling=base_model.scene_scaling.detach().to(device),
        density_offset=base_model.density_offset,
        current_sh_deg=base_model.current_sh_deg,
        max_sh_deg=base_model.max_sh_deg,
        chunk_size=chunk_size,
    )


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
        density_lr:   float = 1e-3,
        color_lr:     float = 1e-3,
        gradient_lr:  float = 1e-3,
        sh_lr:        float = 1e-3,
        weight_decay: float = 1e-10,
        lambda_tv:    float = 0.0,
        lambda_density: float = 0.0,
    ) -> None:
        self.model = model
        self.weight_decay   = weight_decay
        self.lambda_tv      = lambda_tv
        self.lambda_density = lambda_density

        # ------------------------------------------------------------------
        # single optimiser with four parameter groups
        # ------------------------------------------------------------------
        self.optim = optim.CustomAdam([
            {"params": [model.density],  "lr": density_lr,  "name": "density"},
            {"params": [model.rgb],      "lr": color_lr,    "name": "color"},
            {"params": [model.gradient], "lr": gradient_lr, "name": "gradient"},
            {"params": [model.sh],       "lr": sh_lr,       "name": "sh"},
        ], betas=[0.9, 0.99])

        # alias for external training scripts that expected these names
        self.net_optim   = self.optim
        self.vertex_optim = None  # geometry is frozen

        # TV structure (pairs & face areas) for regulariser ----------------
        if self.lambda_tv > 0:
            self.pairs, self.face_area = build_tv_struct(
                self.model.vertices.detach(), self.model.indices, device=model.device
            )

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

    @property
    def sh_optim(self):
        return None

    # no LR schedule by default; hook for user logic --------------------
    def update_learning_rate(self, *_, **__):
        pass

    @torch.no_grad()
    def split(self, clone_indices, split_point, split_mode):
        print("Split called on frozen optimizer")

    # ------------------------------------------------------------------
    # regularisers ------------------------------------------------------
    # ------------------------------------------------------------------
    def regularizer(self, *_):
        wd_loss = self.weight_decay * sum((p ** 2).mean() for p in [
            self.model.density, self.model.rgb, self.model.gradient, self.model.sh
        ])

        if self.lambda_density > 0:
            density = self.model.density.squeeze(-1)
            density_loss = density.mean()
        else:
            density_loss = 0.0

        if self.lambda_tv > 0:
            # simple TV on densities as example
            diff = (self.model.density[self.pairs[:, 0]] - self.model.density[self.pairs[:, 1]]).abs()
            tv_loss = (self.face_area * diff).sum() / self.face_area.sum()
        else:
            tv_loss = 0.0

        return wd_loss + self.lambda_density * density_loss + self.lambda_tv * tv_loss

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
    base_model: "Model",
    *,
    density_lr:   float = 1e-4,
    color_lr:     float = 1e-4,
    gradient_lr:  float = 1e-4,
    sh_lr:        float = 1e-4,
    weight_decay: float = 1e-10,
    lambda_tv:    float = 0.0,
    lambda_density: float = 0.0,
    detach: bool = True,
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
    frozen_model = bake_from_model(base_model, detach=detach, chunk_size=chunk_size)

    frozen_optim = FrozenTetOptimizer(
        frozen_model,
        density_lr=density_lr,
        color_lr=color_lr,
        gradient_lr=gradient_lr,
        sh_lr=sh_lr,
        weight_decay=weight_decay,
        lambda_tv=lambda_tv,
        lambda_density=lambda_density,
    )

    # free GPU memory used by the big backbone (optional but handy)
    _offload_model_to_cpu(base_model)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    return frozen_model, frozen_optim
