import torch
import math
from data.camera import Camera
from utils import optim
from gdel3d import Del
from torch import nn
from icecream import ic

from utils.topo_utils import (
    calculate_circumcenters_torch,
    fibonacci_spiral_on_sphere,
    get_tet_adjacency,
    build_adj,
)
from utils import topo_utils
from utils.train_util import get_expon_lr_func, SpikingLR
from utils.graphics_utils import l2_normalize_th
from pathlib import Path
import numpy as np
from utils.args import Args
from scipy.spatial import Delaunay
import open3d as o3d
from utils.model_util import *
from models.base_model import BaseModel
from utils.safe_math import safe_exp, safe_div


def project_points_to_tetrahedra(points, tets):
    """Project points onto tetrahedra by clamping negative barycentrics.

    Args:
        points: (N, 3) points to project (e.g. circumcenters).
        tets: (N, 4, 3) tetrahedron vertices.

    Returns:
        (N, 3) projected points guaranteed to lie inside (or on) the tet.
    """
    v0 = tets[:, 0, :]
    T = (tets[:, 1:, :] - v0.unsqueeze(1)).permute(0, 2, 1)  # (N, 3, 3)

    x = torch.linalg.solve(T, (points - v0).unsqueeze(2)).squeeze(2)  # (N, 3)

    w0 = 1 - x.sum(dim=1, keepdim=True)  # (N, 1)
    bary = torch.cat([w0, x], dim=1)  # (N, 4)
    bary = bary.clamp(min=0)

    norm = bary.sum(dim=1, keepdim=True).clamp(min=1e-8)
    mask = (norm > 1).reshape(-1)
    bary[mask] = bary[mask] / norm[mask]

    p_proj = (T * bary[:, 1:].unsqueeze(1)).sum(dim=2) + v0
    return p_proj


class Model(BaseModel):
    def __init__(
        self,
        vertices: torch.Tensor,
        ext_vertices: torch.Tensor,
        center: torch.Tensor,
        scene_scaling: float,
        density_offset=-1,
        current_sh_deg=2,
        max_sh_deg=2,
        additional_attr=0,
        ablate_gradient=False,
        kernel_sigma=1.0,
        project_cc=False,
        **kwargs,
    ):
        super().__init__()
        self.device = vertices.device
        self.density_offset = density_offset
        self.max_sh_deg = max_sh_deg
        self.current_sh_deg = current_sh_deg
        self.sh_dim = ((1 + max_sh_deg) ** 2 - 1) * 3
        self.additional_attr = additional_attr
        self.ablate_gradient = ablate_gradient
        self.kernel_sigma = kernel_sigma
        self.project_cc = project_cc

        # Per-vertex value dimension:
        # [0:1] density, [1:4] rgb, [4:7] gradient dir,
        # [7:7+sh_dim] SH coeffs, [7+sh_dim:] additional attrs
        sh_dim_total = self.sh_dim  # already multiplied by 3
        self.value_dim = 7 + sh_dim_total + additional_attr

        self.chunk_size = 308576
        self.mask_values = True
        self.frozen = False
        self.linear = False
        self.feature_dim = 7 + additional_attr
        self.alpha = 0
        self.compile = False  # no hash grid to compile

        self.register_buffer("ext_vertices", ext_vertices.to(self.device))
        self.register_buffer("center", center.reshape(1, 3))
        self.register_buffer(
            "scene_scaling",
            torch.tensor(float(scene_scaling), device=self.device),
        )
        self.interior_vertices = nn.Parameter(vertices.detach())

        # Per-vertex learnable values
        n_int = vertices.shape[0]
        self.interior_vertex_values = nn.Parameter(
            torch.randn(n_int, self.value_dim, device=self.device) * 0.01
        )

        n_ext = ext_vertices.shape[0]
        self.register_buffer(
            "ext_vertex_values",
            torch.zeros(n_ext, self.value_dim, device=self.device),
        )

        self.update_triangulation()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_int_verts(self):
        return self.interior_vertices.shape[0]

    @property
    def vertices(self):
        return torch.cat([self.interior_vertices, self.ext_vertices])

    @property
    def vertex_values(self):
        return torch.cat([self.interior_vertex_values, self.ext_vertex_values])

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------

    def _compute_flap_indices(self):
        """Compute the flap vertex for each tet/face pair.

        For tet t, face j, if there is a neighbor tet n = tet_adj[t, j],
        the flap vertex is the vertex of n that is NOT shared with t.
        Result shape: (T, 4), with -1 for boundary faces.
        """
        T = self.indices.shape[0]
        device = self.indices.device
        flap = torch.full((T, 4), -1, dtype=torch.long, device=device)

        for j in range(4):
            neigh = self.tet_adj[:, j]  # (T,)
            has_neigh = neigh >= 0
            if not has_neigh.any():
                continue
            # Own vertex set for tets with this neighbor
            own_verts = self.indices[has_neigh]  # (K, 4)
            neigh_verts = self.indices[neigh[has_neigh]]  # (K, 4)
            # For each neighbor tet, find the vertex not in own_verts
            # own_verts: (K, 4), neigh_verts: (K, 4)
            # Compare each neighbor vertex against all own vertices
            # not_shared[k, i] = True if neigh_verts[k, i] not in own_verts[k]
            not_shared = (
                neigh_verts.unsqueeze(2) != own_verts.unsqueeze(1)
            ).all(
                dim=2
            )  # (K, 4)
            # Pick the first True column per row
            # If multiple (shouldn't happen for valid meshes), take first
            flap_local = not_shared.float().argmax(dim=1)  # (K,)
            flap_vertex = neigh_verts[
                torch.arange(neigh_verts.shape[0], device=device), flap_local
            ]
            flap[has_neigh, j] = flap_vertex.long()

        self.flap_indices = flap

    # ------------------------------------------------------------------
    # Core shell kernel computation
    # ------------------------------------------------------------------

    def compute_batch_features(
        self, vertices, indices, start, end, circumcenters=None,
        flap_indices=None,
    ):
        """Shell kernel: interpolate per-vertex values using Gaussian weights."""
        batch_indices = indices[start:end]  # (B, 4)
        B = batch_indices.shape[0]
        device = vertices.device

        # 1. Gather own 4 vertex positions + values
        own_pos = vertices[batch_indices]  # (B, 4, 3)
        all_values = self.vertex_values
        own_val = all_values[batch_indices]  # (B, 4, D)

        # 2. Compute circumcenters + circumradii
        if circumcenters is None:
            cc, R = calculate_circumcenters_torch(own_pos.double())
            cc = cc.float()
            R = R.float()
        else:
            cc = circumcenters[start:end]
            R = torch.linalg.norm(
                cc - vertices[batch_indices[:, 0]], dim=-1
            )

        # Optionally project circumcenter into the tet so the kernel
        # center always lies inside, giving distinct per-vertex distances.
        kernel_center = cc
        if self.project_cc:
            kernel_center = project_points_to_tetrahedra(cc, own_pos)

        # 3. Gather flap vertices (up to 4 neighbors)
        all_flap = flap_indices if flap_indices is not None else self.flap_indices
        batch_flap = all_flap[start:end]  # (B, 4)
        flap_valid = batch_flap >= 0  # (B, 4)
        safe_flap = batch_flap.clamp(min=0)  # safe indexing
        flap_pos = vertices[safe_flap]  # (B, 4, 3)
        flap_val = all_values[safe_flap]  # (B, 4, D)

        # 4. Distances from all 8 vertices to kernel center
        cc_exp = kernel_center.unsqueeze(1)  # (B, 1, 3)
        own_dist = torch.linalg.norm(own_pos - cc_exp, dim=-1)  # (B, 4)
        flap_dist = torch.linalg.norm(flap_pos - cc_exp, dim=-1)  # (B, 4)

        # 5. Gaussian kernel weights: w = safe_exp(-(d/R)^2 / (2*kernel_sigma^2))
        R_safe = R.unsqueeze(1).clamp(min=1e-8)  # (B, 1)
        inv_2sigma2 = 1.0 / (2.0 * self.kernel_sigma * self.kernel_sigma)
        own_w = safe_exp(-(own_dist / R_safe) ** 2 * inv_2sigma2)  # (B, 4)
        flap_w = safe_exp(-(flap_dist / R_safe) ** 2 * inv_2sigma2)  # (B, 4)

        # 6. Zero out weights for boundary flaps
        flap_w = flap_w * flap_valid.float()

        # 7. Concatenate and normalize weights, compute weighted average
        all_w = torch.cat([own_w, flap_w], dim=1)  # (B, 8)
        # w_sum = all_w.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        # all_w = all_w / w_sum  # (B, 8)

        all_val = torch.cat([own_val, flap_val], dim=1)  # (B, 8, D)
        raw = (all_w.unsqueeze(2) * all_val).sum(dim=1)  # (B, D)

        # 8. Unpack and activate
        sh_dim_per_channel = ((1 + self.max_sh_deg) ** 2 - 1)
        sh_dim_total = self.sh_dim  # sh_dim_per_channel * 3

        raw_density = raw[:, 0:1]
        raw_rgb = raw[:, 1:4]
        raw_grd = raw[:, 4:7]
        raw_sh = raw[:, 7 : 7 + sh_dim_total]
        raw_attr = raw[:, 7 + sh_dim_total :]

        # Density: safe_exp(raw + offset)
        density = safe_exp(raw_density + self.density_offset)

        # RGB base color
        rgb = raw_rgb + 0.5

        # Gradient direction: grd / sqrt(grd^2 + 1)
        grd = raw_grd.reshape(-1, 1, 3)
        grd = grd / ((grd * grd).sum(dim=-1, keepdim=True) + 1).sqrt()

        # SH coefficients
        sh = raw_sh.half().reshape(B, sh_dim_per_channel, 3)

        # Additional attributes
        if self.additional_attr > 0:
            attr = raw_attr
        else:
            attr = torch.empty((B, 0), device=device)

        return cc, density, rgb, grd, sh, attr

    # ------------------------------------------------------------------
    # Cell values (same chunked loop as ingp_color)
    # ------------------------------------------------------------------

    def get_cell_values(
        self, camera: Camera, mask=None, all_circumcenters=None, radii=None
    ):
        if mask is not None:
            indices = self.indices[mask]
            flap = self.flap_indices[mask]
        else:
            indices = self.indices
            flap = self.flap_indices
        vertices = self.vertices

        sh_dim = (self.max_sh_deg + 1) ** 2 - 1
        features = torch.empty(
            (indices.shape[0], self.feature_dim), device=self.device
        )
        shs = torch.empty(
            (indices.shape[0], sh_dim, 3), device=self.device
        )
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])
            circumcenters, density, rgb, grd, sh, attr = (
                self.compute_batch_features(
                    vertices,
                    indices,
                    start,
                    end,
                    circumcenters=all_circumcenters,
                    flap_indices=flap,
                )
            )
            if self.ablate_gradient:
                grd = torch.zeros_like(grd)
            centroids = vertices[indices[start:end]].mean(dim=1)
            sh_r = sh.reshape(end - start, sh_dim, 3)
            shs[start:end] = sh_r
            dvrgbs = activate_output(
                camera.camera_center.to(self.device),
                density,
                rgb,
                grd,
                sh_r,
                attr,
                indices[start:end],
                centroids,
                vertices.detach(),
                self.current_sh_deg,
                self.max_sh_deg,
            )
            features[start:end] = dvrgbs
        return shs, features

    # ------------------------------------------------------------------
    # Circumcenter helper
    # ------------------------------------------------------------------

    def get_circumcenters(self):
        circumcenter = pre_calc_cell_values(
            self.vertices, self.indices, self.center, self.scene_scaling
        )
        return circumcenter

    # ------------------------------------------------------------------
    # Init from point cloud
    # ------------------------------------------------------------------

    @staticmethod
    def init_from_pcd(
        point_cloud, cameras, device, max_sh_deg, voxel_size=0.00, **kwargs
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
        ext_vertices = fibonacci_spiral_on_sphere(
            num_ext, new_radius, device="cpu"
        ) + center.reshape(1, 3).cpu()

        vertices = torch.cat([vertices, ext_vertices], dim=0)
        ext_vertices = torch.empty((0, 3))

        model = Model(
            vertices.cuda(),
            ext_vertices,
            center,
            scaling,
            max_sh_deg=max_sh_deg,
            **kwargs,
        )
        return model

    @staticmethod
    def init_rand_from_pcd(
        point_cloud, cameras, device, max_sh_deg, num_points=10000, **kwargs
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
        pcd_scaling = torch.linalg.norm(
            vertices - center.cpu().reshape(1, 3), dim=1, ord=2
        ).max()
        new_radius = pcd_scaling.cpu().item()

        vertices = topo_utils.sample_uniform_in_sphere(
            num_points, 3, base_radius=0, radius=new_radius, device="cpu"
        ) + center.reshape(1, 3).cpu()

        num_ext = 1000
        ext_vertices = fibonacci_spiral_on_sphere(
            num_ext, new_radius, device="cpu"
        ) + center.reshape(1, 3).cpu()

        model = Model(
            vertices.cuda(),
            ext_vertices,
            center,
            scaling,
            max_sh_deg=max_sh_deg,
            **kwargs,
        )
        return model

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_ckpt(path: Path, device):
        ckpt_path = path / "ckpt.pth"
        config_path = path / "config.json"
        config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)
        vertices = ckpt["interior_vertices"]
        indices = ckpt["indices"]
        del ckpt["indices"]
        if "empty_indices" in ckpt:
            empty_indices = ckpt["empty_indices"]
            del ckpt["empty_indices"]
        else:
            empty_indices = torch.empty(
                (0, 4), dtype=indices.dtype, device=indices.device
            )
        print(f"Loaded {vertices.shape[0]} vertices")
        ext_vertices = ckpt["ext_vertices"]
        model = Model(
            vertices.to(device),
            ext_vertices,
            ckpt["center"],
            ckpt["scene_scaling"],
            **config.as_dict(),
        )
        model.load_state_dict(ckpt)
        model.min_t = getattr(config, "min_t", 0.2)
        model.indices = torch.as_tensor(indices).cuda()
        model.empty_indices = torch.as_tensor(empty_indices).cuda()
        # Recompute topology
        model.tet_adj = build_adj(model.vertices, model.indices, device=model.device)
        model._compute_flap_indices()
        return model

    # ------------------------------------------------------------------
    # Density / feature helpers (same as ingp_color)
    # ------------------------------------------------------------------

    def calc_tet_density(self):
        densities = []
        verts = self.vertices
        for start in range(0, self.indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.indices.shape[0])
            _, density, _, _, _, _ = self.compute_batch_features(
                verts, self.indices, start, end
            )
            densities.append(density.reshape(-1))
        return torch.cat(densities)

    def compute_features(self, offset=False):
        vertices = self.vertices
        indices = self.indices
        cs, ds, rs, gs, ss = [], [], [], [], []
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])
            circumcenters, density, rgb, grd, sh, _ = (
                self.compute_batch_features(vertices, indices, start, end)
            )
            tets = vertices[indices[start:end]]
            cs.append(circumcenters)
            ds.append(density)
            ss.append(sh)
            if offset:
                base_color_v0_raw, normed_grd = offset_normalize(
                    rgb, grd, circumcenters, tets
                )
                rs.append(rgb)
                gs.append(normed_grd)
            else:
                rs.append(rgb)
                gs.append(grd)
        cs = torch.cat(cs, dim=0)
        ds = torch.cat(ds, dim=0)
        rs = torch.cat(rs, dim=0)
        gs = torch.cat(gs, dim=0)
        ss = torch.cat(ss, dim=0)
        return cs, ds, rs, gs, ss

    # ------------------------------------------------------------------
    # SH / adjacency
    # ------------------------------------------------------------------

    def sh_up(self):
        self.current_sh_deg = min(self.max_sh_deg, self.current_sh_deg + 1)

    def compute_adjacency(self):
        self.faces, self.side_index = get_tet_adjacency(self.indices)

    # ------------------------------------------------------------------
    # Triangulation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_triangulation(
        self,
        high_precision=False,
        density_threshold=0.0,
        alpha_threshold=0.0,
    ):
        torch.cuda.empty_cache()
        verts = self.vertices
        if high_precision:
            d = Delaunay(verts.detach().cpu().numpy())
            indices_np = d.simplices.astype(np.int32)
        else:
            v = Del(verts.shape[0])
            indices_np, prev = v.compute(verts.detach().cpu().double())
            indices_np = indices_np.clone().numpy()
            valid_mask = (indices_np >= 0) & (indices_np < verts.shape[0])
            indices_np = indices_np[valid_mask.all(axis=1)]
            del prev

        # Ensure positive volume
        indices = torch.as_tensor(indices_np).cuda()
        vols = topo_utils.tet_volumes(verts[indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            indices[reverse_mask] = indices[reverse_mask][:, [1, 0, 2, 3]]

        self.indices = indices

        if density_threshold > 0 or alpha_threshold > 0:
            tet_density = self.calc_tet_density()
            tet_alpha = self.calc_tet_alpha(mode="min", density=tet_density)
            mask = (tet_density > density_threshold) | (
                tet_alpha > alpha_threshold
            )
            self.empty_indices = self.indices[~mask]
            self.indices = self.indices[mask]
            self.mask = mask
        else:
            self.empty_indices = torch.empty(
                (0, 4), dtype=self.indices.dtype, device="cuda"
            )

        # Build adjacency + flap indices
        self.tet_adj = build_adj(
            self.vertices, self.indices, device=self.device
        )
        self._compute_flap_indices()

        torch.cuda.empty_cache()

    def __len__(self):
        return self.vertices.shape[0]

    # ------------------------------------------------------------------
    # Weight decay (L2 on vertex values)
    # ------------------------------------------------------------------

    def compute_weight_decay(self):
        return (self.interior_vertex_values ** 2).mean()

    @torch.no_grad()
    def log_kernel_diagnostics(self):
        """Compute and return shell kernel diagnostics for investigation."""
        vertices = self.vertices.detach()
        n_verts = vertices.shape[0]
        weight_sums = torch.zeros(n_verts, device=self.device)
        tet_count = torch.zeros(n_verts, dtype=torch.long, device=self.device)
        all_own_w, all_flap_w = [], []
        cc_outside_count = 0
        total_tets = 0

        for start in range(0, self.indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.indices.shape[0])
            batch_indices = self.indices[start:end]
            B = batch_indices.shape[0]
            total_tets += B
            own_pos = vertices[batch_indices]  # (B, 4, 3)

            cc, R = calculate_circumcenters_torch(own_pos.double())
            cc = cc.float()
            R = R.float()

            if self.project_cc:
                kernel_center = project_points_to_tetrahedra(cc, own_pos)
            else:
                kernel_center = cc

            # Check how many circumcenters are outside the tet
            v0 = own_pos[:, 0, :]
            T_mat = (own_pos[:, 1:, :] - v0.unsqueeze(1)).permute(0, 2, 1)
            x = torch.linalg.solve(T_mat, (cc - v0).unsqueeze(2)).squeeze(2)
            w0 = 1 - x.sum(dim=1)
            bary = torch.cat([w0.unsqueeze(1), x], dim=1)
            cc_outside_count += (bary.min(dim=1).values < -1e-6).sum().item()

            batch_flap = self.flap_indices[start:end]
            flap_valid = batch_flap >= 0
            safe_flap = batch_flap.clamp(min=0)
            flap_pos = vertices[safe_flap]

            cc_exp = kernel_center.unsqueeze(1)
            own_dist = torch.linalg.norm(own_pos - cc_exp, dim=-1)
            flap_dist = torch.linalg.norm(flap_pos - cc_exp, dim=-1)

            R_safe = R.unsqueeze(1).clamp(min=1e-8)
            inv_2sigma2 = 1.0 / (2.0 * self.kernel_sigma * self.kernel_sigma)
            own_w = safe_exp(-(own_dist / R_safe) ** 2 * inv_2sigma2)
            flap_w = safe_exp(-(flap_dist / R_safe) ** 2 * inv_2sigma2)
            flap_w = flap_w * flap_valid.float()

            all_own_w.append(own_w)
            all_flap_w.append(flap_w)

            # Accumulate per-vertex weight sums and tet counts
            for j in range(4):
                weight_sums.scatter_add_(0, batch_indices[:, j], own_w[:, j])
                tet_count.scatter_add_(0, batch_indices[:, j],
                                       torch.ones(B, dtype=torch.long, device=self.device))
            for j in range(4):
                valid = flap_valid[:, j]
                if valid.any():
                    weight_sums.scatter_add_(0, safe_flap[valid, j], flap_w[valid, j])

        all_own_w = torch.cat(all_own_w, dim=0)
        all_flap_w = torch.cat(all_flap_w, dim=0)

        # Per-tet weight stats
        tet_total_w = all_own_w.sum(dim=1) + all_flap_w.sum(dim=1)
        own_frac = all_own_w.sum(dim=1) / tet_total_w.clamp(min=1e-8)

        # Weight entropy per tet (how uniform is the weighting)
        all_w_cat = torch.cat([all_own_w, all_flap_w], dim=1)
        w_norm = all_w_cat / all_w_cat.sum(dim=1, keepdim=True).clamp(min=1e-8)
        entropy = -(w_norm * (w_norm + 1e-10).log()).sum(dim=1)
        max_entropy = math.log(8)  # uniform over 8 vertices

        # Own weight variance (how much the kernel differentiates own vertices)
        own_w_std = all_own_w.std(dim=1)

        # Per-vertex stats (training speed imbalance)
        int_ws = weight_sums[:self.num_int_verts]
        int_tc = tet_count[:self.num_int_verts].float()

        # Gradient magnitudes
        val_grad = self.interior_vertex_values.grad
        pos_grad = self.interior_vertices.grad

        diag = {
            "cc_outside_pct": 100.0 * cc_outside_count / max(total_tets, 1),
            "own_w_mean": all_own_w.mean().item(),
            "own_w_std_across_tet": own_w_std.mean().item(),
            "flap_w_mean": all_flap_w[all_flap_w > 0].mean().item() if (all_flap_w > 0).any() else 0,
            "own_frac_mean": own_frac.mean().item(),
            "entropy_mean": entropy.mean().item(),
            "entropy_ratio": (entropy.mean().item() / max_entropy),
            "vtx_weight_sum_mean": int_ws.mean().item(),
            "vtx_weight_sum_std": int_ws.std().item(),
            "vtx_weight_sum_min": int_ws.min().item(),
            "vtx_weight_sum_max": int_ws.max().item(),
            "vtx_tet_count_mean": int_tc.mean().item(),
            "vtx_tet_count_std": int_tc.std().item(),
            "vtx_tet_count_min": int_tc.min().item(),
            "vtx_tet_count_max": int_tc.max().item(),
        }
        if val_grad is not None:
            vg = val_grad.norm(dim=1)
            diag["val_grad_mean"] = vg.mean().item()
            diag["val_grad_std"] = vg.std().item()
            diag["val_grad_max"] = vg.max().item()
        if pos_grad is not None:
            pg = pos_grad.norm(dim=1)
            diag["pos_grad_mean"] = pg.mean().item()
            diag["pos_grad_std"] = pg.std().item()
            diag["pos_grad_max"] = pg.max().item()

        return diag


class TetOptimizer:
    def __init__(
        self,
        model: Model,
        values_lr: float = 5e-3,
        final_values_lr: float = 5e-4,
        vertices_lr: float = 4e-4,
        final_vertices_lr: float = 4e-7,
        vertices_lr_delay_multi: float = 0.01,
        lr_delay: int = 500,
        freeze_start: int = 10000,
        vert_lr_delay: int = 500,
        spike_duration: int = 20,
        densify_interval: int = 500,
        densify_end: int = 15000,
        midpoint: int = 2000,
        percent_alpha: float = 0.02,
        # Accept and ignore ingp_color-specific kwargs
        encoding_lr: float = 1e-2,
        final_encoding_lr: float = 1e-2,
        network_lr: float = 1e-3,
        final_network_lr: float = 1e-3,
        **kwargs,
    ):
        # Use encoding_lr/network_lr as fallback for values_lr if not explicitly changed
        effective_values_lr = values_lr
        effective_final_values_lr = final_values_lr

        self.optim = optim.CustomAdam(
            [
                {
                    "params": [model.interior_vertex_values],
                    "lr": effective_values_lr,
                    "name": "interior_vertex_values",
                },
            ],
            ignore_param_list=[],
            betas=[0.9, 0.999],
            eps=1e-15,
        )
        self.vert_lr_multi = float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam(
            [
                {
                    "params": [model.interior_vertices],
                    "lr": self.vert_lr_multi * vertices_lr,
                    "name": "interior_vertices",
                },
            ],
        )
        self.model = model

        self.alpha_sched = get_expon_lr_func(
            lr_init=percent_alpha * float(model.scene_scaling.cpu()),
            lr_final=1e-20,
            lr_delay_mult=1e-8,
            lr_delay_steps=0,
            max_steps=freeze_start // 3,
        )

        base_values_scheduler = get_expon_lr_func(
            lr_init=effective_values_lr,
            lr_final=effective_final_values_lr,
            lr_delay_mult=1e-8,
            lr_delay_steps=lr_delay,
            max_steps=freeze_start,
        )
        self.values_scheduler_args = SpikingLR(
            spike_duration,
            freeze_start,
            base_values_scheduler,
            midpoint,
            densify_interval,
            densify_end,
            effective_values_lr,
            effective_values_lr,
        )

        self.vertices_lr = self.vert_lr_multi * vertices_lr
        self.final_vertices_lr = self.vert_lr_multi * final_vertices_lr
        base_vertex_scheduler = get_expon_lr_func(
            lr_init=self.vertices_lr,
            lr_final=self.final_vertices_lr,
            lr_delay_mult=vertices_lr_delay_multi,
            max_steps=freeze_start,
            lr_delay_steps=vert_lr_delay,
        )
        self.vertex_scheduler_args = SpikingLR(
            spike_duration,
            freeze_start,
            base_vertex_scheduler,
            midpoint,
            densify_interval,
            densify_end,
            self.vertices_lr,
            self.vertices_lr,
        )
        self.iteration = 0

    def update_learning_rate(self, iteration):
        self.iteration = iteration
        self.model.alpha = self.alpha_sched(iteration)
        for param_group in self.optim.param_groups:
            if param_group["name"] == "interior_vertex_values":
                lr = self.values_scheduler_args(iteration)
                param_group["lr"] = lr
        for param_group in self.vertex_optim.param_groups:
            if param_group["name"] == "interior_vertices":
                lr = self.vertex_scheduler_args(iteration)
                self.vertices_lr = lr
                param_group["lr"] = lr

    def add_points(self, new_verts: torch.Tensor, raw_verts=False):
        n_new = new_verts.shape[0]
        new_values = torch.zeros(
            n_new, self.model.value_dim, device=new_verts.device
        )
        self.model.interior_vertices = (
            self.vertex_optim.cat_tensors_to_optimizer(
                dict(interior_vertices=new_verts)
            )["interior_vertices"]
        )
        self.model.interior_vertex_values = (
            self.optim.cat_tensors_to_optimizer(
                dict(interior_vertex_values=new_values)
            )["interior_vertex_values"]
        )
        self.model.update_triangulation()

    def remove_points(self, keep_mask: torch.Tensor):
        keep_mask = keep_mask[: self.model.interior_vertices.shape[0]]
        self.model.interior_vertices = (
            self.vertex_optim.prune_optimizer(keep_mask)["interior_vertices"]
        )
        self.model.interior_vertex_values = (
            self.optim.prune_optimizer(keep_mask)["interior_vertex_values"]
        )
        self.model.update_triangulation()

    @torch.no_grad()
    def split(self, split_point, **kwargs):
        self.add_points(split_point)

    def main_step(self):
        self.optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad(set_to_none=True)

    @property
    def sh_optim(self):
        return None

    def regularizer(self, render_pkg, lambda_weight_decay, **kwargs):
        weight_decay = lambda_weight_decay * self.model.compute_weight_decay()
        return weight_decay

    def update_triangulation(self, **kwargs):
        self.model.update_triangulation(**kwargs)

    def prune(self, diff_threshold, **kwargs):
        if diff_threshold <= 0:
            return
        density = self.model.calc_tet_density()
        owners, face_area = topo_utils.build_tv_struct(
            self.model.vertices, self.model.indices, device=self.model.device
        )
        diff = density[owners[:, 0]] - density[owners[:, 1]]
        tet_diff = (face_area * diff.abs()).reshape(-1)

        indices = self.model.indices.long()
        device = indices.device
        vert_diff = torch.zeros(
            (self.model.vertices.shape[0],), device=device
        )
        reduce_type = "amax"
        vert_diff.scatter_reduce_(
            dim=0, index=indices[..., 0], src=tet_diff, reduce=reduce_type
        )
        vert_diff.scatter_reduce_(
            dim=0, index=indices[..., 1], src=tet_diff, reduce=reduce_type
        )
        vert_diff.scatter_reduce_(
            dim=0, index=indices[..., 2], src=tet_diff, reduce=reduce_type
        )
        vert_diff.scatter_reduce_(
            dim=0, index=indices[..., 3], src=tet_diff, reduce=reduce_type
        )
        keep_mask = vert_diff > diff_threshold
        print(
            f"Pruned {(~keep_mask).sum()} points. "
            f"VD: {vert_diff.mean()} TD: {tet_diff.mean()}"
        )
        self.remove_points(keep_mask)

    def clip_grad_norm_(self, max_norm):
        torch.nn.utils.clip_grad_norm_(
            self.model.interior_vertex_values, max_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.model.interior_vertices, max_norm
        )
