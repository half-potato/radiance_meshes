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
from utils import topo_utils
from utils.model_util import activate_output
from utils import optim
from utils.model_util import *
from utils.safe_math import safe_log, safe_exp
from utils.train_util import get_expon_lr_func, SpikingLR
from utils import mesh_util
from utils.args import Args


class BaseModel(nn.Module):

    def get_cell_values(
        self,
        camera: Camera,
        mask: Optional[torch.Tensor] = None,
        all_circumcenters: Optional[torch.Tensor] = None,
        radii: Optional[torch.Tensor] = None,
    ):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices

        if self.chunk_size is None:
            cc, normalized, density, rgb, grd, sh = self.compute_batch_features(
                vertices, indices, start, end, circumcenters=all_circumcenters
            )
            cell_output = activate_output(
                camera.camera_center.to(self.device),
                density, rgb, grd, sh, indices,
                cc, vertices,
                self.max_sh_deg, self.max_sh_deg,
            )
            return normalized, cell_output
        else:
            outputs = []
            normed_cc = []
            start = 0
            for start in range(0, indices.shape[0], self.chunk_size):
                end = min(start + self.chunk_size, indices.shape[0])
                circumcenters, normalized, density, rgb, grd, sh = self.compute_batch_features(
                    vertices, indices, start, end, circumcenters=all_circumcenters)
                dvrgbs = activate_output(camera.camera_center.to(self.device),
                                         density, rgb, grd, sh, indices[start:end],
                                         circumcenters,
                                         vertices, self.current_sh_deg, self.max_sh_deg)
                normed_cc.append(normalized)
                outputs.append(dvrgbs)
            features = torch.cat(outputs, dim=0)
            normed_cc = torch.cat(normed_cc, dim=0)
            return normed_cc, features

    def __len__(self):
        return self.vertices.shape[0]

    @property
    def num_int_verts(self):
        return self.contracted_vertices.shape[0]

    def get_circumcenters(self):
        circumcenter =  pre_calc_cell_values(
            self.vertices, self.indices, self.center, self.scene_scaling)
        return circumcenter

    def calc_tet_area(self):
        verts = self.vertices
        v0, v1, v2, v3 = verts[self.indices].unbind(dim=1)
        mat = torch.stack([v1-v0, v2-v0, v3-v0], dim=-1)
        return torch.det(mat).abs() / 6.0

    def calc_vert_density(self):
        verts = self.vertices
        vertex_density = torch.zeros((verts.shape[0],), device=self.device)
        indices = self.indices.long()
        density = self.calc_tet_density()
        reduce_type = "amax"
        vertex_density.scatter_reduce_(dim=0, index=indices[..., 0], src=density, reduce=reduce_type)
        vertex_density.scatter_reduce_(dim=0, index=indices[..., 1], src=density, reduce=reduce_type)
        vertex_density.scatter_reduce_(dim=0, index=indices[..., 2], src=density, reduce=reduce_type)
        vertex_density.scatter_reduce_(dim=0, index=indices[..., 3], src=density, reduce=reduce_type)
        return vertex_density

    def calc_tet_alpha(self, mode="min", density=None):
        alpha_list = []
        start = 0
        
        verts = self.vertices
        inds = self.indices
        v0, v1, v2, v3 = verts[inds[:, 0]], verts[inds[:, 1]], verts[inds[:, 2]], verts[inds[:, 3]]
        
        edge_lengths = torch.stack([
            torch.norm(v0 - v1, dim=1), torch.norm(v0 - v2, dim=1), torch.norm(v0 - v3, dim=1),
            torch.norm(v1 - v2, dim=1), torch.norm(v1 - v3, dim=1), torch.norm(v2 - v3, dim=1)
        ], dim=0)
        if mode == "min":
            el = edge_lengths.min(dim=0)[0]
        elif mode == "max":
            el = edge_lengths.max(dim=0)[0]
        elif mode == "mean":
            el = edge_lengths.mean(dim=0)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'min', 'max', or 'mean'.")
        
        density = self.calc_tet_density() if density is None else density
        alpha = 1 - torch.exp(-density.reshape(-1) * el.reshape(-1))
        return alpha

    @torch.no_grad
    def save2ply(self, path):
        path.parent.mkdir(exist_ok=True, parents=True)

        xyz = self.vertices.detach().cpu().numpy().astype(np.float32)  # shape (num_vertices, 3)

        vertex_dict = {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
        }

        N = self.indices.shape[0]
        sh_dim = ((self.max_sh_deg+1)**2-1)

        vertices = self.vertices
        indices = self.indices
        circumcenters, density, base_color_v0_raw, normed_grd, sh = self.compute_features(offset=True)

        base_color_v0_raw = base_color_v0_raw.cpu().numpy().astype(np.float32)
        grds = normed_grd.reshape(-1, 3).cpu().numpy().astype(np.float32)
        densities = density.reshape(-1).cpu().numpy().astype(np.float32)
        sh_coeffs = sh.reshape(-1, sh_dim, 3).cpu().numpy().astype(np.float32)

        tetra_dict = {}
        tetra_dict["vertex_indices"] = self.indices.cpu().numpy().astype(np.int32)
        tetra_dict["s"] = np.ascontiguousarray(densities)
        for i, co in enumerate(["x", "y", "z"]):
            tetra_dict[f"grd_{co}"]         = np.ascontiguousarray(grds[:, i])

        sh_0 = RGB2SH(base_color_v0_raw)
        tetra_dict[f"sh_0_r"] = np.ascontiguousarray(sh_0[:, 0])
        tetra_dict[f"sh_0_g"] = np.ascontiguousarray(sh_0[:, 1])
        tetra_dict[f"sh_0_b"] = np.ascontiguousarray(sh_0[:, 2])
        for i in range(sh_coeffs.shape[1]):
            tetra_dict[f"sh_{i+1}_r"] = np.ascontiguousarray(sh_coeffs[:, i, 0])
            tetra_dict[f"sh_{i+1}_g"] = np.ascontiguousarray(sh_coeffs[:, i, 1])
            tetra_dict[f"sh_{i+1}_b"] = np.ascontiguousarray(sh_coeffs[:, i, 2])


        data_dict = {
            "vertex": vertex_dict,
            "tetrahedron": tetra_dict,
        }

        tinyplypy.write_ply(str(path), data_dict, is_binary=True)

    @torch.no_grad
    def extract_mesh(self, cameras, path, tile_size=16, contrib_threshold=0.1, density_threshold=0.5, alpha_threshold=0.2, **kwargs):
        path.mkdir(exist_ok=True, parents=True)
        n_tets = self.indices.shape[0]
        top1 = torch.zeros(n_tets, device=self.device)  # highest seen so far
        top2 = torch.zeros(n_tets, device=self.device)  # second-highest seen so far

        for cam in cameras:
            target = cam.original_image.cuda()
            camera_center = cam.camera_center.to(self.device)

            image_votes, extras = render_err(
                target, cam, self,
                scene_scaling=self.scene_scaling,
                tile_size=tile_size,
                lambda_ssim=0
            )

            tc = extras["tet_count"][..., 0]
            max_T = extras["tet_count"][..., 1].float() / 65535
            
            # --- Create a single mask for valid updates ---
            # Mask for tets that have a reasonable number of samples in the current view
            # --- Moments (s-1: sum of T, s1: sum of err, s2: sum of err^2)
            image_T, image_err, image_err1 = image_votes[:, 0], image_votes[:, 1], image_votes[:, 2]
            # total_T_p, image_err, image_err1 = image_votes[:, 3], image_votes[:, 4], image_votes[:, 5]
            _, image_Terr, image_ssim = image_votes[:, 2], image_votes[:, 4], image_votes[:, 5]
            N = tc

            # contrib = (image_T / N.clip(min=1)).reshape(-1)
            # contrib = (image_T / N.clip(min=1)).reshape(-1)
            contrib = max_T
            prev_top1 = top1
            top1 = torch.maximum(prev_top1, contrib)
            top2 = torch.maximum(top2, torch.minimum(prev_top1, contrib))

        inds = self.indices
        verts = self.vertices
        tets = verts[inds]
        circumcenters, tet_density, rgb, grd, sh = self.compute_features()
        # tet_alpha = self.calc_tet_alpha(mode="min")
        # tet_alpha = self.calc_tet_alpha(mode="min") * max_color.clip(min=1)

        v0, v1, v2, v3 = verts[inds[:, 0]], verts[inds[:, 1]], verts[inds[:, 2]], verts[inds[:, 3]]
        edge_lengths = torch.stack([
            torch.norm(v0 - v1, dim=1), torch.norm(v0 - v2, dim=1), torch.norm(v0 - v3, dim=1),
            torch.norm(v1 - v2, dim=1), torch.norm(v1 - v3, dim=1), torch.norm(v2 - v3, dim=1)
        ], dim=0)
        aspect = edge_lengths.min(dim=0).values / edge_lengths.max(dim=0).values 
        vol = topo_utils.tet_volumes(tets).abs()
        tet_alpha = 1 - torch.exp(-tet_density.reshape(-1) * vol.reshape(-1)**(1/3) * aspect)
        # mask = (peak_contrib > contrib_threshold)# | (tet_density > density_threshold)
        # mask = (peak_contrib > contrib_threshold) | (tet_alpha > alpha_threshold) | (tet_density > density_threshold)

        # mask = (peak_contrib > contrib_threshold) | (tet_alpha > alpha_threshold)
        mask = (top1.reshape(-1) > contrib_threshold) | (tet_density.reshape(-1) > density_threshold)
        ic(top1.mean(), tet_density.mean(), mask.sum())
        # mask = (tet_density > density_threshold) | (tet_alpha > alpha_threshold)

        rgb = rgb[mask].detach()
        tets = tets[mask]
        circumcenters, radius = calculate_circumcenters_torch(tets.double())
        grd = grd[mask].detach()
        sh = sh[mask].detach()
        normed_grd = grd.reshape(-1, 1, 3) * rgb.reshape(-1, 3, 1).mean(dim=1, keepdim=True).detach()
        tet_color_raw = eval_sh(
            tets.mean(dim=1).detach(),
            RGB2SH(rgb),
            sh.reshape(-1, (self.max_sh_deg+1)**2 - 1, 3),
            camera_center,
            self.max_sh_deg).float()
        tet_color = torch.nn.functional.softplus(tet_color_raw.reshape(-1, 3, 1), beta=10)
        vcolors = compute_vertex_colors_from_field(
            tets.detach(), tet_color.reshape(-1, 3), normed_grd.float(), circumcenters.float().detach()).clip(min=0)
        # vcolors = torch.nn.functional.softplus(vcolors, beta=10)

        # mesh_util.export_textured_meshes(path, 
        #     verts=verts.detach().cpu().numpy(),
        #     tets=self.indices[mask].cpu().numpy(),
        #     tet_v_rgb=vcolors.detach().cpu().numpy(),
        #     min_faces_for_export=10000,
        #     texture_size=4096*2**0,
        # )

        # meshes = mesh_util.extract_meshes_per_face_color(
        meshes = mesh_util.extract_meshes(
            vcolors.detach().cpu().numpy(),
            verts.detach().cpu().numpy(),
            self.indices[mask].cpu().numpy())
        for i, mesh in enumerate(meshes):
            F = mesh['face']['vertex_indices'].shape[0]
            if F > 1:
                mpath = path / f"{i}.ply"
                print(f"Saving #F:{F} to {mpath}")
                tinyplypy.write_ply(str(mpath), mesh, is_binary=False)
