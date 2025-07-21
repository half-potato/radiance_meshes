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
from delaunay_rasterization.internal.render_err import render_err

class BaseModel(nn.Module):

    def __len__(self):
        return self.vertices.shape[0]

    @property
    def num_int_verts(self):
        return self.interior_vertices.shape[0]

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

    def calc_aniso_loss(self, density):
        verts = self.vertices
        inds = self.indices
        v0, v1, v2, v3 = verts[inds[:, 0]], verts[inds[:, 1]], verts[inds[:, 2]], verts[inds[:, 3]]
        
        edge_lengths = torch.stack([
            torch.norm(v0 - v1, dim=1), torch.norm(v0 - v2, dim=1), torch.norm(v0 - v3, dim=1),
            torch.norm(v1 - v2, dim=1), torch.norm(v1 - v3, dim=1), torch.norm(v2 - v3, dim=1)
        ], dim=0)
        alpha_min = 1 - torch.exp(-density.reshape(-1) * edge_lengths.min(dim=0)[0].reshape(-1))
        alpha_max = 1 - torch.exp(-density.reshape(-1) * edge_lengths.max(dim=0)[0].reshape(-1))
        return (safe_div((alpha_max-0.05).clip(min=0), alpha_min) - 1).clip(min=-10, max=10)

    def calc_tet_alpha(self, mode="min", density=None):
        verts = self.vertices
        inds = self.indices
        v0, v1, v2, v3 = verts[inds[:, 0]], verts[inds[:, 1]], verts[inds[:, 2]], verts[inds[:, 3]]
        
        edge_lengths = torch.stack([
            torch.norm(v0 - v1, dim=1), torch.norm(v0 - v2, dim=1), torch.norm(v0 - v3, dim=1),
            torch.norm(v1 - v2, dim=1), torch.norm(v1 - v3, dim=1), torch.norm(v2 - v3, dim=1)
        ], dim=0)
        if mode == "min":
            el = edge_lengths.min(dim=0)[0]
        elif mode == "second":
            el = edge_lengths.sort(dim=0).values[1, :]
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
        circumcenters, density, base_color_v0_raw, normed_grd, sh = self.compute_features(offset=False)
        radius = torch.linalg.norm(vertices[indices[:, 0]] - circumcenters.reshape(-1, 3), dim=-1)
        normed_grd = safe_div(normed_grd.reshape(-1, 3), radius.reshape(-1, 1))

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
        peak_contrib = torch.zeros((n_tets), device=self.device)

        for cam in cameras:
            target = cam.original_image.cuda()
            camera_center = cam.camera_center.to(self.device)

            image_votes, extras = render_err(
                target, cam, self,
                scene_scaling=self.scene_scaling,
                tile_size=tile_size,
                lambda_ssim=0
            )

            tc = extras["tet_count"]
            
            # --- Create a single mask for valid updates ---
            # Mask for tets that have a reasonable number of samples in the current view
            # --- Moments (s-1: sum of T, s1: sum of err, s2: sum of err^2)
            image_T, image_err, image_err1 = image_votes[:, 0], image_votes[:, 1], image_votes[:, 2]
            # total_T_p, image_err, image_err1 = image_votes[:, 3], image_votes[:, 4], image_votes[:, 5]
            _, image_Terr, image_ssim = image_votes[:, 2], image_votes[:, 4], image_votes[:, 5]
            N = tc
            peak_contrib = torch.maximum(image_T / N.clip(min=1), peak_contrib)
        tet_alpha = self.calc_tet_alpha(mode="min")
        mask = (peak_contrib > contrib_threshold) | (tet_alpha > alpha_threshold)

        verts = self.vertices
        # tet_density = self.calc_tet_density()
        # mask = (tet_density > density_threshold) | (tet_alpha > alpha_threshold)

        circumcenters, density, rgb, grd, sh = self.compute_features(offset=False)
        rgb = rgb[mask].detach()
        tets = verts[self.indices[mask]]
        circumcenters, radius = calculate_circumcenters_torch(tets.double())
        grd = grd[mask].detach()
        sh = sh[mask].detach()
        grd = grd.reshape(-1, 1, 3) * rgb.reshape(-1, 3, 1).mean(dim=1, keepdim=True).detach()
        normed_grd = safe_div(grd, radius.reshape(-1, 1, 1))
        tet_color_raw = eval_sh(
            tets.mean(dim=1).detach(),
            RGB2SH(rgb),
            sh.reshape(-1, (self.max_sh_deg+1)**2 - 1, 3).half(),
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

        meshes = mesh_util.extract_meshes_per_face_color(
        # meshes = mesh_util.extract_meshes(
            vcolors.detach().cpu().numpy(),
            verts.detach().cpu().numpy(),
            self.indices[mask].cpu().numpy())
        for i, mesh in enumerate(meshes):
            F = mesh['face']['vertex_indices'].shape[0]
            if F > 10000:
                mpath = path / f"{i}.ply"
                print(f"Saving #F:{F} to {mpath}")
                tinyplypy.write_ply(str(mpath), mesh, is_binary=False)


