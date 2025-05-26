import torch
import math
from data.camera import Camera
from utils import optim
from sh_slang.eval_sh import eval_sh
from gDel3D.build.gdel3d import Del
from torch import nn
from icecream import ic

from utils.topo_utils import calculate_circumcenters_torch, fibonacci_spiral_on_sphere, calc_barycentric, sample_uniform_in_sphere, project_points_to_tetrahedra, contraction_jacobian_d_in_chunks
from utils.safe_math import safe_exp, safe_div, safe_sqrt
from utils.contraction import contract_mean_std
from utils.contraction import contract_points, inv_contract_points

from utils.train_util import get_expon_lr_func, SpikingLR
from utils.graphics_utils import l2_normalize_th
from torch.utils.checkpoint import checkpoint
from pathlib import Path
import numpy as np
from utils.args import Args
import tinyplypy
from scipy.spatial import ConvexHull
from scipy.spatial import  Delaunay
import open3d as o3d
from data.types import BasicPointCloud
from simple_knn._C import distCUDA2
from utils import mesh_util
from utils.model_util import *

torch.set_float32_matmul_precision('high')


class Model(nn.Module):
    def __init__(self,
                 vertices: torch.Tensor,
                 ext_vertices: torch.Tensor,
                 center: torch.Tensor,
                 scene_scaling: float,
                 contract_vertices=True,
                 density_offset=-1,
                 current_sh_deg=2,
                 max_sh_deg=2,
                 **kwargs):
        super().__init__()
        self.device = vertices.device
        self.density_offset = density_offset
        self.max_sh_deg = max_sh_deg
        self.current_sh_deg = current_sh_deg
        self.dir_offset = torch.tensor([
            [0, 0],
            [math.pi, 0],
        ], device=self.device)
        sh_dim = ((1+max_sh_deg)**2-1)*3
        self.backbone = torch.compile(iNGPDW(sh_dim, **kwargs)).to(self.device)
        self.chunk_size = 408576
        self.mask_values = True
        self.frozen = False

        self.register_buffer('ext_vertices', ext_vertices.to(self.device))
        self.register_buffer('center', center.reshape(1, 3))
        self.register_buffer('scene_scaling', torch.tensor(float(scene_scaling), device=self.device))
        self.contract_vertices = contract_vertices
        if self.contract_vertices:
            self.contracted_vertices = nn.Parameter(self.contract(vertices.detach()))
        else:
            self.contracted_vertices = nn.Parameter(vertices.detach())
        self.update_triangulation()

    @property
    def num_int_verts(self):
        return self.contracted_vertices.shape[0]

    def get_circumcenters(self):
        circumcenter =  pre_calc_cell_values(
            self.vertices, self.indices, self.center, self.scene_scaling)
        return circumcenter

    def get_cell_values(self, camera: Camera, mask=None,
                        all_circumcenters=None, radii=None):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices

        # vertex_color_raw = eval_sh(
        #     vertices,
        #     torch.zeros((vertices.shape[0], 3), device=vertices.device),
        #     self.vertex_lights.reshape(-1, (self.max_sh_deg+1)**2 - 1, 3),
        #     camera.camera_center.to(self.device),
        #     self.current_sh_deg) - 0.5

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

    @staticmethod
    def init_from_pcd(point_cloud, cameras, device, max_sh_deg, ext_convex_hull,
                      voxel_size=0.00, init_repeat=3, **kwargs):
        torch.manual_seed(2)

        vertices = torch.as_tensor(point_cloud.points).float()

        dist = torch.clamp_min(distCUDA2(vertices.cuda()), 0.0000001).sqrt().cpu()

        vertices = vertices.reshape(-1, 1, 3).expand(-1, init_repeat, 3)
        vertices = vertices + torch.randn(*vertices.shape) * dist.reshape(-1, 1, 1).clip(min=0.01)
        vertices = vertices.reshape(-1, 3)

        # Convert BasicPointCloud to Open3D PointCloud
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(vertices.numpy())

        # Perform voxel downsampling
        if voxel_size > 0:
            o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)

        N = point_cloud.points.shape[0]
        vertices = torch.as_tensor(np.asarray(o3d_pcd.points)).float()
        vertices = vertices + torch.randn(*vertices.shape) * 1e-2

        ccenters = torch.stack([c.camera_center.reshape(3) for c in cameras], dim=0).to(device)
        center = ccenters.mean(dim=0)
        scaling = torch.linalg.norm(ccenters - center.reshape(1, 3), dim=1, ord=torch.inf).max()

        # vertices = vertices + torch.randn(*vertices.shape) * 1e-3
        # v = Del(vertices.shape[0])
        # indices_np, prev = v.compute(vertices.detach().cpu().double())
        # indices_np = indices_np.numpy()
        # indices_np = indices_np[(indices_np < vertices.shape[0]).all(axis=1)]
        # vertices = vertices[indices_np].mean(dim=1)
        # vertices = vertices + torch.randn(*vertices.shape) * 1e-3

        # add sphere
        pcd_scaling = torch.linalg.norm(vertices - center.cpu().reshape(1, 3), dim=1, ord=2).max()

        if ext_convex_hull:
            ext_vertices = expand_convex_hull(vertices, 5, device=vertices.device)
            num_ext = ext_vertices.shape[0]
        else:
            new_radius = pcd_scaling.cpu().item()
            within_sphere = sample_uniform_in_sphere(10000, 3, base_radius=new_radius, radius=new_radius, device='cpu') + center.reshape(1, 3).cpu()
            vertices = torch.cat([vertices, within_sphere], dim=0)
            num_ext = 1000
            ext_vertices = fibonacci_spiral_on_sphere(num_ext, new_radius, device='cpu') + center.reshape(1, 3).cpu()
        num_ext = ext_vertices.shape[0]

        # num_ext = 1000
        # ext_vertices = fibonacci_spiral_on_sphere(num_ext, 2* pcd_scaling.cpu(), device='cpu') + center.reshape(1, 3).cpu()

        vertices = nn.Parameter(vertices.cuda())
        model = Model(vertices, ext_vertices, center, scaling, max_sh_deg=max_sh_deg, **kwargs)
        return model

    def compute_batch_features(self, vertices, indices, start, end, circumcenters=None):
        if circumcenters is None:
            # circumcenter, cv, cr, normalized =  pre_calc_cell_values(
            #     vertices, indices[start:end], self.center, self.scene_scaling)
            tets = vertices[indices[start:end]]
            circumcenter, radius = calculate_circumcenters_torch(tets.double())
        else:
            circumcenter = circumcenters[start:end]
        normalized = (circumcenter - self.center) / self.scene_scaling
        radius = torch.linalg.norm(circumcenter - vertices[indices[start:end, 0]], dim=-1)
        cv, cr = contract_mean_std(normalized, radius / self.scene_scaling)
        x = (cv/2 + 1)/2
        output = checkpoint(self.backbone, x, cr, use_reentrant=True)
        return circumcenter, normalized, *output

    @staticmethod
    def load_ckpt(path: Path, device):
        # data_dict = tinyplypy.read_ply(str(path / "ckpt.ply"))
        # tet_data = data_dict["tetrahedron"]
        # indices = tet_data["vertex_indices"]  # shape (N,4)
        ckpt_path = path / "ckpt.pth"
        config_path = path / "alldata.json"
        config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)
        vertices = ckpt['contracted_vertices']
        indices = ckpt["indices"]  # shape (N,4)
        del ckpt["indices"]
        print(f"Loaded {vertices.shape[0]} vertices")
        temp = config.contract_vertices
        config.contract_vertices = False
        ext_vertices = ckpt['ext_vertices']
        model = Model(vertices.to(device), ext_vertices, ckpt['center'], ckpt['scene_scaling'], **config.as_dict())
        model.load_state_dict(ckpt)
        model.contract_vertices = temp
        model.min_t = model.scene_scaling * config.base_min_t
        model.indices = torch.as_tensor(indices).cuda()
        return model

    @torch.no_grad
    def perturb_vertices(self, perturbation):
        if self.contract_vertices:
            norm_verts = (self.vertices - self.center) / self.scene_scaling 
            J_d = contraction_jacobian_d_in_chunks(norm_verts).float().reshape(-1, 1)
        else:
            J_d = 1
        self.contracted_vertices.data += (J_d * perturbation)[:self.contracted_vertices.shape[0]]

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
        densities = np.zeros((N), dtype=np.float32)
        grds = np.zeros((N, 3), dtype=np.float32)
        sh_dim = ((self.max_sh_deg+1)**2-1)
        sh_coeffs = np.zeros((N, sh_dim, 3), dtype=np.float32)

        vertices = self.vertices
        indices = self.indices
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])

            circumcenters, _, density, rgb, grd, sh = self.compute_batch_features(vertices, indices, start, end)
            tets = vertices[indices[start:end]]
            base_color_v0_raw, normed_grd = offset_normalize(rgb, grd, circumcenters, tets)
            base_color_v0_raw = base_color_v0_raw.cpu().numpy().astype(np.float32)
            normed_grd = normed_grd.cpu().numpy().astype(np.float32)
            density = density.cpu().numpy().astype(np.float32)
            sh_coeff = sh.reshape(-1, sh_dim, 3)
            sh_coeffs[start:end] = sh_coeff.cpu().numpy()
            grds[start:end] = normed_grd.reshape(-1, 3)
            densities[start:end] = density.reshape(-1)

        tetra_dict = {}
        tetra_dict["vertex_indices"] = self.indices.cpu().numpy().astype(np.int32)
        tetra_dict["s"] = np.ascontiguousarray(densities)
        for i, co in enumerate(["x", "y", "z"]):
            tetra_dict[f"grd_{co}"]         = np.ascontiguousarray(grds[:, i])

        for i in range(sh_coeffs.shape[1]):
            tetra_dict[f"sh_{i+1}_r"] = np.ascontiguousarray(sh_coeffs[:, i, 0])
            tetra_dict[f"sh_{i+1}_g"] = np.ascontiguousarray(sh_coeffs[:, i, 1])
            tetra_dict[f"sh_{i+1}_b"] = np.ascontiguousarray(sh_coeffs[:, i, 2])


        data_dict = {
            "vertex": vertex_dict,
            "tetrahedron": tetra_dict,
        }

        tinyplypy.write_ply(str(path), data_dict, is_binary=True)

    def inv_contract(self, points):
        return inv_contract_points(points) * self.scene_scaling + self.center

    def contract(self, points):
        return contract_points((points - self.center) / self.scene_scaling)

    @property
    def vertices(self):
        if self.contract_vertices:
            verts = self.inv_contract(self.contracted_vertices)
        else:
            verts = self.contracted_vertices
        return torch.cat([verts, self.ext_vertices])

    def sh_up(self):
        self.current_sh_deg = min(self.max_sh_deg, self.current_sh_deg+1)

    @torch.no_grad()
    def update_triangulation(self, high_precision=False, density_threshold=0.0):
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
        
        self.indices = torch.as_tensor(indices_np).cuda()
        if density_threshold > 0:
            mask = self.calc_tet_density() > density_threshold
            self.indices = self.indices[mask]
            
        torch.cuda.empty_cache()

    def __len__(self):
        return self.vertices.shape[0]
        

class TetOptimizer:
    def __init__(self,
                 model: Model,
                 encoding_lr: float=1e-2,
                 final_encoding_lr: float=1e-2,
                 network_lr: float=1e-3,
                 final_network_lr: float=1e-3,
                 vertices_lr: float=4e-4,
                 final_vertices_lr: float=4e-7,
                 vertices_lr_delay_multi: float=0.01,
                 weight_decay=1e-10,
                 lambda_color=1e-10,
                 split_std: float = 0.5,
                 lr_delay: int = 500,
                 max_steps: int = 10000,
                 vert_lr_delay: int = 500,
                 sh_interval: int = 1000,
                 lambda_tv: float = 0.0,
                 lambda_density: float = 0.0,
                 spike_duration: int = 20,
                 densify_start: int = 2500,
                 densify_interval: int = 500,
                 densify_end: int = 15000,

                 density_lr:  float = 1e-3,
                 color_lr:    float = 1e-3,
                 gradient_lr: float = 1e-3,
                 sh_lr:       float = 1e-3,

                 lambda_dist: float = 1e-5,
                 dist_delay: int = 2000,

                 **kwargs):
        self.weight_decay = weight_decay
        self.lambda_color = lambda_color
        self.lambda_tv = lambda_tv
        self.lambda_density = lambda_density
        self.optim = optim.CustomAdam([
            {"params": model.backbone.encoding.parameters(), "lr": encoding_lr, "name": "encoding"},
        ], ignore_param_list=["encoding", "network"], betas=[0.9, 0.99], eps=1e-15)
        self.net_optim = optim.CustomAdam([
            {"params": model.backbone.network.parameters(), "lr": network_lr, "name": "network"},
            {"params": model.backbone.density_net.parameters(),   "lr": density_lr,  "name": "density"},
            {"params": model.backbone.color_net.parameters(),     "lr": color_lr,    "name": "color"},
            {"params": model.backbone.gradient_net.parameters(),  "lr": gradient_lr, "name": "gradient"},
            {"params": model.backbone.sh_net.parameters(),        "lr": sh_lr,       "name": "sh"},
        ], ignore_param_list=[], betas=[0.9, 0.99])
        self.ratios = dict(
            network = 1,
            density = density_lr / network_lr,
            color = color_lr / network_lr,
            gradient = gradient_lr / network_lr,
            sh = sh_lr / network_lr,
        )
        self.vert_lr_multi = 1 if model.contract_vertices else float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.contracted_vertices], "lr": self.vert_lr_multi*vertices_lr, "name": "contracted_vertices"},
        ])
        self.model = model
        self.tracker_n = 0
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None
        self.split_std = split_std

        self.dist_scheduler = get_expon_lr_func(lr_init=lambda_dist,
                                                lr_final=lambda_dist,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=dist_delay,
                                                max_steps=max_steps)

        base_net_scheduler = get_expon_lr_func(lr_init=network_lr,
                                                lr_final=final_network_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=max_steps)

        self.net_scheduler_args = SpikingLR(
            spike_duration, max_steps, base_net_scheduler,
            densify_start, densify_interval, densify_end,
            network_lr, network_lr)
            # network_lr, final_network_lr)

        base_encoder_scheduler = get_expon_lr_func(lr_init=encoding_lr,
                                                lr_final=final_encoding_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=max_steps)

        self.encoder_scheduler_args = SpikingLR(
            spike_duration, max_steps, base_encoder_scheduler,
            densify_start, densify_interval, densify_end,
            encoding_lr, encoding_lr)
            # encoding_lr, final_encoding_lr)

        self.vertex_lr = self.vert_lr_multi*vertices_lr
        base_vertex_scheduler = get_expon_lr_func(lr_init=self.vertex_lr,
                                                lr_final=self.vert_lr_multi*final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_multi,
                                                max_steps=max_steps,
                                                lr_delay_steps=vert_lr_delay)

        self.vertex_scheduler_args = SpikingLR(
            spike_duration, max_steps, base_vertex_scheduler,
            densify_start, densify_interval, densify_end,
            self.vertex_lr, self.vert_lr_multi*final_vertices_lr)
        # self.vertex_scheduler_args = base_vertex_scheduler
        self.iteration = 0

    def lambda_dist(self, iteration):
        return float(self.dist_scheduler(iteration))

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self.iteration = iteration
        for param_group in self.net_optim.param_groups:
            # if param_group["name"] == "network":
            ratio = self.ratios[param_group["name"]]
            lr = self.net_scheduler_args(iteration)
            param_group['lr'] = ratio * lr
        for param_group in self.optim.param_groups:
            if param_group["name"] == "encoding":
                lr = self.encoder_scheduler_args(iteration)
                param_group['lr'] = lr
        for param_group in self.vertex_optim.param_groups:
            if param_group["name"] == "contracted_vertices":
                lr = self.vertex_scheduler_args(iteration)
                self.vertex_lr = lr
                param_group['lr'] = lr

    def update_ema(self):
        self.ema.update()

    def add_points(self, new_verts: torch.Tensor, raw_verts=False):
        if self.model.contract_vertices and not raw_verts:
            new_verts = self.model.contract(new_verts)
        self.model.contracted_vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            contracted_vertices = new_verts
        ))['contracted_vertices']
        self.model.update_triangulation()

    def remove_points(self, mask: torch.Tensor):
        mask = mask[:self.model.contracted_vertices.shape[0]]
        self.model.contracted_vertices = self.vertex_optim.prune_optimizer(mask)['contracted_vertices']
        self.model.update_triangulation()

    @torch.no_grad()
    def split(self, clone_indices, split_point, split_mode):
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
        elif split_mode == "barycentric":
            barycentric = torch.rand((clone_indices.shape[0], clone_indices.shape[1], 1), device=device).clip(min=0.01, max=0.99)
            barycentric_weights = barycentric / (1e-3+barycentric.sum(dim=1, keepdim=True))
            new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights).sum(dim=1)
        elif split_mode == "split_point":
            split_point += 1e-4*torch.randn(*split_point.shape, device=self.model.device)
            new_vertex_location = split_point
            barycentric_weights = calc_barycentric(split_point, clone_vertices)
            barycentric_weights = barycentric_weights / (1e-3+barycentric_weights.sum(dim=1, keepdim=True))
            # new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights.unsqueeze(-1)).sum(dim=1)
        elif split_mode == "split_point_c":
            barycentric_weights = calc_barycentric(split_point, clone_vertices)
            barycentric_weights = barycentric_weights / (1e-3+barycentric_weights.sum(dim=1, keepdim=True))
            barycentric_weights += 1e-4*torch.randn(*barycentric_weights.shape, device=self.model.device)
            new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise Exception(f"Split mode: {split_mode} not supported")
        self.add_points(new_vertex_location)

    def main_step(self):
        self.optim.step()
        self.net_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad(set_to_none=True)
        self.net_optim.zero_grad(set_to_none=True)

    @property
    def sh_optim(self):
        return None

    def clip_gradient(self, grad_clip):
        for module in self.model.backbone.encoding.embeddings:
            torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip, error_if_nonfinite=True)

    def regularizer(self, render_pkg):
        weight_decay = self.weight_decay * sum([(embed.weight**2).mean() for embed in self.model.backbone.encoding.embeddings])

        if self.lambda_density > 0 or self.lambda_tv > 0:
            density = self.model.calc_tet_density()
            density_loss = (self.model.calc_tet_area().detach() * density).sum()
            if self.lambda_tv > 0:
                diff  = density[self.pairs[:,0]] - density[self.pairs[:,1]]
                tv_loss  = (self.face_area * diff.abs())
                tv_loss  = tv_loss.sum() / self.face_area.sum()
            else:
                tv_loss = 0
        else:
            density_loss = 0
            tv_loss = 0

        return weight_decay + self.lambda_tv * tv_loss + self.lambda_density * density_loss

    def update_triangulation(self, **kwargs):
        self.model.update_triangulation(**kwargs)
        self.build_tv()

    def build_tv(self):
        if self.lambda_tv > 0:
            self.pairs, self.face_area = topo_utils.build_tv_struct(
                self.model.vertices.detach(), self.model.indices, device='cuda')
