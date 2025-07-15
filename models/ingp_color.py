import torch
import math
from data.camera import Camera
from utils import optim
from sh_slang.eval_sh import eval_sh
from gDel3D.build.gdel3d import Del
from torch import nn
from icecream import ic

from utils.topo_utils import calculate_circumcenters_torch, fibonacci_spiral_on_sphere, calc_barycentric, sample_uniform_in_sphere, project_points_to_tetrahedra, contraction_jacobian_d_in_chunks
from utils import topo_utils
from utils.safe_math import safe_exp, safe_div, safe_sqrt
from utils.contraction import contract_mean_std
from utils.contraction import contract_points, inv_contract_points

from utils.train_util import get_expon_lr_func, SpikingLR, TwoPhaseLR
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
from models.base_model import BaseModel
from muon import SingleDeviceMuonWithAuxAdam

torch.set_float32_matmul_precision('high')


class Model(BaseModel):
    def __init__(self,
                 vertices: torch.Tensor,
                 ext_vertices: torch.Tensor,
                 center: torch.Tensor,
                 scene_scaling: float,
                 density_offset=-1,
                 current_sh_deg=2,
                 max_sh_deg=2,
                 glo_dim=0,
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
        self.backbone = torch.compile(iNGPDW(sh_dim, glo_dim=glo_dim, **kwargs)).to(self.device)
        self.default_glo = None if glo_dim == 0 else torch.zeros((1, glo_dim), device=self.device)
        self.chunk_size = 408576
        self.mask_values = True
        self.frozen = False
        self.alpha = 0
        self.linear = False
        self.feature_dim = 7

        self.register_buffer('ext_vertices', ext_vertices.to(self.device))
        self.register_buffer('center', center.reshape(1, 3))
        self.register_buffer('scene_scaling', torch.tensor(float(scene_scaling), device=self.device))
        self.interior_vertices = nn.Parameter(vertices.detach())
        self.update_triangulation()

    @property
    def num_int_verts(self):
        return self.interior_vertices.shape[0]

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
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])
            circumcenters, normalized, density, rgb, grd, sh = self.compute_batch_features(
                vertices, indices, start, end, circumcenters=all_circumcenters, glo=glo)
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
    def init_from_pcd(point_cloud, cameras, device, max_sh_deg,
                      voxel_size=0.00, **kwargs):
        torch.manual_seed(2)

        ccenters = torch.stack([c.camera_center.reshape(3) for c in cameras], dim=0).to(device)
        center = ccenters.mean(dim=0)
        scaling = torch.linalg.norm(ccenters - center.reshape(1, 3), dim=1, ord=torch.inf).max()
        print(f"Scene scaling: {scaling}. Center: {center}")

        vertices = torch.as_tensor(point_cloud.points).float()

        dist = torch.clamp_min(distCUDA2(vertices.cuda()), 0.0000001).sqrt().cpu()

        # vertices = vertices.reshape(-1, 1, 3).expand(-1, init_repeat, 3)
        # vertices = vertices + torch.randn(*vertices.shape) * dist.reshape(-1, 1, 1).clip(min=0.01)
        # vertices = vertices.reshape(-1, 3)

        # Convert BasicPointCloud to Open3D PointCloud
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(vertices.numpy())

        # Perform voxel downsampling
        if voxel_size > 0:
            o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)

        N = point_cloud.points.shape[0]
        vertices = torch.as_tensor(np.asarray(o3d_pcd.points)).float()
        vertices = vertices + torch.randn(*vertices.shape) * 1e-3

        # add sphere
        pcd_scaling = torch.linalg.norm(vertices - center.cpu().reshape(1, 3), dim=1, ord=2).max()
        new_radius = pcd_scaling.cpu().item()

        # vertices = sample_uniform_in_sphere(10000, 3, base_radius=0, radius=new_radius, device='cpu') + center.reshape(1, 3).cpu()

        # vertices = vertices + torch.randn(*vertices.shape) * 1e-3
        # v = Del(vertices.shape[0])
        # indices_np, prev = v.compute(vertices.detach().cpu().double())
        # indices_np = indices_np.numpy()
        # indices_np = indices_np[(indices_np < vertices.shape[0]).all(axis=1)]
        # vertices = vertices[indices_np].mean(dim=1)
        # vertices = vertices + torch.randn(*vertices.shape) * 1e-3

        # within_sphere = sample_uniform_in_sphere(10000, 3, base_radius=new_radius, radius=new_radius, device='cpu') + center.reshape(1, 3).cpu()
        # vertices = torch.cat([vertices, within_sphere], dim=0)
        num_ext = 1000
        ext_vertices = fibonacci_spiral_on_sphere(num_ext, new_radius, device='cpu') + center.reshape(1, 3).cpu()
        num_ext = ext_vertices.shape[0]

        model = Model(vertices.cuda(), ext_vertices, center, scaling,
                      max_sh_deg=max_sh_deg, **kwargs)
        return model

    def compute_batch_features(self, vertices, indices, start, end, circumcenters=None, glo=None):
        if circumcenters is None:
            tets = vertices[indices[start:end]]
            circumcenter, radius = calculate_circumcenters_torch(tets.double())
        else:
            circumcenter = circumcenters[start:end]
        if self.training:
            circumcenter += self.alpha*torch.rand_like(circumcenter)
        normalized = (circumcenter - self.center) / self.scene_scaling
        radius = torch.linalg.norm(circumcenter - vertices[indices[start:end, 0]], dim=-1)
        cv, cr = contract_mean_std(normalized, radius / self.scene_scaling)
        x = (cv/2 + 1)/2

        glo = glo if glo is not None else self.default_glo

        output = checkpoint(self.backbone, x, cr, glo, use_reentrant=True)
        return circumcenter, normalized, *output

    @staticmethod
    def load_ckpt(path: Path, device):
        ckpt_path = path / "ckpt.pth"
        config_path = path / "config.json"
        config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)
        vertices = ckpt['interior_vertices']
        indices = ckpt["indices"]  # shape (N,4)
        del ckpt["indices"]
        print(f"Loaded {vertices.shape[0]} vertices")
        ext_vertices = ckpt['ext_vertices']
        model = Model(vertices.to(device), ext_vertices, ckpt['center'], ckpt['scene_scaling'], **config.as_dict())
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

    def compute_features(self, offset=False):
        vertices = self.vertices
        indices = self.indices
        cs, ds, rs, gs, ss = [], [], [], [], []
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])

            circumcenters, _, density, rgb, grd, sh = self.compute_batch_features(vertices, indices, start, end, glo=self.default_glo)
            tets = vertices[indices[start:end]]
            cs.append(circumcenters)
            ds.append(density)
            ss.append(sh)
            if offset:
                base_color_v0_raw, normed_grd = offset_normalize(rgb, grd, circumcenters, tets)
                rs.append(base_color_v0_raw)
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

    def inv_contract(self, points):
        return inv_contract_points(points) * self.scene_scaling + self.center

    def contract(self, points):
        return contract_points((points - self.center) / self.scene_scaling)

    @property
    def vertices(self):
        verts = self.interior_vertices
        return torch.cat([verts, self.ext_vertices])

    def sh_up(self):
        self.current_sh_deg = min(self.max_sh_deg, self.current_sh_deg+1)

    @torch.no_grad()
    def update_triangulation(self, high_precision=False, density_threshold=0.0, alpha_threshold=0.0):
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
        if density_threshold > 0 or alpha_threshold > 0:
            tet_density = self.calc_tet_density()
            tet_alpha = self.calc_tet_alpha(mode="min", density=tet_density)
            mask = (tet_density > density_threshold) | (tet_alpha > alpha_threshold)
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
                 split_std: float = 0.5,
                 lr_delay: int = 500,
                 final_iter: int = 10000,
                 vert_lr_delay: int = 500,
                 lambda_tv: float = 0.0,
                 lambda_density: float = 0.0,

                 spike_duration: int = 20,
                 densify_start: int = 2500,
                 densify_interval: int = 500,
                 densify_end: int = 15000,
                 midpoint: int = 2000,

                 glo_network_lr: float = 1e-3,
                 percent_alpha: float = 0.02,

                 **kwargs):
        self.weight_decay = weight_decay
        self.lambda_tv = lambda_tv
        self.lambda_density = lambda_density
        self.optim = optim.CustomAdam([
            {"params": model.backbone.encoding.parameters(), "lr": encoding_lr, "name": "encoding"},
        ], ignore_param_list=["encoding", "network"], betas=[0.9, 0.999], eps=1e-15)
        # self.net_optim = optim.CustomAdam([
        params = dict(
            weight_decay=0,
        )
        def process(body, lr):
            hidden_weights = [p for p in body.parameters() if p.ndim >= 2]
            hidden_gains_biases = [p for p in body.parameters() if p.ndim < 2]
            a = dict(
                params=hidden_weights,
                use_muon = True,
                momentum=0.95,
                lr=lr,
                **params
            )
            b = dict(
                params=hidden_gains_biases,
                use_muon = False,
                betas=(0.9, 0.999),
                eps=1e-15,
                **params
            )
            return [a, b]
        self.net_optim = SingleDeviceMuonWithAuxAdam(
            process(model.backbone.density_net, network_lr) + \
            process(model.backbone.color_net, network_lr) + \
            process(model.backbone.gradient_net, network_lr) + \
            process(model.backbone.sh_net, network_lr) + \
            process(model.backbone.glo_net, glo_network_lr)
        )
        self.vert_lr_multi = float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.interior_vertices], "lr": self.vert_lr_multi*vertices_lr, "name": "interior_vertices"},
        ])
        self.model = model
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None
        self.split_std = split_std
        def make_spiking(init_lr, final_peak_lr, final_lr):
            return TwoPhaseLR(
                final_iter, densify_start, densify_interval, densify_end, init_lr, final_peak_lr, (init_lr + final_lr) / 2, final_lr)

        self.alpha_sched = get_expon_lr_func(lr_init=percent_alpha*float(model.scene_scaling.cpu()),
                                                lr_final=1e-20,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=0,
                                                max_steps=final_iter//3)

        base_net_scheduler = get_expon_lr_func(lr_init=network_lr,
                                                lr_final=final_network_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=final_iter)

        self.net_scheduler_args = SpikingLR(
            spike_duration, final_iter, base_net_scheduler,
            midpoint, densify_interval, densify_end,
            network_lr, network_lr)
        self.net_scheduler_args = make_spiking(
            network_lr, network_lr, final_network_lr)
            # network_lr, final_network_lr)

        base_encoder_scheduler = get_expon_lr_func(lr_init=encoding_lr,
                                                lr_final=final_encoding_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=final_iter)

        self.encoder_scheduler_args = SpikingLR(
            spike_duration, final_iter, base_encoder_scheduler,
            midpoint, densify_interval, densify_end,
            encoding_lr, encoding_lr)
            # encoding_lr, final_encoding_lr)
        self.encoder_scheduler_args = make_spiking(
            encoding_lr, encoding_lr, final_encoding_lr)

        self.vertex_lr = self.vert_lr_multi*vertices_lr
        base_vertex_scheduler = get_expon_lr_func(lr_init=self.vertex_lr,
                                                lr_final=self.vert_lr_multi*final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_multi,
                                                max_steps=final_iter,
                                                lr_delay_steps=vert_lr_delay)

        self.vertex_scheduler_args = base_vertex_scheduler
        self.vertex_scheduler_args = SpikingLR(
            spike_duration, final_iter, base_vertex_scheduler,
            midpoint, densify_interval, densify_end,
            # self.vertex_lr, self.vert_lr_multi*final_vertices_lr)
            self.vertex_lr, self.vertex_lr)

        self.vertex_scheduler_args = make_spiking(
            self.vertex_lr, self.vertex_lr, self.vert_lr_multi*final_vertices_lr)
        self.iteration = 0

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self.iteration = iteration
        self.model.alpha = self.alpha_sched(iteration)
        for param_group in self.net_optim.param_groups:
            lr = self.net_scheduler_args(iteration)
            param_group['lr'] = lr
        for param_group in self.optim.param_groups:
            if param_group["name"] == "encoding":
                lr = self.encoder_scheduler_args(iteration)
                param_group['lr'] = lr
        for param_group in self.vertex_optim.param_groups:
            if param_group["name"] == "interior_vertices":
                lr = self.vertex_scheduler_args(iteration)
                self.vertex_lr = lr
                param_group['lr'] = lr

    def add_points(self, new_verts: torch.Tensor, raw_verts=False):
        self.model.interior_vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            interior_vertices = new_verts
        ))['interior_vertices']
        self.model.update_triangulation()

    def remove_points(self, keep_mask: torch.Tensor):
        keep_mask = keep_mask[:self.model.interior_vertices.shape[0]]
        self.model.interior_vertices = self.vertex_optim.prune_optimizer(keep_mask)['interior_vertices']
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

    def main_step(self):
        self.optim.step()
        self.net_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad(set_to_none=True)
        self.net_optim.zero_grad(set_to_none=True)

    @property
    def sh_optim(self):
        return None

    def regularizer(self, render_pkg, weight_decay, lambda_tv, **kwargs):
        weight_decay = weight_decay * sum([(embed.weight**2).mean() for embed in self.model.backbone.encoding.embeddings])

        if lambda_tv > 0:
            density = self.model.calc_tet_density()
            diff  = density[self.pairs[:,0]] - density[self.pairs[:,1]]
            tv_loss  = (self.face_area * diff.abs())
            tv_loss  = tv_loss.sum() / self.face_area.sum()
        else:
            tv_loss = 0

        return weight_decay + lambda_tv * tv_loss

    def update_triangulation(self, **kwargs):
        self.model.update_triangulation(**kwargs)
        if self.lambda_tv > 0:
            self.build_tv()

    def build_tv(self):
        self.pairs, self.face_area = topo_utils.build_tv_struct(
            self.model.vertices.detach(), self.model.indices, device='cuda')

    def prune(self, diff_threshold, **kwargs):
        if diff_threshold <= 0:
            return
        density = self.model.calc_tet_density()
        self.build_tv()
        diff  = density[self.pairs[:,0]] - density[self.pairs[:,1]]
        tet_diff  = (self.face_area * diff.abs()).reshape(-1)

        indices = self.model.indices.long()
        device = indices.device
        vert_diff = torch.zeros((self.model.vertices.shape[0],), device=device)

        reduce_type = "amax"
        vert_diff.scatter_reduce_(dim=0, index=indices[..., 0], src=tet_diff, reduce=reduce_type)
        vert_diff.scatter_reduce_(dim=0, index=indices[..., 1], src=tet_diff, reduce=reduce_type)
        vert_diff.scatter_reduce_(dim=0, index=indices[..., 2], src=tet_diff, reduce=reduce_type)
        vert_diff.scatter_reduce_(dim=0, index=indices[..., 3], src=tet_diff, reduce=reduce_type)
        keep_mask = vert_diff > diff_threshold
        print(f"Pruned {(~keep_mask).sum()} points. VD: {vert_diff.mean()} TD: {tet_diff.mean()}")
        self.remove_points(keep_mask)
