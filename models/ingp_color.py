import torch
import math
from data.camera import Camera
from utils import optim
from gdel3d import Del
from torch import nn
from icecream import ic

from utils import topo_utils
from utils.contraction import contract_mean_std

from utils.train_util import get_expon_lr_func, SpikingLR, TwoPhaseLR
from utils.graphics_utils import l2_normalize_th
from torch.utils.checkpoint import checkpoint
from pathlib import Path
import numpy as np
from utils.args import Args
from scipy.spatial import  Delaunay
# import open3d as o3d
from sh_slang.eval_sh import eval_sh
from utils.model_util import *
from models.base_model import BaseModel
from muon import SingleDeviceMuonWithAuxAdam

torch.set_float32_matmul_precision('high')


class Model(BaseModel):
    def __init__(self,
                 vertices: torch.Tensor,
                 center: torch.Tensor,
                 scene_scaling: float,
                 density_offset=-1,
                 current_sh_deg=2,
                 max_sh_deg=2,
                 glo_dim=0,
                 ablate_circumsphere=False,
                 **kwargs):
        super().__init__()
        self.device = vertices.device
        self.density_offset = density_offset
        self.max_sh_deg = max_sh_deg
        self.sh_dim = ((1+max_sh_deg)**2-1)*3
        self.backbone = torch.compile(iNGPDW(self.sh_dim, glo_dim=glo_dim, **kwargs)).to(self.device)
        self.default_glo = None if glo_dim == 0 else torch.zeros((1, glo_dim), device=self.device)
        self.glo_dim = glo_dim
        self.chunk_size = 408576
        self.mask_values = True
        self.frozen = False
        self.alpha = 0
        self.linear = False
        self.feature_dim = 7
        self.current_sh_deg = current_sh_deg
        self.ablate_circumsphere = ablate_circumsphere

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
        cam_center = camera.camera_center.to(self.device)
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])
            tets = vertices[indices[start:end]]
            circumcenters, normalized, density, rgb, grd, sh = self.compute_batch_features(
                vertices, indices, start, end, circumcenters=all_circumcenters, glo=glo)
            tet_color_raw = eval_sh(
                tets.mean(dim=1).detach(),
                RGB2SH(rgb),
                sh.reshape(-1, (self.max_sh_deg+1)**2 - 1, 3),
                cam_center,
                self.current_sh_deg).float()
            dvrgbs = activate_output(cam_center, tet_color_raw,
                                     density, grd,
                                     circumcenters,
                                     tets)
            normed_cc.append(normalized)
            outputs.append(dvrgbs)
        features = torch.cat(outputs, dim=0)
        normed_cc = torch.cat(normed_cc, dim=0)
        return normed_cc, features

    def compute_features(self):
        vertices = self.vertices
        indices = self.indices
        cs, ds, rs, gs, ss = [], [], [], [], []
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])

            circumcenters, _, density, rgb, grd, sh = self.compute_batch_features(vertices, indices, start, end, glo=self.default_glo)
            tets = vertices[indices[start:end]]
            radius = torch.linalg.norm(tets[:, :1] - circumcenters[:, None, :], dim=-1, keepdim=True)
            cs.append(circumcenters)
            ds.append(density)
            ss.append(sh)
            rs.append(rgb)
            gs.append(grd)
            gs.append(safe_div(grd, radius))
        cs = torch.cat(cs, dim=0)
        ds = torch.cat(ds, dim=0)
        rs = torch.cat(rs, dim=0)
        gs = torch.cat(gs, dim=0)
        ss = torch.cat(ss, dim=0)
        return cs, ds, rs, gs, ss


    def compute_batch_features(self, vertices, indices, start, end, circumcenters=None, glo=None):
        tets = vertices[indices[start:end]]
        if circumcenters is None:
            circumcenter, radius = topo_utils.calculate_circumcenters_torch(tets.double())
        else:
            circumcenter = circumcenters[start:end]
        if self.ablate_circumsphere:
            circumcenter = tets.mean(dim=1)
        normalized = (circumcenter - self.center) / self.scene_scaling
        radius = torch.linalg.norm(circumcenter - vertices[indices[start:end, 0]], dim=-1)
        cv, cr = contract_mean_std(normalized, radius / self.scene_scaling)
        x = (cv/2 + 1)/2

        glo = glo if glo is not None else self.default_glo

        density, rgb, grd, sh = checkpoint(self.backbone, x, cr, glo, use_reentrant=True)
        # density = safe_div(density, radius.reshape(-1, 1).detach())
        # vol = topo_utils.tet_volumes(tets).clip(min=1, max=1000)
        # density = safe_div(density, vol.reshape(-1, 1).detach())
        return circumcenter, normalized, density, rgb, grd, sh

    @staticmethod
    def load_ckpt(path: Path, device, config=None):
        ckpt_path = path / "ckpt.pth"
        if config is None:
            config_path = path / "config.json"
            config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)
        vertices = ckpt['interior_vertices']
        indices = ckpt["indices"]  # shape (N,4)
        del ckpt["indices"]
        print(f"Loaded {vertices.shape[0]} vertices")
        model = Model(vertices.to(device), ckpt['center'], ckpt['scene_scaling'], **config.as_dict())
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

    @property
    def vertices(self):
        return self.interior_vertices

    def sh_up(self):
        self.current_sh_deg = min(self.current_sh_deg + 1, self.max_sh_deg)

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
        
        # Ensure volume is positive
        indices = torch.as_tensor(indices_np).cuda()
        vols = topo_utils.tet_volumes(verts[indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            indices[reverse_mask] = indices[reverse_mask][:, [1, 0, 2, 3]]

        # Cull tets with low density
        self.indices = indices
        denom = topo_utils.tet_denom(self.vertices.detach()[self.indices]).detach()
        if density_threshold > 0 or alpha_threshold > 0:
            tet_density = self.calc_tet_density()
            tet_alpha = self.calc_tet_alpha(mode="min", density=tet_density)
            mask = ((tet_density > density_threshold) | (tet_alpha > alpha_threshold)) & (denom > 1e-10)
            self.indices = self.indices[mask]
        else:
            mask = (denom > 1e-10)
            self.indices = self.indices[mask]
            
        torch.cuda.empty_cache()

    def __len__(self):
        return self.vertices.shape[0]
        

    @staticmethod
    def init_from_pcd(point_cloud, cameras, device, max_sh_deg,
                      ext_convex_hull, voxel_size=0.00, **kwargs):
        torch.manual_seed(2)

        ccenters = torch.stack([c.camera_center.reshape(3) for c in cameras], dim=0).to(device)
        center = ccenters.mean(dim=0)
        scaling = torch.linalg.norm(ccenters - center.reshape(1, 3), dim=1, ord=torch.inf).max()
        print(f"Scene scaling: {scaling}. Center: {center}")

        vertices = torch.as_tensor(point_cloud.points).float()

        # dist = torch.clamp_min(distCUDA2(vertices.cuda()), 0.0000001).sqrt().cpu()

        # vertices = vertices.reshape(-1, 1, 3).expand(-1, init_repeat, 3)
        # vertices = vertices + torch.randn(*vertices.shape) * dist.reshape(-1, 1, 1).clip(min=0.01)
        # vertices = vertices.reshape(-1, 3)

        # Convert BasicPointCloud to Open3D PointCloud
        # o3d_pcd = o3d.geometry.PointCloud()
        # o3d_pcd.points = o3d.utility.Vector3dVector(vertices.numpy())
        #
        # # Perform voxel downsampling
        # if voxel_size > 0:
        #     o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)
        #
        # N = point_cloud.points.shape[0]
        # vertices = torch.as_tensor(np.asarray(o3d_pcd.points)).float()

        vertices = vertices + torch.randn(*vertices.shape) * 1e-3

        # add sphere
        pcd_scaling = torch.linalg.norm(vertices - center.cpu().reshape(1, 3), dim=1, ord=2)
        pcd_scaling = (vertices - vertices.mean(dim=0, keepdim=True)).abs().max(dim=0).values
        new_radius = math.sqrt(2) * pcd_scaling.cpu()

        # vertices = topo_util.sample_uniform_in_sphere(10000, 3, base_radius=0, radius=new_radius, device='cpu') + center.reshape(1, 3).cpu()

        # vertices = vertices + torch.randn(*vertices.shape) * 1e-3
        # v = Del(vertices.shape[0])
        # indices_np, prev = v.compute(vertices.detach().cpu().double())
        # indices_np = indices_np.numpy()
        # indices_np = indices_np[(indices_np < vertices.shape[0]).all(axis=1)]
        # vertices = vertices[indices_np].mean(dim=1)
        # vertices = vertices + torch.randn(*vertices.shape) * 1e-3

        # within_sphere = topo_util.sample_uniform_in_sphere(10000, 3, base_radius=new_radius, radius=new_radius, device='cpu') + center.reshape(1, 3).cpu()
        # vertices = torch.cat([vertices, within_sphere], dim=0)
        if ext_convex_hull:
            num_ext = 5000
            ext_vertices = topo_utils.expand_convex_hull(vertices, 1, device=vertices.device)
            if ext_vertices.shape[0] > num_ext:
                inds = np.random.default_rng().permutation(ext_vertices.shape[0])[:num_ext]
                ext_vertices = ext_vertices[inds]
            else:
                num_ext = ext_vertices.shape[0]
        else:
            num_ext = 5000
            ext_vertices = topo_utils.fibonacci_spiral_on_sphere(num_ext, new_radius.reshape(1, 3), device='cpu') + center.reshape(1, 3).cpu()
            # ext_vertices = torch.empty((0, 3), device='cpu')
            num_ext = ext_vertices.shape[0]
        vertices = torch.cat([vertices, ext_vertices], dim=0)

        model = Model(vertices.cuda(), center, scaling,
                      max_sh_deg=max_sh_deg, **kwargs)
        return model

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
                 sh_lr_div: int = 20,

                 glo_net_decay: float = 0,
                 glo_network_lr: float = 1e-3,
                 percent_alpha: float = 0.02,
                 sh_weight_decay: float = 1e-5,
                 **kwargs):
        self.optim = optim.CustomAdam([
            {"params": model.backbone.encoding.parameters(), "lr": encoding_lr, "name": "encoding"},
        ], ignore_param_list=["encoding", "network"], betas=[0.9, 0.999], eps=1e-15)
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
        self.sh_lr_div = sh_lr_div
        self.net_optim = SingleDeviceMuonWithAuxAdam(
            process(model.backbone.density_net, network_lr) + \
            process(model.backbone.color_net, network_lr) + \
            process(model.backbone.gradient_net, network_lr) + \
            glo_p
        )
        self.sh_net_optim = SingleDeviceMuonWithAuxAdam(
            process(model.backbone.sh_net, network_lr/self.sh_lr_div, weight_decay=sh_weight_decay)
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
        # self.net_scheduler_args = make_spiking(
        #     network_lr, network_lr, final_network_lr)
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
        # self.encoder_scheduler_args = make_spiking(
        #     encoding_lr, encoding_lr, final_encoding_lr)

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

        # self.vertex_scheduler_args = make_spiking(
        #     self.vertex_lr, self.vertex_lr, self.vert_lr_multi*final_vertices_lr)
        self.iteration = 0

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self.iteration = iteration
        self.model.alpha = self.alpha_sched(iteration)
        for param_group in self.net_optim.param_groups:
            lr = self.net_scheduler_args(iteration)
            param_group['lr'] = lr
        for param_group in self.sh_net_optim.param_groups:
            lr = self.net_scheduler_args(iteration)
            param_group['lr'] = lr/self.sh_lr_div
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
    def split(self, clone_indices, split_point, split_std, **kwargs):
        device = self.model.device
        _, radius = topo_utils.calculate_circumcenters_torch(self.model.vertices[clone_indices])
        split_point += (split_std * radius.reshape(-1, 1)).clip(min=1e-3, max=3) * torch.randn(*split_point.shape, device=self.model.device)
        self.add_points(split_point)

    def main_step(self):
        self.optim.step()
        self.net_optim.step()
        self.sh_net_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad(set_to_none=True)
        self.net_optim.zero_grad(set_to_none=True)
        self.sh_net_optim.zero_grad(set_to_none=True)

    @property
    def sh_optim(self):
        return None

    def regularizer(self, render_pkg, weight_decay, lambda_tv, **kwargs):
        weight_decay = weight_decay * sum([(embed.weight**2).mean() for embed in self.model.backbone.encoding.embeddings])

        return weight_decay

    def update_triangulation(self, **kwargs):
        self.model.update_triangulation(**kwargs)
