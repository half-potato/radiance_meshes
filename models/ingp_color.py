import torch
import math
from data.camera import Camera
from utils import optim
from sh_slang.eval_sh import eval_sh
from gdel3d import Del
from torch import nn
from icecream import ic

from utils.topo_utils import calculate_circumcenters_torch, fibonacci_spiral_on_sphere, calc_barycentric, sample_uniform_in_sphere, project_points_to_tetrahedra, contraction_jacobian_d_in_chunks
from utils import topo_utils
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
from scipy.spatial import  Delaunay, ConvexHull
import open3d as o3d
from data.types import BasicPointCloud
from utils import mesh_util
from utils.model_util import *
from models.base_model import BaseModel
from utils.ingp_util import grid_scale, compute_grid_offsets

torch.set_float32_matmul_precision('high')

def pad_for_tinycudann(x: torch.Tensor, granularity: int):
    """
    Pads the input tensor to ensure its total number of elements is a multiple
    of the tinycudann's required batch size granularity.

    Args:
        x (torch.Tensor): The input tensor to the tinycudann module.
        granularity (int): The BATCH_SIZE_GRANULARITY required by tinycudann.

    Returns:
        tuple: A tuple containing the padded tensor and the number of padded elements.
    """
    num_elements = x.shape[0]
    remainder = num_elements % granularity

    if remainder == 0:
        return x, 0
    else:
        padding_needed = granularity - remainder
        padding_shape = list(x.shape)
        padding_shape[0] = padding_needed
        padding = torch.zeros(padding_shape, dtype=x.dtype, device=x.device)
        x_padded = torch.cat([x, padding], dim=0)
        
        return x_padded, padding_needed

class Model(BaseModel):
    def __init__(self,
                 vertices: torch.Tensor,
                 ext_vertices: torch.Tensor,
                 center: torch.Tensor,
                 scene_scaling: float,
                 contract_vertices=True,
                 density_offset=-1,
                 current_sh_deg=2,
                 max_sh_deg=2,
                 ablate_circumsphere=False,
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

        # backbone = iNGPDW(sh_dim, **kwargs).to(self.device)
        # sample_cv = torch.rand((512, 3)).to(self.device)
        # sample_cr = torch.rand((512, 1)).to(self.device)
        # traced_backbone = torch.jit.trace(backbone, example_inputs=(sample_cv, sample_cr))
        # self.backbone = traced_backbone

        config = self.backbone.config
        offsets, pred_total = compute_grid_offsets(config, 3)
        total = list(self.backbone.encoding.parameters())[0].shape[0]
        # ic(offsets, pred_total, total)
        # assert total == pred_total, f"Pred #params: {pred_total} vs {total}"
        resolution = grid_scale(
            config['n_levels']-1, config['per_level_scale'], config['base_resolution'])
        self.different_size = 0
        self.nominal_offset_size = offsets[-1] - offsets[-2]
        for o1, o2 in zip(offsets[:-1], offsets[1:]):
            if o2 - o1 == self.nominal_offset_size:
                break
            else:
                self.different_size += 1
        self.offsets = offsets

        self.chunk_size = 308576
        self.mask_values = True
        self.frozen = False
        self.linear = False
        self.feature_dim = 7
        self.alpha = 0
        self.ablate_circumsphere = ablate_circumsphere

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

    def compute_batch_features(self, vertices, indices, start, end, circumcenters=None):
        tets = vertices[indices[start:end]]
        if circumcenters is None:
            circumcenter, radius = calculate_circumcenters_torch(tets.double())
        else:
            circumcenter = circumcenters[start:end]
            radius = torch.linalg.norm(circumcenter - vertices[indices[start:end, 0]], dim=-1)
        if self.ablate_circumsphere:
            circumcenter = tets.mean(dim=1)
        if self.training:
            circumcenter += self.alpha*torch.randn_like(circumcenter) * radius.reshape(-1, 1)
        normalized = (circumcenter - self.center) / self.scene_scaling
        cv, cr = contract_mean_std(normalized, radius / self.scene_scaling)
        x = (cv/2 + 1)/2

        output = checkpoint(self.backbone, x, cr, use_reentrant=True)

        # cr = cr.reshape(-1, 1)
        # x, n = pad_for_tinycudann(x, 256)
        # cr, n = pad_for_tinycudann(cr, 256)
        # N = circumcenter.shape[0]
        # output = self.backbone(x, cr.reshape(-1, 1))
        # output = [v[:N] for v in output]
        return circumcenter, normalized, *output

    def get_cell_values(self, camera: Camera, mask=None,
                        all_circumcenters=None, radii=None):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices.detach()

        features = torch.empty((indices.shape[0], self.feature_dim), device=self.device)
        start = 0
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])
            circumcenters, normalized, density, rgb, grd, sh = self.compute_batch_features(
                vertices, indices, start, end, circumcenters=all_circumcenters)
            dvrgbs = activate_output(camera.camera_center.to(self.device),
                                     density, rgb, grd, sh, indices[start:end],
                                     circumcenters,
                                     vertices, self.current_sh_deg, self.max_sh_deg)
            features[start:end] = dvrgbs
        return None, features

    @staticmethod
    def init_from_pcd(point_cloud, cameras, device, max_sh_deg,
                      voxel_size=0.00, **kwargs):
        torch.manual_seed(2)

        ccenters = torch.stack([c.camera_center.reshape(3) for c in cameras], dim=0).to(device)
        center = ccenters.mean(dim=0)
        scaling = torch.linalg.norm(ccenters - center.reshape(1, 3), dim=1, ord=torch.inf).max()
        print(f"Scene scaling: {scaling}. Center: {center}")

        vertices = torch.as_tensor(point_cloud.points).float()


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

        # num_ext = 1000
        # ext_vertices = topo_utils.expand_convex_hull(vertices, 1, device=vertices.device)
        # if ext_vertices.shape[0] > num_ext:
        #     inds = np.random.default_rng().permutation(ext_vertices.shape[0])[:num_ext]
        #     ext_vertices = ext_vertices[inds]
        # else:
        #     num_ext = ext_vertices.shape[0]

        model = Model(vertices.cuda(), ext_vertices, center, scaling,
                      max_sh_deg=max_sh_deg, **kwargs)
        return model

    @staticmethod
    def load_ckpt(path: Path, device):
        ckpt_path = path / "ckpt_prefreeze.pth"
        config_path = path / "config.json"
        config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)
        print(ckpt.keys())
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

    def calc_tet_density(self):
        densities = []
        verts = self.vertices
        for start in range(0, self.indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.indices.shape[0])
            
            _, _, density, _, _, _ = self.compute_batch_features(verts, self.indices, start, end)

            densities.append(density.reshape(-1))
        return torch.cat(densities)

    def compute_features(self, offset=False):
        vertices = self.vertices
        indices = self.indices
        cs, ds, rs, gs, ss = [], [], [], [], []
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])

            circumcenters, _, density, rgb, grd, sh = self.compute_batch_features(vertices, indices, start, end)
            tets = vertices[indices[start:end]]
            cs.append(circumcenters)
            ds.append(density)
            ss.append(sh)
            if offset:
                base_color_v0_raw, normed_grd = offset_normalize(rgb, grd, circumcenters, tets)
                # rs.append(base_color_v0_raw)
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
        self.full_indices = indices.clone()
        self.indices = indices
        if density_threshold > 0 or alpha_threshold > 0:
            tet_density = self.calc_tet_density()
            tet_alpha = self.calc_tet_alpha(mode="min", density=tet_density)
            mask = (tet_density > density_threshold) | (tet_alpha > alpha_threshold)
            self.indices = self.indices[mask]
            self.mask = mask
        else:
            self.mask = torch.ones((self.full_indices.shape[0]), dtype=bool, device='cuda')
            
        torch.cuda.empty_cache()

    def __len__(self):
        return self.vertices.shape[0]
        

    def compute_weight_decay(self):
        return sum([(embed.weight**2).mean() for embed in self.backbone.encoding.embeddings])
        param = list(self.backbone.encoding.parameters())[0]
        weight_decay = 0
        ind = 0
        for i in range(self.different_size):
            o = self.offsets[i+1] - self.offsets[i]
            weight_decay = weight_decay + (param[ind:self.offsets[i+1]]**2).mean()
            ind += o
        weight_decay = weight_decay + (param[ind:].reshape(-1, self.nominal_offset_size)**2).mean(dim=1).sum()
        return weight_decay

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
                 freeze_start: int = 10000,
                 vert_lr_delay: int = 500,
                 sh_interval: int = 1000,
                 lambda_tv: float = 0.0,
                 lambda_density: float = 0.0,

                 spike_duration: int = 20,
                 densify_start: int = 2500,
                 densify_interval: int = 500,
                 densify_end: int = 15000,
                 midpoint: int = 2000,

                 density_lr:  float = 1e-3,
                 color_lr:    float = 1e-3,
                 gradient_lr: float = 1e-3,
                 sh_lr:       float = 1e-3,

                 lambda_dist: float = 1e-5,
                 percent_alpha: float = 0.02,
                 dist_delay: int = 2000,

                 **kwargs):
        self.weight_decay = weight_decay
        self.lambda_color = lambda_color
        self.lambda_tv = lambda_tv
        self.lambda_density = lambda_density

        self.optim = optim.CustomAdam([
            {"params": model.backbone.encoding.parameters(), "lr": encoding_lr, "name": "encoding"},
        ], ignore_param_list=["encoding", "network"], betas=[0.9, 0.999], eps=1e-15)

        self.net_optim = optim.CustomAdam([
            # {"params": model.backbone.network.parameters(),   "lr": network_lr,  "name": "density"},
            {"params": model.backbone.density_net.parameters(),   "lr": network_lr,  "name": "density"},
            {"params": model.backbone.color_net.parameters(),     "lr": network_lr,    "name": "color"},
            # {"params": model.backbone.density_color_net.parameters(),     "lr": network_lr,    "name": "color"},
            {"params": model.backbone.gradient_net.parameters(),  "lr": network_lr, "name": "gradient"},
            {"params": model.backbone.sh_net.parameters(),        "lr": network_lr,       "name": "sh"},
        ], ignore_param_list=[], betas=[0.9, 0.999], eps=1e-15)
        self.vert_lr_multi = 1 if model.contract_vertices else float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.contracted_vertices], "lr": self.vert_lr_multi*vertices_lr, "name": "contracted_vertices"},
        ])
        self.model = model
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None
        self.split_std = split_std

        self.alpha_sched = get_expon_lr_func(lr_init=percent_alpha*float(model.scene_scaling.cpu()),
                                                lr_final=1e-20,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=0,
                                                max_steps=freeze_start//3)

        base_net_scheduler = get_expon_lr_func(lr_init=network_lr,
                                                lr_final=final_network_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=freeze_start)

        self.net_scheduler_args = SpikingLR(
            spike_duration, freeze_start, base_net_scheduler,
            midpoint, densify_interval, densify_end,
            network_lr, network_lr)
            # network_lr, final_network_lr)

        base_encoder_scheduler = get_expon_lr_func(lr_init=encoding_lr,
                                                lr_final=final_encoding_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=freeze_start)

        self.encoder_scheduler_args = SpikingLR(
            spike_duration, freeze_start, base_encoder_scheduler,
            midpoint, densify_interval, densify_end,
            encoding_lr, encoding_lr)
            # encoding_lr, final_encoding_lr)

        self.vertex_lr = self.vert_lr_multi*vertices_lr
        base_vertex_scheduler = get_expon_lr_func(lr_init=self.vertex_lr,
                                                lr_final=self.vert_lr_multi*final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_multi,
                                                max_steps=freeze_start,
                                                lr_delay_steps=vert_lr_delay)

        self.vertex_scheduler_args = base_vertex_scheduler
        self.vertex_scheduler_args = SpikingLR(
            spike_duration, freeze_start, base_vertex_scheduler,
            midpoint, densify_interval, densify_end,
            # self.vertex_lr, self.vert_lr_multi*final_vertices_lr)
            self.vertex_lr, self.vertex_lr)
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
            if param_group["name"] == "contracted_vertices":
                lr = self.vertex_scheduler_args(iteration)
                self.vertex_lr = lr
                param_group['lr'] = lr

    def add_points(self, new_verts: torch.Tensor, raw_verts=False):
        if self.model.contract_vertices and not raw_verts:
            new_verts = self.model.contract(new_verts)
        self.model.contracted_vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            contracted_vertices = new_verts
        ))['contracted_vertices']
        self.model.update_triangulation()

    def remove_points(self, keep_mask: torch.Tensor):
        keep_mask = keep_mask[:self.model.contracted_vertices.shape[0]]
        self.model.contracted_vertices = self.vertex_optim.prune_optimizer(keep_mask)['contracted_vertices']
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

    def regularizer(self, render_pkg):
        # weight_decay = self.weight_decay * sum([(embed.weight**2).mean() for embed in self.model.backbone.encoding.embeddings])
        weight_decay = self.weight_decay * self.model.compute_weight_decay()

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
