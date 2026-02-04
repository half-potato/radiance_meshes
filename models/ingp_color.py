import torch
import math
from data.camera import Camera
from utils import optim
from gdel3d import Del
from torch import nn
from icecream import ic

from utils.topo_utils import calculate_circumcenters_torch, fibonacci_spiral_on_sphere, get_tet_adjacency
from utils import topo_utils
from utils.contraction import contract_mean_std

from utils.train_util import get_expon_lr_func, SpikingLR
from utils.graphics_utils import l2_normalize_th
from torch.utils.checkpoint import checkpoint
from pathlib import Path
import numpy as np
from utils.args import Args
from scipy.spatial import  Delaunay
import open3d as o3d
from utils.model_util import *
from models.base_model import BaseModel
from utils.ingp_util import grid_scale, compute_grid_offsets
from utils.hashgrid import HashEmbedderOptimized
from utils import hashgrid
import tinycudann as tcnn
import time

def get_tet_adjacency_from_scipy(tri):
    # tri.simplices is (N, 4)
    # tri.neighbors is (N, 4)
    
    tets = tri.simplices
    neighbors = tri.neighbors
    N = tets.shape[0]

    # Define the local face indices (which vertices form face j)
    # Face 0: vertices [1, 2, 3], Face 1: [0, 2, 3], etc.
    face_map = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    # 1. Create all faces and their corresponding tet indices
    # We only take faces where the current tet index < neighbor index
    # or the neighbor is -1 (boundary) to avoid double-counting.
    
    tet_ids, face_ids = np.where((np.arange(N)[:, None] < neighbors) | (neighbors == -1))
    faces = tets[tet_ids[:, None], face_map[face_ids]]
    side_index = np.stack([tet_ids, neighbors[tet_ids, face_ids]], axis=1)
    
    return faces, side_index

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

def gaussian_in_circumsphere(cc: torch.Tensor,       # (T,3)
                             r:  torch.Tensor,       # (T,1)
                             k:  int,
                             trunc_sigma: float = 0.3) -> torch.Tensor:
    """
    Draw `k` 3‑D points from N(cc, (trunc_sigma*r)^2 I), truncated so ‖x‑cc‖≤r.

    Returns: (T,k,3)
    """
    T = cc.shape[0]
    # iid standard normal                                                          (T,k,3)
    x  = torch.randn((T, k, 3), device=cc.device)

    # scale by radius*trunc_sigma
    x  = x * (trunc_sigma * r).unsqueeze(1)

    # rejection‑sampling for the few out‑of‑sphere samples ------------------------
    inside = (x.norm(dim=-1, p=2) <= r.unsqueeze(1)).all(dim=-1)
    while not inside.all():
        # re‑draw only the failed rows (≈ 1 % for σ=0.3)
        mask   = ~inside
        num    = mask.sum()
        x[mask] = torch.randn((num, 3), device=x.device) * (trunc_sigma * r[mask])
        inside  = (x.norm(dim=-1) <= r[mask]).all(dim=-1)

    return cc.unsqueeze(1) + x                 # (T,k,3)

def approx_erf(x):
  """An approximation of erf() that is accurate to within 0.007."""
  return torch.sign(x) * torch.sqrt(1 - torch.exp(-(4 / torch.pi) * x**2))

class iNGPDW(nn.Module):
    def __init__(self, 
                 sh_dim=0,
                 scale_multi=0.5,
                 log2_hashmap_size=16,
                 base_resolution=16,
                 per_level_scale=2,
                 L=10,
                 hashmap_dim=4,
                 hidden_dim=64,
                 g_init=1,
                 s_init=1e-4,
                 d_init=0.1,
                 c_init=0.6,
                 density_offset=-4,
                 ablate_downweighing=False,
                 k_samples=1,
                 trunc_sigma=1,
                 additional_attr=0,
                 use_tcnn=False,
                 **kwargs):
        super().__init__()
        self.scale_multi = scale_multi
        self.L = L
        self.dim = hashmap_dim
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.density_offset = density_offset
        self.ablate_downweighing = ablate_downweighing
        self.k_samples = k_samples
        self.trunc_sigma = trunc_sigma
        self.additional_attr = additional_attr
        self.sh_dim = sh_dim

        self.config = dict(
            per_level_scale=per_level_scale,
            n_levels=L,
            otype="HashGrid",
            n_features_per_level=self.dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
        )
        if use_tcnn:
            self.encoding = tcnn.Encoding(3, self.config)
            print("Using TCNN")
            self.compile = False
        else:
            print("Using PyTorch iNGP")
            self.encoding = hashgrid.HashEmbedderOptimized(
                [torch.zeros((3)), torch.ones((3))],
                self.L, n_features_per_level=self.dim,
                log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution,
                finest_resolution=base_resolution*per_level_scale**self.L)
            self.compile = True


        def mk_head(n):
            network = nn.Sequential(
                nn.Linear(self.encoding.n_output_dims, hidden_dim),
                nn.SELU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SELU(inplace=True),
                nn.Linear(hidden_dim, n)
            )
            gain = nn.init.calculate_gain('relu')  # for example, if using ReLU activations
            network.apply(lambda m: init_linear(m, gain))
            return network

        self.network = mk_head(additional_attr)

        self.density_net   = mk_head(1)
        self.color_net     = mk_head(3)
        self.gradient_net  = mk_head(3)
        self.sh_net        = mk_head(sh_dim)
        
        self.g_init = g_init
        self.s_init = s_init
        self.d_init = d_init
        self.c_init = c_init
        self.init_weights()

    def init_weights(self, skip_density=False):
        nets = [self.gradient_net, self.sh_net, self.color_net]
        vals = [self.g_init, self.s_init, self.c_init]
        if not skip_density:
            nets += [self.density_net]
            vals += [self.d_init]
        last = self.network[-1]
        with torch.no_grad():
            # last.weight[4:, :].zero_()
            nn.init.uniform_(last.weight.data, a=-1, b=1)
            last.bias[4:].zero_()
            for network, eps in zip(nets, vals):
                for layer in network[:-1]:
                    if hasattr(layer, 'weight'):
                        nn.init.xavier_uniform_(layer.weight.data, nn.init.calculate_gain('relu'))
                last = network[-1]
                with torch.no_grad():
                    nn.init.uniform_(last.weight.data, a=-eps, b=eps)
                    # nn.init.xavier_uniform_(m.weight, gain)
                    last.bias.zero_()

    def _encode(self, x: torch.Tensor, cr: torch.Tensor):
        output = self.encoding(x)
        output = output.reshape(-1, self.dim, self.L)
        if not self.ablate_downweighing:
            cr = cr.detach() * self.scale_multi
            n = torch.arange(self.L, device=x.device).reshape(1, 1, -1)
            erf_x = safe_div(torch.tensor(1.0, device=x.device),
                            safe_sqrt(self.per_level_scale * 4*n*cr.reshape(-1, 1, 1)))
            scaling = approx_erf(erf_x)
            output = output * scaling
        return output


    def forward(self, x, cr):
        # output = self._encode(x, cr)
        if self.k_samples > 1:
            output = self._encode(x, cr)
            output = output.view(-1, self.k_samples, output.shape[-1]).mean(dim=1)
        else:
            output = self._encode(x, cr)

        h = output.reshape(-1, self.L * self.dim).float()
        # h = self.network(h)
        # sigma = self.d_init * h[:, :1]
        # rgb = self.c_init * h[:, 1:4]
        # field_samples = self.g_init * h[:, 4:7]
        # sh = self.s_init * h[:, 7:7+self.sh_dim].half()
        # attr = h[:, 7+self.sh_dim:]

        sigma = self.density_net(h)
        rgb = self.color_net(h)
        field_samples = self.gradient_net(h)
        sh  = self.sh_net(h).half()

        if self.additional_attr > 0:
            attr = self.network(h)
        else:
            attr = torch.empty((h.shape[0], 0), device=output.device)

        rgb = rgb.reshape(-1, 3, 1) + 0.5
        density = safe_exp(sigma+self.density_offset)
        # density = torch.nn.functional.softplus(sigma+self.density_offset)
        # grd = torch.tanh(field_samples.reshape(-1, 1, 3)) / math.sqrt(3)
        grd = field_samples.reshape(-1, 1, 3)
        grd = grd / ((grd * grd).sum(dim=-1, keepdim=True) + 1).sqrt()
        # grd = rgb * torch.tanh(field_samples.reshape(-1, 3, 3))  # shape (T, 3, 3)
        return density, rgb.reshape(-1, 3), grd, sh, attr

class Model(BaseModel):
    def __init__(self,
                 vertices: torch.Tensor,
                 ext_vertices: torch.Tensor,
                 center: torch.Tensor,
                 scene_scaling: float,
                 density_offset=-1,
                 current_sh_deg=2,
                 max_sh_deg=2,
                 additional_attr=0,
                 ablate_circumsphere=False,
                 ablate_gradient=False,
                 **kwargs):
        super().__init__()
        self.device = vertices.device
        self.density_offset = density_offset
        self.max_sh_deg = max_sh_deg
        self.current_sh_deg = current_sh_deg
        self.dir_offset = torch.tensor([
            [0, 0],
            [math.pi, 0],
            [math.pi/3, math.pi/3],
        ], device=self.device)
        sh_dim = ((1+max_sh_deg)**2-1)*3

        module = iNGPDW(sh_dim, additional_attr=additional_attr, **kwargs)
        self.compile = module.compile
        if module.compile:
            self.backbone = torch.compile(module).to(self.device)
        else:
            self.backbone = module.to(self.device)


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
        self.feature_dim = 7 + additional_attr
        self.alpha = 0
        self.ablate_circumsphere = ablate_circumsphere
        self.ablate_gradient = ablate_gradient

        k_samples = self.backbone.k_samples
        # indices = torch.arange(k_samples, dtype=torch.float32) + 0.5
        # r = (indices / k_samples).pow(1.0/3.0) # Shape [k_samples]
        # golden_ratio = (1. + math.sqrt(5.)) / 2.
        # phi = indices * (2. * math.pi / golden_ratio)  # Azimuthal angle
        # y = 1. - (indices * (2. / k_samples))          # y-coordinate (from 1 down to -1)
        # r_surface = torch.sqrt(1. - y*y)
        # x = r_surface * torch.cos(phi)
        # z = r_surface * torch.sin(phi)
        # surface_dirs = torch.stack([x, y, z], dim=-1) # Shape [k_samples, 3]
        # sphere_pattern = surface_dirs * r.unsqueeze(-1) # Shape [k_samples, 3]

        # sobol_engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        # qmc_samples_01 = sobol_engine.draw(k_samples) + 1e-6
        # ic(qmc_samples_01)
        # sphere_pattern = torch.special.ndtri(qmc_samples_01)
        # ic(sphere_pattern*0.35)

        s = math.sqrt(1/3)
        fixed_pattern = torch.tensor([
            [ 0.0,  0.0,  0.0],
            [   s,    s,    s],
            [   s,   -s,   -s],
            [  -s,    s,   -s],
            [  -s,   -s,    s]
        ], dtype=torch.float32)
        sphere_pattern = fixed_pattern[:k_samples]
        self.register_buffer('sphere_pattern', sphere_pattern.to(self.device))

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

    def compute_batch_features(self, vertices, indices, start, end, circumcenters=None):
        tets = vertices[indices[start:end]]
        if circumcenters is None:
            circumcenter, radius = calculate_circumcenters_torch(tets.double())
        else:
            circumcenter = circumcenters[start:end]
            radius = torch.linalg.norm(circumcenter - vertices[indices[start:end, 0]], dim=-1)
        if self.ablate_circumsphere:
            circumcenter = tets.mean(dim=1)
            # radius = torch.linalg.norm(circumcenter - vertices[indices[start:end, 0]], dim=-1)
        if self.training:
            circumcenter += self.alpha*torch.randn_like(circumcenter) * radius.reshape(-1, 1)
        if self.backbone.k_samples > 1:
            # eps = torch.randn((circumcenter.shape[0], self.backbone.k_samples, 3),
            #                   device=circumcenter.device)

            eps = self.sphere_pattern.unsqueeze(0)
            if self.training:
                rand_mat = torch.randn(3, 3, device=circumcenter.device)
                q, _ = torch.linalg.qr(rand_mat)
                if torch.det(q) < 0:
                    q[:, 0] = -q[:, 0] # Flip one axis
                eps = torch.matmul(q, eps.transpose(-1, -2)).transpose(-1, -2)

            sampled_cc = circumcenter.reshape(-1, 1, 3) + eps * radius.reshape(-1, 1, 1) * self.backbone.trunc_sigma
            sampled_radius = radius.reshape(-1, 1, 1).expand(-1, self.backbone.k_samples, 1)
            normalized = (sampled_cc.detach() - self.center) / self.scene_scaling
            cv, cr = contract_mean_std(normalized.reshape(-1, 3), sampled_radius.reshape(-1) / self.scene_scaling)
            x = (cv/2 + 1)/2
        else:
            normalized = (circumcenter.detach() - self.center) / self.scene_scaling
            cv, cr = contract_mean_std(normalized, radius / self.scene_scaling)
            x = (cv/2 + 1)/2

        output = checkpoint(self.backbone, x, cr, use_reentrant=True)

        # cr = cr.reshape(-1, 1)
        # x, n = pad_for_tinycudann(x, 256)
        # cr, n = pad_for_tinycudann(cr, 256)
        # N = circumcenter.shape[0]
        # output = self.backbone(x, cr.reshape(-1, 1))
        # output = [v[:N] for v in output]
        return circumcenter, *output

    def get_cell_values(self, camera: Camera, mask=None,
                        all_circumcenters=None, radii=None):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices

        sh_dim = (self.max_sh_deg+1)**2 - 1
        features = torch.empty((indices.shape[0], self.feature_dim), device=self.device)
        shs = torch.empty((indices.shape[0], sh_dim, 3), device=self.device)
        start = 0
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])
            circumcenters, density, rgb, grd, sh, attr = self.compute_batch_features(
                vertices, indices, start, end, circumcenters=all_circumcenters)
            if self.ablate_gradient:
                grd = torch.zeros_like(grd)
            centroids = vertices[indices[start:end]].mean(dim=1)
            shs[start:end] = sh.reshape(-1, sh_dim, 3)
            dvrgbs = activate_output(camera.camera_center.to(self.device),
                                     density, rgb, grd,
                                     sh.reshape(-1, sh_dim, 3),
                                     attr,
                                     indices[start:end],
                                     centroids,
                                     vertices.detach(),
                                     self.current_sh_deg, self.max_sh_deg)
            features[start:end] = dvrgbs
        return shs, features

    @staticmethod
    def init_rand_from_pcd(point_cloud, cameras, device, max_sh_deg,
                      num_points=10000, **kwargs):
        torch.manual_seed(2)

        ccenters = torch.stack([c.camera_center.reshape(3) for c in cameras], dim=0).to(device)
        center = ccenters.mean(dim=0)
        scaling = torch.linalg.norm(ccenters - center.reshape(1, 3), dim=1, ord=torch.inf).max()
        print(f"Scene scaling: {scaling}. Center: {center}")

        vertices = torch.as_tensor(point_cloud.points).float()

        # add sphere
        pcd_scaling = torch.linalg.norm(vertices - center.cpu().reshape(1, 3), dim=1, ord=2).max()
        new_radius = pcd_scaling.cpu().item()

        vertices = sample_uniform_in_sphere(num_points, 3, base_radius=0, radius=new_radius, device='cpu') + center.reshape(1, 3).cpu()

        num_ext = 1000
        ext_vertices = fibonacci_spiral_on_sphere(num_ext, new_radius, device='cpu') + center.reshape(1, 3).cpu()
        num_ext = ext_vertices.shape[0]

        model = Model(vertices.cuda(), ext_vertices, center, scaling,
                      max_sh_deg=max_sh_deg, **kwargs)
        return model

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
        # vertices = torch.cat([vertices, ext_vertices], dim=0)
        # ext_vertices = torch.empty((0, 3))

        model = Model(vertices.cuda(), ext_vertices, center, scaling,
                      max_sh_deg=max_sh_deg, **kwargs)
        return model

    @staticmethod
    def load_ckpt(path: Path, device):
        ckpt_path = path / "ckpt.pth"
        config_path = path / "config.json"
        config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)
        vertices = ckpt['interior_vertices']
        indices = ckpt["indices"]  # shape (N,4)
        del ckpt["indices"]
        if 'empty_indices' in ckpt:
            empty_indices = ckpt['empty_indices']
            del ckpt['empty_indices']
        else:
            empty_indices = torch.empty((0, 4), dtype=indices.dtype, device=indices.device)
        print(f"Loaded {vertices.shape[0]} vertices")
        ext_vertices = ckpt['ext_vertices']
        model = Model(vertices.to(device), ext_vertices, ckpt['center'], ckpt['scene_scaling'], **config.as_dict())
        model.load_state_dict(ckpt)
        model.min_t = config.min_t
        model.indices = torch.as_tensor(indices).cuda()
        model.empty_indices = torch.as_tensor(empty_indices).cuda()
        return model

    def calc_tet_density(self):
        densities = []
        verts = self.vertices
        for start in range(0, self.indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.indices.shape[0])
            
            _, density, _, _, _, _ = self.compute_batch_features(verts, self.indices, start, end)

            densities.append(density.reshape(-1))
        return torch.cat(densities)

    def compute_features(self, offset=False):
        vertices = self.vertices
        indices = self.indices
        cs, ds, rs, gs, ss = [], [], [], [], []
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])

            circumcenters, density, rgb, grd, sh, _ = self.compute_batch_features(vertices, indices, start, end)
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

    @property
    def vertices(self):
        verts = self.interior_vertices
        return torch.cat([verts, self.ext_vertices])

    def sh_up(self):
        self.current_sh_deg = min(self.max_sh_deg, self.current_sh_deg+1)

    def compute_adjacency(self):
        self.faces, self.side_index = get_tet_adjacency(self.indices)

    @torch.no_grad()
    def update_triangulation(self, high_precision=False, density_threshold=0.0, alpha_threshold=0.0):
        torch.cuda.empty_cache()
        verts = self.vertices
        if high_precision:
            d = Delaunay(verts.detach().cpu().numpy())
            # faces, side_index = get_tet_adjacency_from_scipy(d)
            # self.faces = torch.as_tensor(faces).cuda()
            # self.side_index = torch.as_tensor(side_index).cuda()
            indices_np = d.simplices.astype(np.int32)
            # self.indices = torch.tensor(indices_np, device=verts.device).int().cuda()
        else:
            v = Del(verts.shape[0])
            indices_np, prev = v.compute(verts.detach().cpu().double())
            indices_np = indices_np.clone().numpy()
            valid_mask = (indices_np >= 0) & (indices_np < verts.shape[0])
            indices_np = indices_np[valid_mask.all(axis=1)]
            del prev
        

        # Ensure volume is positive
        indices = torch.as_tensor(indices_np).cuda()
        vols = topo_utils.tet_volumes(verts[indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            indices[reverse_mask] = indices[reverse_mask][:, [1, 0, 2, 3]]

        # Cull tets with low density
        # self.full_indices = indices.clone()
        self.indices = indices
        # if not high_precision:
        #     self.compute_adjacency()
        if density_threshold > 0 or alpha_threshold > 0:
            tet_density = self.calc_tet_density()
            tet_alpha = self.calc_tet_alpha(mode="min", density=tet_density)
            mask = (tet_density > density_threshold) | (tet_alpha > alpha_threshold)
            self.empty_indices = self.indices[~mask]
            self.indices = self.indices[mask]
            self.mask = mask
        else:
            self.empty_indices = torch.empty((0, 4), dtype=self.indices.dtype, device='cuda')
            # self.mask = torch.ones((self.full_indices.shape[0]), dtype=bool, device='cuda')
            
        torch.cuda.empty_cache()

    def __len__(self):
        return self.vertices.shape[0]
        

    def compute_weight_decay(self):
        if self.compile:
            return sum([(embed.weight**2).mean() for embed in self.backbone.encoding.embeddings])
        else:
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
                 lr_delay: int = 500,
                 freeze_start: int = 10000,
                 vert_lr_delay: int = 500,

                 spike_duration: int = 20,
                 densify_interval: int = 500,
                 densify_end: int = 15000,
                 midpoint: int = 2000,

                 percent_alpha: float = 0.02,

                 **kwargs):

        self.optim = optim.CustomAdam([
            {"params": model.backbone.encoding.parameters(), "lr": encoding_lr, "name": "encoding"},
        ], ignore_param_list=["encoding", "network"], betas=[0.9, 0.999], eps=1e-15)

        self.net_optim = optim.CustomAdam([
            {"params": model.backbone.network.parameters(),   "lr": network_lr,  "name": "network"},
            {"params": model.backbone.density_net.parameters(),   "lr": network_lr,  "name": "density"},
            {"params": model.backbone.color_net.parameters(),     "lr": network_lr,    "name": "color"},
            # {"params": model.backbone.density_color_net.parameters(),     "lr": network_lr,    "name": "color"},
            {"params": model.backbone.gradient_net.parameters(),  "lr": network_lr, "name": "gradient"},
            {"params": model.backbone.sh_net.parameters(),        "lr": network_lr,       "name": "sh"},
        ], ignore_param_list=[], betas=[0.9, 0.999], eps=1e-15)
        self.vert_lr_multi = float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.interior_vertices], "lr": self.vert_lr_multi*vertices_lr, "name": "interior_vertices"},
        ])
        self.model = model

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
            # encoding_lr, encoding_lr)
            encoding_lr, encoding_lr)

        self.vertices_lr = self.vert_lr_multi*vertices_lr
        self.final_vertices_lr = self.vert_lr_multi*final_vertices_lr
        base_vertex_scheduler = get_expon_lr_func(lr_init=self.vertices_lr,
                                                lr_final=self.final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_multi,
                                                max_steps=freeze_start,
                                                lr_delay_steps=vert_lr_delay)

        self.vertex_scheduler_args = base_vertex_scheduler
        self.vertex_scheduler_args = SpikingLR(
            spike_duration, freeze_start, base_vertex_scheduler,
            midpoint, densify_interval, densify_end,
            # self.vertices_lr, self.final_vertices_lr)
            self.vertices_lr, self.vertices_lr)
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
                self.vertices_lr = lr
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
    def split(self, split_point, **kwargs):
        self.add_points(split_point)

    def main_step(self):
        self.optim.step()
        self.net_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad(set_to_none=True)
        self.net_optim.zero_grad(set_to_none=True)

    @property
    def sh_optim(self):
        return None

    def regularizer(self, render_pkg, lambda_weight_decay, **kwargs):
        # weight_decay = self.weight_decay * sum([(embed.weight**2).mean() for embed in self.model.backbone.encoding.embeddings])
        weight_decay = lambda_weight_decay * self.model.compute_weight_decay()

        # if self.lambda_density > 0 or self.lambda_tv > 0:
        #     density = self.model.calc_tet_density()
        #     density_loss = (self.model.calc_tet_area().detach() * density).sum()
        #     if self.lambda_tv > 0:
        #         diff  = density[self.pairs[:,0]] - density[self.pairs[:,1]]
        #         tv_loss  = (self.face_area * diff.abs())
        #         tv_loss  = tv_loss.sum() / self.face_area.sum()
        #     else:
        #         tv_loss = 0
        # else:
        #     density_loss = 0
        #     tv_loss = 0

        return weight_decay# + self.lambda_tv * tv_loss + self.lambda_density * density_loss

    def update_triangulation(self, **kwargs):
        self.model.update_triangulation(**kwargs)

    def prune(self, diff_threshold, **kwargs):
        if diff_threshold <= 0:
            return
        density = self.model.calc_tet_density()
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

    def clip_grad_norm_(self, max_norm):
        torch.nn.utils.clip_grad_norm_(self.model.backbone.parameters(), max_norm)
        torch.nn.utils.clip_grad_norm_(self.model.interior_vertices, max_norm)
