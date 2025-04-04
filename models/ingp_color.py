import torch
import time
import math
from data.camera import Camera
from utils import optim
from sh_slang.eval_sh import eval_sh
# from delaunay_rasterization.internal.alphablend_tiled_slang import AlphaBlendTiledRender, render_alpha_blend_tiles_slang_raw
# from delaunay_rasterization.internal.render_grid import RenderGrid
# from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from gDel3D.build.gdel3d import Del
from torch import nn
from icecream import ic
from utils.train_util import RGB2SH
import tinycudann as tcnn
from utils.topo_utils import calculate_circumcenters_torch, project_points_to_tetrahedra
from utils.safe_math import safe_exp, safe_div, safe_sqrt, safe_pow, safe_cos, safe_sin, remove_zero, safe_arctan2
from utils.contraction import contract_mean_std
from utils.contraction import contract_points, inv_contract_points
from utils.train_util import RGB2SH, safe_exp, get_expon_lr_func, sample_uniform_in_sphere
from utils import topo_utils
from utils.graphics_utils import l2_normalize_th
from typing import List
from utils import hashgrid
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
from plyfile import PlyData, PlyElement
from pathlib import Path
import numpy as np
from utils.args import Args
import tinyplypy
from utils.phong_shading import to_sphere, activate_lights, light_function, compute_tet_color

def init_weights(m, gain):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@torch.jit.script
def pre_calc_cell_values(vertices, indices, center, scene_scaling: float, per_level_scale: float, L: int, scale_multi: float, base_resolution: float):
    device = vertices.device
    tets = vertices[indices]
    circumcenter, radius = calculate_circumcenters_torch(tets.double())
    clipped_circumcenter = project_points_to_tetrahedra(circumcenter.float(), tets)
    normalized = (clipped_circumcenter - center) / scene_scaling
    cv, cr = contract_mean_std(normalized, radius / scene_scaling)
    cr = cr.float() * scale_multi
    n = torch.arange(L, device=device).reshape(1, 1, -1)
    erf_x = safe_div(torch.tensor(1.0, device=device), safe_sqrt(per_level_scale * 4*n*cr.reshape(-1, 1, 1)))
    scaling = torch.erf(erf_x)
    # sphere_area = 4/3*math.pi*cr**3
    # scaling = safe_div(base_resolution * per_level_scale**n, sphere_area.reshape(-1, 1, 1)).clip(max=1)
    return clipped_circumcenter, cv.float(), normalized, scaling

@torch.jit.script
def compute_vertex_colors_from_field(triangle_verts, field_samples, circumcenters):
    """
    Compute per-vertex colors for each triangle.
    
    For each vertex:
      color = base + dot(gradient, (vertex - circumcenter))
    
    - The first 3 coefficients of field_samples provide the base color.
    - The next 6 coefficients (reshaped as (3,2)) define the gradients.
    """
    base = (field_samples[:, :3]) + 0.5  # shape (T, 3)
    gradients = base.reshape(-1, 3, 1).clip(min=0) * torch.tanh(field_samples[:, 3:12].reshape(-1, 3, 3))  # shape (T, 3, 3)
    offsets = triangle_verts - circumcenters[:, None, :]  # shape (T, 4, 3)
    offsets = l2_normalize_th(offsets)
    grad_contrib = torch.einsum('tcd,tvd->tvc', gradients, offsets)
    vertex_colors = base[:, None, :] + grad_contrib
    return vertex_colors

class Model(nn.Module):
    def __init__(self,
                 vertices: torch.Tensor,
                 vertex_lights: torch.Tensor,
                 center: torch.Tensor,
                 scene_scaling: float,
                 scale_multi=0.5,
                 log2_hashmap_size=16,
                 base_resolution=16,
                 per_level_scale=2,
                 L=10,
                 hashmap_dim=4,
                 hidden_dim=64,
                 contract_vertices=True,
                 density_offset=-1,
                 max_lights=2,
                 num_lights=2,
                 light_offset=-3,
                 **kwargs):
        super().__init__()
        self.scale_multi = scale_multi
        self.L = L
        self.dim = hashmap_dim
        self.device = vertices.device
        self.density_offset = density_offset
        self.num_lights = num_lights
        self.max_lights = max_lights
        self.light_offset = light_offset
        self.dir_offset = torch.tensor([
            [0, 0],
            [math.pi, 0],
        ], device=self.device)
        config = dict(
            otype="HashGrid",
            n_levels=self.L,
            n_features_per_level=self.dim,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            # base_resolution=32,
            per_level_scale=per_level_scale
        )
        self.per_level_scale = per_level_scale
        # self.encoding = tcnn.Encoding(3, config).to(self.device)
        self.base_resolution = base_resolution
        self.chunk_size = 508576
        # self.chunk_size = 508576

        self.encoding = torch.compile(hashgrid.HashEmbedderOptimized(
            [torch.zeros((3), device=self.device), torch.ones((3), device=self.device)],
            self.L, n_features_per_level=self.dim,
            log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution,
            finest_resolution=base_resolution*per_level_scale**self.L)).to(self.device)

        self.network = torch.compile(nn.Sequential(
            nn.Linear(self.encoding.n_output_dims, hidden_dim),
            # nn.ReLU(inplace=True),
            nn.SELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(inplace=True),
            nn.SELU(inplace=True),
            nn.Linear(hidden_dim, 1+12)
        )).to(self.device)
        gain = nn.init.calculate_gain('relu')  # for example, if using ReLU activations
        self.network.apply(lambda m: init_weights(m, gain))
        self.vertex_lights = nn.Parameter(vertex_lights)

        self.register_buffer('center', center.reshape(1, 3))
        self.register_buffer('scene_scaling', torch.tensor(float(scene_scaling), device=self.device))
        self.contract_vertices = contract_vertices
        if self.contract_vertices:
            self.contracted_vertices = nn.Parameter(self.contract(vertices.detach()))
        else:
            self.contracted_vertices = nn.Parameter(vertices.detach())
        self.update_triangulation()

    def load_ckpt(path: Path, device):
        data_dict = tinyplypy.read_ply(str(path / "ckpt.ply"))
        tet_data = data_dict["tetrahedron"]
        indices = tet_data["vertex_indices"]  # shape (N,4)
        ckpt_path = path / "ckpt.pth"
        config_path = path / "alldata.json"
        config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)
        vertices = ckpt['contracted_vertices']
        print(f"Loaded {vertices.shape[0]} vertices")
        temp = config.contract_vertices
        config.contract_vertices = False
        lights = ckpt['vertex_lights']
        model = Model(vertices.to(device), lights, ckpt['center'], ckpt['scene_scaling'], **config.as_dict())
        model.load_state_dict(ckpt)
        model.contract_vertices = temp
        model.min_t = model.scene_scaling * config.base_min_t
        model.indices = torch.as_tensor(indices).cuda()
        model.boundary_tets = torch.zeros((indices.shape[0]), dtype=bool, device='cuda')
        return model

    def calc_vert_alpha(self):
        tet_alphas = self.calc_tet_alpha()
        vertex_alpha = torch.full((self.vertices.shape[0],), 0.0, device=self.device)
        indices = self.indices.long()

        reduce_type = "amax"
        vertex_alpha.scatter_reduce_(dim=0, index=indices[..., 0], src=tet_alphas, reduce=reduce_type)
        vertex_alpha.scatter_reduce_(dim=0, index=indices[..., 1], src=tet_alphas, reduce=reduce_type)
        vertex_alpha.scatter_reduce_(dim=0, index=indices[..., 2], src=tet_alphas, reduce=reduce_type)
        vertex_alpha.scatter_reduce_(dim=0, index=indices[..., 3], src=tet_alphas, reduce=reduce_type)
        return vertex_alpha

    def calc_tet_alpha(self):
        # Compute the density mask in chunks
        alpha_list = []
        start = 0
        
        verts = self.vertices
        # while start < self.indices.shape[0]:
        for start in range(0, self.indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.indices.shape[0])
            
            _, _, output = self.compute_batch_features(verts, self.indices, start, end)

            density = safe_exp(output[:, 0]+self.density_offset)
            indices_chunk = self.indices[start:end]
            v0, v1, v2, v3 = verts[indices_chunk[:, 0]], verts[indices_chunk[:, 1]], verts[indices_chunk[:, 2]], verts[indices_chunk[:, 3]]
            
            edge_lengths = torch.stack([
                torch.norm(v0 - v1, dim=1), torch.norm(v0 - v2, dim=1), torch.norm(v0 - v3, dim=1),
                torch.norm(v1 - v2, dim=1), torch.norm(v1 - v3, dim=1), torch.norm(v2 - v3, dim=1)
            ], dim=0).max(dim=0)[0]
            
            # Compute the maximum possible alpha using the largest edge length
            alpha = 1 - torch.exp(-density * edge_lengths)
            alpha_list.append(alpha)
            del edge_lengths, density
        
        # Concatenate mask and apply it
        alphas = torch.cat(alpha_list, dim=0)
        return alphas

    @torch.no_grad
    def extract_mesh(self, path, alpha_threshold=0.5):
        verts = self.vertices
        v = Del(verts.shape[0])
        indices_np, tri_output = v.compute(verts.detach().cpu())
        indices_np = indices_np.numpy()
        inf_mask = (indices_np < verts.shape[0]).all(axis=1)
        indices_np = indices_np[inf_mask]
        
        # Convert to tensor and move to CUDA
        self.indices = torch.as_tensor(indices_np).cuda()
        mask = self.calc_tet_alpha() > alpha_threshold
        
        full_mask = torch.zeros((inf_mask.shape[0]), dtype=bool)
        full_mask[inf_mask] = mask.cpu()
        meshes = tri_output.extract_meshes(verts.detach().cpu(), full_mask)['meshes']
        sizes = [len(mesh['x']) for mesh in meshes]

        
        largest = meshes[np.argmax(sizes)]
        tinyplypy.write_ply(str(path), largest, is_binary=False)

    @torch.no_grad
    def save2ply(self, path):
        path.parent.mkdir(exist_ok=True, parents=True)

        xyz = self.vertices.detach().cpu().numpy().astype(np.float32)  # shape (num_vertices, 3)

        vertex_dict = {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
        }
        vertex_lights = self.vertex_lights.detach().cpu().numpy().astype(np.float32).reshape(-1, (self.num_lights+1)**2 - 1, 3)
        for i in range(vertex_lights.shape[1]):
            vertex_dict[f"sh_{i+1}_r"] = np.ascontiguousarray(vertex_lights[:, i, 0])
            vertex_dict[f"sh_{i+1}_g"] = np.ascontiguousarray(vertex_lights[:, i, 1])
            vertex_dict[f"sh_{i+1}_b"] = np.ascontiguousarray(vertex_lights[:, i, 2])

        N = self.indices.shape[0]
        densities = np.zeros((N), dtype=np.float32)
        lights = np.zeros((N, 4, 3))

        vertices = self.vertices
        indices = self.indices
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])

            circumcenters, _, output = self.compute_batch_features(vertices, indices, start, end)
            density = safe_exp(output[:, 0:1]+self.density_offset)
            field_samples = output[:, 1:]
            vcolors = compute_vertex_colors_from_field(
                vertices[indices[start:end]].detach(), field_samples.float(), circumcenters.float().detach())
            vcolors = vcolors.cpu().numpy().astype(np.float32)
            density = density.cpu().numpy().astype(np.float32)
            lights[start:end] = vcolors
            densities[start:end] = density

        tetra_dict = {}
        tetra_dict["vertex_indices"] = self.indices.cpu().numpy().astype(np.int32)
        tetra_dict["s"] = np.ascontiguousarray(densities)
        for i, co in enumerate(["x", "y", "z", "w"]):
            for j, c in enumerate(["r", "g", "b"]):
                tetra_dict[f"{c}_{co}"]         = np.ascontiguousarray(lights[:, i, j])

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
            return self.inv_contract(self.contracted_vertices)
        else:
            return self.contracted_vertices

    @staticmethod
    def init_from_pcd(point_cloud, cameras, device, num_lights, **kwargs):
        torch.manual_seed(2)
        N = point_cloud.points.shape[0]
        # N = 1000
        vertices = torch.as_tensor(point_cloud.points)[:N]

        ccenters = torch.stack([c.camera_center.reshape(3) for c in cameras], dim=0).to(device)
        center = ccenters.mean(dim=0)
        scaling = torch.linalg.norm(ccenters - center.reshape(1, 3), dim=1, ord=torch.inf).max()

        # vertices = vertices + torch.randn(*vertices.shape) * 1e-3
        # v = Del(vertices.shape[0])
        # indices_np, prev = v.compute(vertices.detach().cpu())
        # indices_np = indices_np.numpy()
        # indices_np = indices_np[(indices_np < vertices.shape[0]).all(axis=1)]
        # vertices = vertices[indices_np].mean(dim=1)
        # vertices = vertices + torch.randn(*vertices.shape) * 1e-3

        repeats = 3
        vertices = vertices.reshape(-1, 1, 3).expand(-1, repeats, 3)
        vertices = vertices + torch.randn(*vertices.shape) * 1e-1
        vertices = vertices.reshape(-1, 3)

        # add sphere
        pcd_scaling = torch.linalg.norm(vertices - center.cpu().reshape(1, 3), dim=1, ord=2).max()
        # x = l2_normalize_th(torch.randn((1000, 3))) * 1.2 * pcd_scaling.cpu() + center.reshape(1, 3).cpu()
        # vertices = torch.cat([vertices, x], dim=0)
        # vertex_base_color = torch.cat([vertex_base_color, torch.zeros_like(x).to(vertex_base_color.device)], dim=0)
        vertex_lights = torch.zeros((vertices.shape[0], ((num_lights+1)**2-1)*3)).to(device)

        vertices = nn.Parameter(vertices.cuda())
        model = Model(vertices, vertex_lights, center, scaling, **kwargs)
        return model

    def sh_up(self):
        self.max_lights = min(self.num_lights, self.max_lights+1)

    def compute_batch_features(self, vertices, indices, start, end):
        circumcenter, cv, normalized, scaling =  pre_calc_cell_values(
            vertices, indices[start:end], self.center, self.scene_scaling,
            self.per_level_scale, self.L, self.scale_multi, self.base_resolution)
        x = (cv/2 + 1)/2
        output = checkpoint(self.encoding, x, use_reentrant=True).float()
        output = output.reshape(-1, self.dim, self.L)

        output = output * scaling
        output = checkpoint(self.network, output.reshape(-1, self.L * self.dim), use_reentrant=True)
        return circumcenter, normalized, output

    def update_triangulation(self, alpha_threshold=0.05/255):
        verts = self.vertices
        v = Del(verts.shape[0])
        indices_np, prev = v.compute(verts.detach().cpu())
        indices_np = indices_np.numpy()
        indices_np = indices_np[(indices_np < verts.shape[0]).all(axis=1)]
        
        # Convert to tensor and move to CUDA
        self.indices = torch.as_tensor(indices_np).cuda()
        
        if alpha_threshold > 0:
            mask = self.calc_tet_alpha() > alpha_threshold
            self.indices = self.indices[mask]
            
            del indices_np, prev, mask
        else:
            del indices_np, prev
        
    def get_cell_values(self, camera: Camera, mask=None,
                        circumcenters=None, radii=None):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices

        vertex_color_raw = eval_sh(
            vertices,
            torch.zeros((vertices.shape[0], 3), device=vertices.device),
            self.vertex_lights.reshape(-1, (self.num_lights+1)**2 - 1, 3),
            camera.camera_center.to(self.device),
            self.max_lights) - 0.5

        outputs = []
        normed_cc = []
        start = 0
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])
            circumcenters, normalized, output = self.compute_batch_features(vertices, indices, start, end)
            density = safe_exp(output[:, 0:1]+self.density_offset)
            field_samples = output[:, 1:]
            vcolors = compute_vertex_colors_from_field(
                vertices[indices[start:end]].detach(), field_samples.float(), circumcenters.float().detach())
            vcolors = torch.nn.functional.softplus(vertex_color_raw[indices[start:end]] + vcolors, beta=10)
            vcolors = vcolors.reshape(-1, 12)
            features = torch.cat([density, vcolors], dim=1)
            normed_cc.append(normalized)
            outputs.append(features)
        features = torch.cat(outputs, dim=0)
        normed_cc = torch.cat(normed_cc, dim=0)
        return normed_cc, features

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
                 net_weight_decay=1e-3,
                 split_std: float = 0.5,
                 lights_lr: float=1e-4,
                 vertices_beta: List[float] = [0.9, 0.99],
                 lr_delay: int = 500,
                 vert_lr_delay: int = 500,
                 **kwargs):
        self.weight_decay = weight_decay
        self.optim = optim.CustomAdam([
            {"params": model.encoding.parameters(), "lr": encoding_lr, "name": "encoding"},
        ], ignore_param_list=["encoding", "network"], betas=[0.9, 0.99], eps=1e-15)
        self.net_optim = optim.CustomAdam([
            {"params": model.network.parameters(), "lr": network_lr, "name": "network"},
        ], ignore_param_list=["encoding", "network"], betas=[0.9, 0.99], weight_decay=net_weight_decay)
        self.vert_lr_multi = 1 if model.contract_vertices else float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.contracted_vertices], "lr": self.vert_lr_multi*vertices_lr, "name": "contracted_vertices"},
        ])
        self.lights_optim = optim.CustomAdam([
            {"params": [model.vertex_lights], "lr": lights_lr, "name": "vertex_lights"},
        ])
        self.model = model
        self.tracker_n = 0
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None
        self.split_std = split_std

        self.net_scheduler_args = get_expon_lr_func(lr_init=network_lr,
                                                lr_final=final_network_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=10000)
        self.encoder_scheduler_args = get_expon_lr_func(lr_init=encoding_lr,
                                                lr_final=final_encoding_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=10000)
        self.vertex_lr = self.vert_lr_multi*vertices_lr
        self.vertex_scheduler_args = get_expon_lr_func(lr_init=self.vertex_lr,
                                                lr_final=self.vert_lr_multi*final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_multi,
                                                max_steps=10000,
                                                lr_delay_steps=vert_lr_delay)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.net_optim.param_groups:
            if param_group["name"] == "network":
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

    def update_ema(self):
        self.ema.update()

    def add_points(self, new_verts: torch.Tensor, new_vertex_lights: torch.Tensor):
        if self.model.contract_vertices:
            new_verts = self.model.contract(new_verts)
        self.model.contracted_vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            contracted_vertices = new_verts
        ))['contracted_vertices']
        self.model.vertex_lights = self.lights_optim.cat_tensors_to_optimizer(dict(
            vertex_lights = new_vertex_lights
        ))['vertex_lights']
        self.model.update_triangulation()

    def remove_points(self, mask: torch.Tensor):
        new_tensors = self.optim.prune_optimizer(mask)
        self.model.contracted_vertices = self.vertex_optim.prune_optimizer(mask)['contracted_vertices']
        self.model.update_triangulation()

    @torch.no_grad()
    def split(self, clone_indices, split_mode, alpha_threshold):
        device = self.model.device
        clone_vertices = self.model.vertices[clone_indices]

        if split_mode == "circumcenter":
            circumcenters, radius = topo_utils.calculate_circumcenters_torch(clone_vertices)
            radius = radius.reshape(-1, 1)
            circumcenters = circumcenters.reshape(-1, 3)
            sphere_loc = sample_uniform_in_sphere(circumcenters.shape[0], 3).to(device)
            r = torch.randn((clone_indices.shape[0], 1), device=self.model.device)
            r[r.abs() < 1e-2] = 1e-2
            sampled_radius = (r * self.split_std + 1) * radius
            new_vertex_location = l2_normalize_th(sphere_loc) * sampled_radius + circumcenters
            new_vertex_lights = (self.model.vertex_lights[clone_indices] * barycentric_weights).sum(dim=1)
        elif split_mode == "barycentric":
            barycentric = torch.rand((clone_indices.shape[0], clone_indices.shape[1], 1), device=device).clip(min=0.01, max=0.99)
            barycentric_weights = barycentric / (1e-3+barycentric.sum(dim=1, keepdim=True))
            new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights).sum(dim=1)
            new_vertex_lights = (self.model.vertex_lights[clone_indices] * barycentric_weights).sum(dim=1)
        else:
            raise Exception(f"Split mode: {split_mode} not supported")
        self.add_points(new_vertex_location, new_vertex_lights)
        mask = self.model.calc_vert_alpha() < alpha_threshold
        print(f"Pruned: {mask.sum()} points")
        self.remove_points(~mask)
        del mask, alpha_threshold

    def main_step(self):
        self.optim.step()
        self.net_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad()
        self.net_optim.zero_grad()

    @property
    def sh_optim(self):
        return self.lights_optim

    def regularizer(self):
        return self.weight_decay * sum([(embed.weight**2).mean() for embed in self.model.encoding.embeddings])