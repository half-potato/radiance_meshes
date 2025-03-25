import torch
import time
import math
from data.camera import Camera
from utils import optim
# from delaunay_rasterization.internal.alphablend_tiled_slang import AlphaBlendTiledRender, render_alpha_blend_tiles_slang_raw
# from delaunay_rasterization.internal.render_grid import RenderGrid
# from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from gDel3D.build.gdel3d import Del
from torch import nn
from icecream import ic
from utils.train_util import RGB2SH
import tinycudann as tcnn
from utils.topo_utils import calculate_circumcenters_torch
from utils.safe_math import safe_exp, safe_div, safe_sqrt, safe_pow, safe_cos, safe_sin, remove_zero, safe_arctan2
from utils.contraction import contract_mean_std
from utils.contraction import contract_points, inv_contract_points
from utils.train_util import RGB2SH, safe_exp, get_expon_lr_func, sample_uniform_in_sphere
from utils import topo_utils
from utils.graphics_utils import l2_normalize_th
from typing import List
from utils import hashgrid
from torch.utils.checkpoint import checkpoint
from pathlib import Path
import numpy as np
from utils.args import Args
import tinyplypy
from utils.phong_shading import to_sphere, activate_lights, light_function, compute_vert_color
from sh_slang.eval_sh import eval_sh

MAX_DENSITY = 1000000

def init_weights(m, gain):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

@torch.jit.script
def pre_calc_cell_values(vertices, indices, center, scene_scaling: float, per_level_scale: float, L: int, scale_multi: float, base_resolution: float):
    device = vertices.device
    circumcenter, radius = calculate_circumcenters_torch(vertices[indices].double())
    normalized = (circumcenter - center) / scene_scaling
    cv, cr = contract_mean_std(normalized, radius / scene_scaling)
    cr = cr.float() * scale_multi
    n = torch.arange(L, device=device).reshape(1, 1, -1)
    erf_x = safe_div(torch.tensor(1.0, device=device), safe_sqrt(per_level_scale * 4*n*cr.reshape(-1, 1, 1)))
    scaling = torch.erf(erf_x)
    # sphere_area = 4/3*math.pi*cr**3
    # scaling = safe_div(base_resolution * per_level_scale**n, sphere_area.reshape(-1, 1, 1)).clip(max=1)
    return cv.float(), scaling

class Model(nn.Module):
    def __init__(self,
                 vertices: torch.Tensor,
                 vertex_base_color: torch.Tensor,
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
                 num_lights=2,
                 max_lights=0,
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
        self.per_level_scale = per_level_scale
        # self.encoding = tcnn.Encoding(3, config).to(self.device)
        self.base_resolution = base_resolution
        self.chunk_size = 508576

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
            nn.Linear(hidden_dim, 1)
        )).to(self.device)
        gain = nn.init.calculate_gain('relu')  # for example, if using ReLU activations
        self.network.apply(lambda m: init_weights(m, gain))

        self.vertex_base_color = nn.Parameter(vertex_base_color)
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
        ckpt_path = path / "ckpt.pth"
        config_path = path / "alldata.json"
        config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)
        vertices = ckpt['contracted_vertices']
        print(f"Loaded {vertices.shape[0]} vertices")
        temp = config.contract_vertices
        config.contract_vertices = False
        base_colors = ckpt['vertex_base_color']
        lights = ckpt['vertex_lights']
        model = Model(vertices.to(device), base_colors, lights, ckpt['center'], ckpt['scene_scaling'], **config.as_dict())
        model.load_state_dict(ckpt)
        model.contract_vertices = temp
        return model

    @torch.no_grad
    def save2ply(self, path, sample_camera):
        """
        Convert the old save2ply function (which used 'plyfile'),
        so it uses our new tinyply-based library via pybind11.
        """

        # Ensure the output directory exists
        path.parent.mkdir(exist_ok=True, parents=True)

        # 1. Gather vertex positions
        xyz = self.vertices.detach().cpu().numpy().astype(np.float32)
        rgb = self.vertex_base_color.detach().cpu().numpy().astype(np.float32)

        # For tinyply, we store them as one dictionary for the "vertex" element:
        #   { "x": array([...]), "y": ..., "z": ... }
        # Make sure to cast to a concrete dtype (e.g. float32).
        vertex_dict = {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            # "r": rgb[:, 0],
            # "g": rgb[:, 1],
            # "b": rgb[:, 2],
        }
        vertex_dict[f"sh_{0}_r"] = np.ascontiguousarray(rgb[:, 0])
        vertex_dict[f"sh_{0}_g"] = np.ascontiguousarray(rgb[:, 1])
        vertex_dict[f"sh_{0}_b"] = np.ascontiguousarray(rgb[:, 2])
        vertex_lights = self.vertex_lights.detach().cpu().numpy().astype(np.float32).reshape(-1, (self.num_lights+1)**2 - 1, 3)
        for i in range(vertex_lights.shape[1]):
            vertex_dict[f"sh_{i+1}_r"] = np.ascontiguousarray(vertex_lights[:, i, 0])
            vertex_dict[f"sh_{i+1}_g"] = np.ascontiguousarray(vertex_lights[:, i, 1])
            vertex_dict[f"sh_{i+1}_b"] = np.ascontiguousarray(vertex_lights[:, i, 2])
        # for i in range(self.num_lights):
        #     offset = i*6
        #     vertex_dict[f"l{i}_r"]         = np.ascontiguousarray(vertex_lights[:, offset + 0])
        #     vertex_dict[f"l{i}_g"]         = np.ascontiguousarray(vertex_lights[:, offset + 1])
        #     vertex_dict[f"l{i}_b"]         = np.ascontiguousarray(vertex_lights[:, offset + 2])
        #     vertex_dict[f"l{i}_roughness"] = np.ascontiguousarray(vertex_lights[:, offset + 3])
        #     vertex_dict[f"l{i}_phi"]       = np.ascontiguousarray(vertex_lights[:, offset + 4])
        #     vertex_dict[f"l{i}_theta"]     = np.ascontiguousarray(vertex_lights[:, offset + 5])

        # 2. Compute your RGBA / lighting data per tetrahedron
        #    (same logic as in your code: iterative chunking, gather, etc.)
        N = self.indices.shape[0]
        densities = np.zeros((N), dtype=np.float32)

        vertices = self.vertices
        indices = self.indices
        # e.g., chunk-based processing (adapt as you did in your original code)
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])

            output = self.compute_batch_features(vertices, indices, start, end)
            density = safe_exp(output[:, 0] + self.density_offset)
            densities[start:end] = density.cpu().numpy().astype(np.float32)
        densities = np.where(self.boundary_tets.cpu().numpy(), MAX_DENSITY, densities).astype(np.float32)

        # 3. Build the dictionary for your "tetrahedron" element
        #    'vertex_indices' is a 2D array (N,4) for the tetra indices
        #    plus 'r', 'g', 'b', 's', and the per-light properties
        tetra_dict = {}

        # Indices: shape (N, 4). Must be stored as an unsigned int (common for face/tet indices).
        tetra_dict["vertex_indices"] = self.indices.cpu().numpy().astype(np.int32)

        # The first 4 columns in rgbs are [r, g, b, s].
        tetra_dict["density"] = np.ascontiguousarray(densities)

        # 4. Final data structure:
        # data_dict[element_name][property_name] = numpy_array
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

        ccenters = torch.stack([c.camera_center.reshape(3) for c in cameras], dim=0)
        minv = ccenters.min(dim=0, keepdim=True).values
        maxv = ccenters.max(dim=0, keepdim=True).values
        # center = (minv + (maxv-minv)/2).to(device)
        # scaling = (maxv-minv).max().to(device)
        center = ccenters.mean(dim=0)
        scaling = torch.linalg.norm(ccenters - center.reshape(1, 3), dim=1, ord=torch.inf).max()
        # ic(center1, center, scaling1, scaling)

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

        vertex_base_color = RGB2SH(torch.as_tensor(point_cloud.colors).float().to(device))
        vertex_base_color = vertex_base_color.reshape(-1, 1, 3).expand(-1, repeats, 3).reshape(-1, 3)

        # add sphere
        # pcd_scaling = torch.linalg.norm(vertices - center.cpu().reshape(1, 3), dim=1, ord=torch.inf).max()
        # x = l2_normalize_th(torch.randn((1000, 3))) * 1.2 * pcd_scaling.cpu() + center.reshape(1, 3).cpu()
        # vertices = torch.cat([vertices, x], dim=0)
        # vertex_base_color = torch.cat([vertex_base_color, torch.zeros_like(x).to(vertex_base_color.device)], dim=0)

        # vertex_base_color = torch.ones_like(vertices, device=device) * 0.1
        # vertex_lights = torch.zeros((vertices.shape[0], num_lights*6)).to(device)
        vertex_lights = torch.zeros((vertices.shape[0], ((num_lights+1)**2-1)*3)).to(device)

        vertices = nn.Parameter(vertices.cuda())
        model = Model(vertices, vertex_base_color, vertex_lights, center, scaling, num_lights=num_lights, **kwargs)
        return model

    def sh_up(self):
        self.max_lights = min(self.num_lights, self.max_lights+1)

    def compute_batch_features(self, vertices, indices, start, end):
        cv, scaling =  pre_calc_cell_values(
            vertices, indices[start:end], self.center, self.scene_scaling,
            self.per_level_scale, self.L, self.scale_multi, self.base_resolution)
        x = (cv/2 + 1)/2
        output = checkpoint(self.encoding, x, use_reentrant=True).float()
        output = output.reshape(-1, self.dim, self.L)

        output = output * scaling
        output = checkpoint(self.network, output.reshape(-1, self.L * self.dim), use_reentrant=True)
        return output

    def update_triangulation(self, alpha_threshold=0.0/255):
        verts = self.vertices
        v = Del(verts.shape[0])
        verts_c = verts.detach().cpu()
        indices_np, prev = v.compute(verts_c)
        indices_np = indices_np.numpy()
        finite_tets = (indices_np < verts.shape[0]).all(axis=1)
        self.boundary_tets = torch.zeros((indices_np.shape[0]), dtype=bool, device='cuda')
        v = prev.get_boundary_tets(verts_c)
        self.boundary_tets[v] = True
        indices_np = indices_np[finite_tets]
        self.boundary_tets = self.boundary_tets[finite_tets]
        self.indices = torch.as_tensor(indices_np).cuda()
 
        if alpha_threshold > 0:
            # Compute the density mask in chunks
            mask_list = []
            start = 0
            
            vertices = self.vertices
            # while start < self.indices.shape[0]:
            for start in range(0, self.indices.shape[0], self.chunk_size):
                end = min(start + self.chunk_size, self.indices.shape[0])
                
                output = self.compute_batch_features(vertices, self.indices, start, end)

                density = safe_exp(output[:, 0]+self.density_offset)
                indices_chunk = self.indices[start:end]
                v0, v1, v2, v3 = verts[indices_chunk[:, 0]], verts[indices_chunk[:, 1]], verts[indices_chunk[:, 2]], verts[indices_chunk[:, 3]]
                
                edge_lengths = torch.stack([
                    torch.norm(v0 - v1, dim=1), torch.norm(v0 - v2, dim=1), torch.norm(v0 - v3, dim=1),
                    torch.norm(v1 - v2, dim=1), torch.norm(v1 - v3, dim=1), torch.norm(v2 - v3, dim=1)
                ], dim=0).max(dim=0)[0]
                
                # Compute the maximum possible alpha using the largest edge length
                alpha = 1 - torch.exp(-density * edge_lengths)
                # mask_list.append(density > density_threshold)
                mask_list.append(alpha > alpha_threshold)
                
                # start = end
            
            # Concatenate mask and apply it
            mask = torch.cat(mask_list, dim=0) & ~self.boundary_tets
            self.indices = self.indices[mask]
            self.boundary_tets = self.boundary_tets[mask]
            
            del indices_np, prev, mask_list, mask
        else:
            del indices_np, prev
        
    def get_cell_values(self, camera: Camera, mask=None,
                        circumcenters=None, radii=None):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices
        boundary_tets = self.boundary_tets[mask] if mask is not None else self.boundary_tets

        densities = torch.empty((indices.shape[0]), device=self.device)
        # densities = []
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])
            output = self.compute_batch_features(vertices, indices, start, end)
            density = safe_exp(output[:, 0]+self.density_offset)
            densities[start:end] = density
            # densities.append(density)
        # densities = torch.cat(densities, dim=0)
        densities = torch.where(boundary_tets, MAX_DENSITY, densities)

        vertex_color_raw = eval_sh(
            vertices,
            self.vertex_base_color,
            self.vertex_lights.reshape(-1, (self.num_lights+1)**2 - 1, 3),
            camera.camera_center,
            self.max_lights)
        vertex_color = torch.nn.functional.softplus(vertex_color_raw, beta=10)
        # vertex_color = compute_vert_color(
        #     self.vertex_base_color, self.vertex_lights.reshape(-1, self.num_lights, 6)[:, :self.max_lights],
        #     vertices, camera.camera_center, self.light_offset, self.dir_offset[:self.max_lights])
        return vertex_color, densities

    def __len__(self):
        return self.vertices.shape[0]
        

class TetOptimizer:
    def __init__(self,
                 model: Model,
                 encoding_lr: float=1e-2,
                 final_encoding_lr: float=1e-2,
                 color_lr: float=1e-2,
                 final_color_lr: float=1e-2,  # <-- Add final LR for base color
                 network_lr: float=1e-3,
                 final_network_lr: float=1e-3,
                 lights_lr: float=1e-4,
                 final_lights_lr: float=1e-4,  # <-- Add final LR for lights
                 vertices_lr: float=4e-4,
                 final_vertices_lr: float=4e-7,
                 vertices_lr_delay_multi: float=0.01,
                 max_steps: int=5000,
                 weight_decay=1e-10,
                 net_weight_decay=1e-3,
                 split_std: float = 0.5,
                 vertices_beta: List[float] = [0.9, 0.99],
                 lr_delay: int = 500,
                 sh_lr_delay: int = 1000,
                 **kwargs):
        self.weight_decay = weight_decay
        self.optim = optim.CustomAdam([
            {"params": model.encoding.parameters(), "lr": encoding_lr, "name": "encoding"},
        ], ignore_param_list=["encoding", "network"], betas=[0.9, 0.99], eps=1e-15)
        self.net_optim = optim.CustomAdam([
            {"params": model.network.parameters(), "lr": network_lr, "name": "network"},
        ], ignore_param_list=["encoding", "network"], betas=[0.9, 0.99])
        self.base_color_optim = optim.CustomAdam([
            {"params": model.vertex_base_color, "lr": color_lr, "name": "base_color"},
        ])
        self.lights_optim = optim.CustomAdam([
            {"params": [model.vertex_lights], "lr": lights_lr, "name": "vertex_lights"},
        ])
        self.vert_lr_multi = 1 if model.contract_vertices else float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.contracted_vertices], "lr": self.vert_lr_multi*vertices_lr, "name": "contracted_vertices"},
        ])
        self.model = model
        self.tracker_n = 0
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None
        self.split_std = split_std

        self.net_scheduler_args = get_expon_lr_func(lr_init=network_lr,
                                                lr_final=final_network_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=500,
                                                max_steps=max_steps)
        self.encoder_scheduler_args = get_expon_lr_func(lr_init=encoding_lr,
                                                lr_final=final_encoding_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=500,
                                                max_steps=max_steps)
        self.vertex_scheduler_args = get_expon_lr_func(lr_init=self.vert_lr_multi*vertices_lr,
                                                lr_final=self.vert_lr_multi*final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_multi,
                                                max_steps=max_steps,
                                                lr_delay_steps=lr_delay)
        # self.color_scheduler_args = get_expon_lr_func(
        #     lr_init=color_lr,
        #     lr_final=final_color_lr,
        #     # lr_delay_mult=1e-8,
        #     lr_delay_steps=0,
        #     max_steps=max_steps
        # )
        self.lights_scheduler_args = get_expon_lr_func(
            lr_init=lights_lr,
            lr_final=final_lights_lr,
            # lr_delay_mult=1e-8,
            lr_delay_steps=sh_lr_delay,
            max_steps=max_steps
        )

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
                param_group['lr'] = lr
        # # Base color
        # for param_group in self.base_color_optim.param_groups:
        #     if param_group["name"] == "base_color":
        #         lr = self.color_scheduler_args(iteration)
        #         param_group['lr'] = lr

        # # Lights
        # for param_group in self.lights_optim.param_groups:
        #     if param_group["name"] == "vertex_lights":
        #         lr = self.lights_scheduler_args(iteration)
        #         param_group['lr'] = lr

    def remove_points(self, mask: torch.Tensor):
        new_tensors = self.optim.prune_optimizer(mask)
        self.model.contracted_vertices = self.vertex_optim.prune_optimizer(mask)['contracted_vertices']
        self.model.update_triangulation()

    def add_points(self, new_verts: torch.Tensor, new_vertex_base_color: torch.Tensor, new_vertex_lights: torch.Tensor):
        if self.model.contract_vertices:
            new_verts = self.model.contract(new_verts)
        self.model.contracted_vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            contracted_vertices = new_verts
        ))['contracted_vertices']
        self.model.vertex_lights = self.lights_optim.cat_tensors_to_optimizer(dict(
            vertex_lights = new_vertex_lights
        ))['vertex_lights']
        self.model.vertex_base_color = self.base_color_optim.cat_tensors_to_optimizer(dict(
            base_color = new_vertex_base_color
        ))['base_color']
        self.model.update_triangulation()

    @torch.no_grad()
    def split(self, clone_indices, split_mode):
        device = self.model.device

        if split_mode == "barycentric":
            barycentric = torch.rand((clone_indices.shape[0], clone_indices.shape[1], 1), device=device).clip(min=0.01, max=0.99)
            barycentric_weights = barycentric / (1e-3+barycentric.sum(dim=1, keepdim=True))
            new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights).sum(dim=1)
            new_vertex_base_color = (self.model.vertex_base_color[clone_indices] * barycentric_weights).sum(dim=1)
            new_vertex_lights = (self.model.vertex_lights[clone_indices] * barycentric_weights).sum(dim=1)
        else:
            raise Exception(f"Split mode: {split_mode} not supported")
        self.add_points(new_vertex_location, new_vertex_base_color, new_vertex_lights)

    def main_step(self):
        self.optim.step()
        self.net_optim.step()
        self.base_color_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad()
        self.net_optim.zero_grad()
        self.base_color_optim.zero_grad()

    @property
    def sh_optim(self):
        return self.lights_optim

    def regularizer(self):
        return self.weight_decay * sum([(embed.weight**2).mean() for embed in self.model.encoding.embeddings])
