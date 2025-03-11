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
from utils.topo_utils import calculate_circumcenters_torch
from utils.safe_math import safe_exp, safe_div, safe_sqrt, safe_pow, safe_cos, safe_sin, remove_zero
from utils.contraction import contract_mean_std
from torch_ema import ExponentialMovingAverage
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


def forward_in_chunks(forward, x, chunk_size=548576):
    """
    Same as forward(), but processes 'x' in chunks to reduce memory usage.
    """
    outputs = []
    start = 0
    while start < x.shape[0]:
        end = min(start + chunk_size, x.shape[0])
        x_chunk = x[start:end]
        outputs.append(forward(x_chunk))
        start = end
    return torch.cat(outputs, dim=0)

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

@torch.jit.script
def to_sphere(coordinates):
    return torch.stack([
        safe_cos(coordinates[..., 0]) * safe_sin(coordinates[..., 1]),
        safe_sin(coordinates[..., 0]) * safe_sin(coordinates[..., 1]),
        safe_cos(coordinates[..., 1]),
    ], dim=-1)
                        
    

@torch.jit.script
def light_function(base_color, reflection_dirs, light_colors, light_roughness, view_dirs, eps:float=torch.finfo(torch.float32).eps):
    similarity = (reflection_dirs * view_dirs).sum(dim=-1, keepdim=True)
    mask = similarity > 0
    spec_intensity = torch.where(mask, (similarity.clip(min=eps) ** light_roughness), 0)
    spec_color = (light_colors * spec_intensity).sum(dim=1)
    return base_color + spec_color

@torch.jit.script
def activate_lights(base_color_raw, lights, light_offset: float, dir_offset):
    base_color = torch.nn.functional.softplus(base_color_raw)
    light_colors = torch.nn.functional.softplus(lights[:, :, :3]+light_offset)
    light_roughness = 4*safe_exp(lights[:, :, 3:4]-1).clip(max=100)
    reflection_dirs = lights[:, :, 4:6] + dir_offset.reshape(1, -1, 2)
    return base_color, light_colors, light_roughness, reflection_dirs

@torch.jit.script
def compute_light_color(base_color_raw, lights, vertices, indices, camera_center, light_offset: float, dir_offset):
    base_color, light_colors, light_roughness, reflection_dirs = activate_lights(
        base_color_raw, lights, light_offset, dir_offset)
    # base_color = torch.nn.functional.softplus(base_color_raw)
    # light_colors = torch.nn.functional.softplus(lights[:, :, :3]+light_offset)
    # light_roughness = 4*safe_exp(lights[:, :, 3:4]-4).clip(max=100)
    reflection_dirs = to_sphere(reflection_dirs)
    barycenters = vertices[indices].mean(dim=1)
    view_dirs = l2_normalize_th(camera_center - barycenters).reshape(-1, 1, 3)
    return light_function(base_color, reflection_dirs, light_colors, light_roughness, view_dirs)

class Model(nn.Module):
    def __init__(self,
                 vertices: torch.Tensor,
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
                 light_offset=-3,
                 **kwargs):
        super().__init__()
        self.scale_multi = scale_multi
        self.L = L
        self.dim = hashmap_dim
        self.device = vertices.device
        self.density_offset = density_offset
        self.num_lights = num_lights
        self.max_lights = 0
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
            nn.Linear(hidden_dim, 4 + self.num_lights * 6)
        )).to(self.device)
        gain = nn.init.calculate_gain('relu')  # for example, if using ReLU activations
        self.network.apply(lambda m: init_weights(m, gain))

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
        model = Model(vertices.to(device), ckpt['center'], ckpt['scene_scaling'], **config.as_dict())
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
        xyz = self.vertices.detach().cpu().numpy().astype(np.float32)  # shape (num_vertices, 3)

        # For tinyply, we store them as one dictionary for the "vertex" element:
        #   { "x": array([...]), "y": ..., "z": ... }
        # Make sure to cast to a concrete dtype (e.g. float32).
        vertex_dict = {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
        }

        # 2. Compute your RGBA / lighting data per tetrahedron
        #    (same logic as in your code: iterative chunking, gather, etc.)
        N = self.indices.shape[0]
        rgbs = np.zeros((N, 4 + self.num_lights * 6), dtype=np.float32)

        vertices = self.vertices
        indices = self.indices
        # e.g., chunk-based processing (adapt as you did in your original code)
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])

            output = self.compute_batch_features(vertices, indices, start, end)
            base_color_raw = output[:, :3]
            density = torch.exp(output[:, 3:4] + self.density_offset)
            lights = output[:, 4:].reshape(-1, self.num_lights, 6)
            base_color, light_colors, light_roughness, reflection_dirs = activate_lights(
                base_color_raw, lights, self.light_offset, self.dir_offset
            )
            rgbs[start:end] = torch.cat([
                base_color,
                density,
                light_colors.reshape(-1, self.num_lights * 3),
                light_roughness.reshape(-1, self.num_lights),
                reflection_dirs.reshape(-1, self.num_lights * 2),
            ], dim=1).cpu().numpy().astype(np.float32)

        # 3. Build the dictionary for your "tetrahedron" element
        #    'vertex_indices' is a 2D array (N,4) for the tetra indices
        #    plus 'r', 'g', 'b', 's', and the per-light properties
        tetra_dict = {}

        # Indices: shape (N, 4). Must be stored as an unsigned int (common for face/tet indices).
        tetra_dict["vertex_indices"] = self.indices.cpu().numpy().astype(np.int32)

        # The first 4 columns in rgbs are [r, g, b, s].
        tetra_dict["r"] = np.ascontiguousarray(rgbs[:, 0])
        tetra_dict["g"] = np.ascontiguousarray(rgbs[:, 1])
        tetra_dict["b"] = np.ascontiguousarray(rgbs[:, 2])
        tetra_dict["s"] = np.ascontiguousarray(rgbs[:, 3])

        # Then the remainder are the per-light columns:
        # Each light has 6 columns: [l_r, l_g, l_b, l_roughness, l_phi, l_theta].
        # They occupy columns: rgbs[:, 4 + i*6 : 4 + (i+1)*6].
        for i in range(self.num_lights):
            offset = 4 + i*6
            tetra_dict[f"l{i}_r"]         = np.ascontiguousarray(rgbs[:, offset + 0])
            tetra_dict[f"l{i}_g"]         = np.ascontiguousarray(rgbs[:, offset + 1])
            tetra_dict[f"l{i}_b"]         = np.ascontiguousarray(rgbs[:, offset + 2])
            tetra_dict[f"l{i}_roughness"] = np.ascontiguousarray(rgbs[:, offset + 3])
            tetra_dict[f"l{i}_phi"]       = np.ascontiguousarray(rgbs[:, offset + 4])
            tetra_dict[f"l{i}_theta"]     = np.ascontiguousarray(rgbs[:, offset + 5])

        # 4. Final data structure:
        # data_dict[element_name][property_name] = numpy_array
        data_dict = {
            "vertex": vertex_dict,
            "tetrahedron": tetra_dict,
        }

        tinyplypy.write_ply(str(path), data_dict, is_binary=True)
        # data_dict = tinyplypy.read_ply(str(path))
        # ic(data_dict['vertex']['x'], xyz[:, 0],
        #    data_dict['tetrahedron']['r'], rgbs[:, 0], tetra_dict['r'],
        #    data_dict['tetrahedron']['g'], rgbs[:, 1],
        #    data_dict['tetrahedron']['b'], rgbs[:, 2],
        #    rgbs.shape, data_dict['tetrahedron']['r'].shape,
        #    vertex_dict['x'].shape, data_dict['vertex']['x'].shape)
        # ic(self.indices, data_dict['tetrahedron']['vertex_indices'])

    # @torch.no_grad
    # def save2ply(self, path: Path, sample_camera: Camera):
    #     path.parent.mkdir(exist_ok=True, parents=True)
        
    #     xyz = self.vertices.detach().cpu().numpy()

    #     dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        
    #     elements = np.empty(xyz.shape[0], dtype=dtype_full)
    #     elements['x'] = xyz[:, 0]
    #     elements['y'] = xyz[:, 1]
    #     elements['z'] = xyz[:, 2]
    #     el = PlyElement.describe(elements, 'vertex')

    #     dtype_tets = [
    #         ('vertex_indices', 'i4', (4,)),
    #         # ('vertex_indices', 'O'),
    #         ('r', 'f4'),
    #         ('g', 'f4'),
    #         ('b', 'f4'),
    #         ('s', 'f4'),
    #     ]
    #     for i in range(self.num_lights):
    #         dtype_tets.extend([
    #             (f'l{i}_r', 'f4'),
    #             (f'l{i}_g', 'f4'),
    #             (f'l{i}_b', 'f4'),
    #             (f'l{i}_roughness', 'f4'),
    #             (f'l{i}_phi', 'f4'),
    #             (f'l{i}_theta', 'f4')])
    #     dtype_tets = np.dtype(dtype_tets)

    #     # Get the color/scalar values (each element should be a tuple: (r, g, b, s))
    #     # need to batch this saving process
    #     N = self.indices.shape[0]
    #     B = 1_000_000
    #     rgbs = np.zeros((N, 4 + self.num_lights * 6))
    #     # rgbs = self.get_cell_values(sample_camera, mask).detach().cpu().numpy()
    #     start = 0
        
    #     vertices = self.vertices
    #     indices = self.indices
    #     for start in range(0, indices.shape[0], self.chunk_size):
    #         end = min(start + self.chunk_size, indices.shape[0])
            
    #         output = self.compute_batch_features(vertices, indices, start, end)
    #         base_color_raw = output[:, :3]
    #         density = safe_exp(output[:, 3:4]+self.density_offset)
    #         lights = output[:, 4:].reshape(-1, self.num_lights, 6)
    #         base_color, light_colors, light_roughness, reflection_dirs = activate_lights(
    #             base_color_raw, lights, self.light_offset, self.dir_offset)
    #         rgbs[start:end] = torch.cat([
    #             base_color,
    #             density,
    #             light_colors.reshape(-1, self.num_lights * 3),
    #             light_roughness.reshape(-1, self.num_lights),
    #             reflection_dirs.reshape(-1, self.num_lights * 2),
    #         ], dim=1).cpu().numpy()


    #     tet_elements = np.empty(self.indices.shape[0], dtype=dtype_tets)
    #     # tet_elements['vertex_indices'] = list(self.indices_np)
    #     tet_elements['vertex_indices'] = self.indices.cpu().numpy()
    #     tet_elements['r'] = rgbs[:, 0]
    #     tet_elements['g'] = rgbs[:, 1]
    #     tet_elements['b'] = rgbs[:, 2]
    #     tet_elements['s'] = rgbs[:, 3]
    #     for i in range(self.num_lights):
    #         tet_elements[f'l{i}_r'] = rgbs[:, 3 + i*6 + 0]
    #         tet_elements[f'l{i}_g'] = rgbs[:, 3 + i*6 + 1]
    #         tet_elements[f'l{i}_b'] = rgbs[:, 3 + i*6 + 2]
    #         tet_elements[f'l{i}_roughness'] = rgbs[:, 3 + i*6 + 3]
    #         tet_elements[f'l{i}_phi'] = rgbs[:, 3 + i*6 + 4]
    #         tet_elements[f'l{i}_theta'] = rgbs[:, 3 + i*6 + 5]

    #     # Create the PlyElement description for tetrahedra
    #     inds = PlyElement.describe(tet_elements, 'tetrahedron')

    #     PlyData([el, inds]).write(str(path))

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
    def init_from_pcd(point_cloud, cameras, device, **kwargs):
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

        vertices = vertices + torch.randn(*vertices.shape) * 1e-3
        v = Del(vertices.shape[0])
        indices_np, prev = v.compute(vertices.detach().cpu())
        indices_np = indices_np.numpy()
        indices_np = indices_np[(indices_np < vertices.shape[0]).all(axis=1)]
        vertices = vertices[indices_np].mean(dim=1)
        vertices = vertices + torch.randn(*vertices.shape) * 1e-3

        # repeats = 3
        # vertices = vertices.reshape(-1, 1, 3).expand(-1, repeats, 3)
        # vertices = vertices + torch.randn(*vertices.shape) * 1e-1
        # vertices = vertices.reshape(-1, 3)

        vertices = nn.Parameter(vertices.cuda())
        model = Model(vertices, center, scaling, **kwargs)
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

    def update_triangulation(self, alpha_threshold=1.0/255):
        verts = self.vertices
        v = Del(verts.shape[0])
        indices_np, prev = v.compute(verts.detach().cpu())
        indices_np = indices_np.numpy()
        indices_np = indices_np[(indices_np < verts.shape[0]).all(axis=1)]
        
        # Convert to tensor and move to CUDA
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

                density = safe_exp(output[:, 3]+self.density_offset)
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
            mask = torch.cat(mask_list, dim=0)
            self.indices = self.indices[mask]
            
            del indices_np, prev, mask_list, mask
        else:
            del indices_np, prev
        
    def get_cell_values(self, camera: Camera, mask=None,
                        circumcenters=None, radii=None):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices

        outputs = []
        start = 0
        # with autocast(dtype=torch.float16):
        for start in range(0, indices.shape[0], self.chunk_size):
        # while start < indices.shape[0]:
            end = min(start + self.chunk_size, indices.shape[0])
            output = self.compute_batch_features(vertices, indices, start, end)
            # base_color = torch.nn.functional.softplus(output[:, :3])
            base_color_raw = output[:, :3]
            density = safe_exp(output[:, 3:4]+self.density_offset)
            lights = output[:, 4:].reshape(-1, self.num_lights, 6)[:, :self.max_lights]
            color = compute_light_color(
                base_color_raw, lights, vertices, indices[start:end],
                camera.camera_center, self.light_offset, self.dir_offset[:self.max_lights])
            # features = torch.cat([
            #     torch.nn.functional.softplus(output[:, :3]), safe_exp(output[:, 3:4]+self.density_offset)], dim=1)
            features = torch.cat([
                color, density], dim=1)
            outputs.append(features)
            # start = end
        features = torch.cat(outputs, dim=0)
        return features

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
                 vertices_lr_max_steps: int=5000,
                 weight_decay=1e-10,
                 net_weight_decay=1e-3,
                 split_std: float = 0.5,
                 vertices_beta: List[float] = [0.9, 0.99],
                 lr_delay: int = 500,
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
        self.ema = ExponentialMovingAverage(list(model.network.parameters()) + list(model.encoding.parameters()), decay=0.99)
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
        self.vertex_scheduler_args = get_expon_lr_func(lr_init=self.vert_lr_multi*vertices_lr,
                                                lr_final=self.vert_lr_multi*final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_multi,
                                                max_steps=10000,
                                                lr_delay_steps=lr_delay)

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

    def update_ema(self):
        self.ema.update()

    def add_points(self, new_verts: torch.Tensor):
        if self.model.contract_vertices:
            new_verts = self.model.contract(new_verts)
        self.model.contracted_vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            contracted_vertices = new_verts
        ))['contracted_vertices']
        self.model.update_triangulation()

    def remove_points(self, mask: torch.Tensor):
        new_tensors = self.optim.prune_optimizer(mask)
        self.model.contracted_vertices = self.vertex_optim.prune_optimizer(mask)['contracted_vertices']
        self.model.update_triangulation()

    @torch.no_grad()
    def split(self, clone_indices, split_mode):
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
        elif split_mode == "barycentric":
            barycentric = torch.rand((clone_indices.shape[0], clone_indices.shape[1], 1), device=device).clip(min=0.01, max=0.99)
            barycentric_weights = barycentric / (1e-3+barycentric.sum(dim=1, keepdim=True))
            new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights).sum(dim=1)
        else:
            raise Exception(f"Split mode: {split_mode} not supported")
        self.add_points(new_vertex_location)

    def main_step(self):
        self.optim.step()
        self.net_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad()
        self.net_optim.zero_grad()

    @property
    def sh_optim(self):
        optim = lambda x: x
        optim.step = lambda x=1: x
        optim.zero_grad = lambda x=1: x
        return optim

    def regularizer(self):
        return self.weight_decay * sum([(embed.weight**2).mean() for embed in self.model.encoding.embeddings])
        # l2_reg = 1e-6 * torch.linalg.norm(list(self.model.encoding.parameters())[0], ord=2)
        # return l2_reg
        # split params
        param = list(self.model.encoding.parameters())[0]
        weight_decay = 0
        ind = 0
        for i in range(self.model.different_size):
            o = self.model.offsets[i+1] - self.model.offsets[i]
            weight_decay = weight_decay + (param[ind:self.model.offsets[i+1]]**2).mean()
            ind += o
        weight_decay = weight_decay + (param[ind:].reshape(-1, self.model.nominal_offset_size)**2).mean(dim=1).sum()
        
        return self.weight_decay * weight_decay# + l2_reg
