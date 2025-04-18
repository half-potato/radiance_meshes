import torch
import math
from data.camera import Camera
from utils import optim
from sh_slang.eval_sh import eval_sh
from gDel3D.build.gdel3d import Del
from torch import nn
from icecream import ic
from utils.topo_utils import calculate_circumcenters_torch, fibonacci_spiral_on_sphere, calc_barycentric, sample_uniform_in_sphere
from utils.safe_math import safe_exp, safe_div, safe_sqrt
from utils.contraction import contract_mean_std
from utils.contraction import contract_points, inv_contract_points
from utils.train_util import safe_exp, get_expon_lr_func
from utils import topo_utils
from utils.graphics_utils import l2_normalize_th
from utils import hashgrid
from torch.utils.checkpoint import checkpoint
from pathlib import Path
import numpy as np
from utils.args import Args
import tinyplypy
from scipy.spatial import ConvexHull
from scipy.spatial import  Delaunay

torch.set_float32_matmul_precision('high')

def init_weights(m, gain):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

@torch.jit.script
def pre_calc_cell_values(vertices, indices, center, scene_scaling: float):
    device = vertices.device
    tets = vertices[indices]
    circumcenter, radius = calculate_circumcenters_torch(tets.double())
    # clipped_circumcenter = project_points_to_tetrahedra(circumcenter.float(), tets)
    clipped_circumcenter = circumcenter
    normalized = (clipped_circumcenter - center) / scene_scaling
    cv, cr = contract_mean_std(normalized, radius / scene_scaling)
    # sphere_area = 4/3*math.pi*cr**3
    # scaling = safe_div(base_resolution * per_level_scale**n, sphere_area.reshape(-1, 1, 1)).clip(max=1)
    return clipped_circumcenter, cv.float(), cr, normalized

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
    gradients = base.reshape(-1, 3, 1).clip(min=0) * torch.tanh(0.1*field_samples[:, 3:12].reshape(-1, 3, 3))  # shape (T, 3, 3)
    offsets = triangle_verts - circumcenters[:, None, :]  # shape (T, 4, 3)
    offsets = l2_normalize_th(offsets)
    grad_contrib = torch.einsum('tcd,tvd->tvc', gradients, offsets)
    vertex_colors = base[:, None, :] + grad_contrib
    return vertex_colors, gradients

@torch.jit.script
def activate_output(output, indices, vertex_color_raw, circumcenters, vertices, density_offset:float):
    density = safe_exp(output[:, 0:1]+density_offset)
    field_samples = output[:, 1:]
    vcolors, _ = compute_vertex_colors_from_field(
        vertices[indices].detach(), field_samples.float(), circumcenters.float().detach())
    vcolors = torch.nn.functional.softplus(vertex_color_raw[indices] + vcolors, beta=10)
    vcolors = vcolors.reshape(-1, 12)
    features = torch.cat([density, vcolors], dim=1)
    return features

class iNGPDW(nn.Module):
    def __init__(self, 
                 scale_multi=0.5,
                 log2_hashmap_size=16,
                 base_resolution=16,
                 per_level_scale=2,
                 L=10,
                 hashmap_dim=4,
                 hidden_dim=64,
                 **kwargs):
        super().__init__()

        self.scale_multi = scale_multi
        self.L = L
        self.dim = hashmap_dim
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution

        self.encoding = hashgrid.HashEmbedderOptimized(
            [torch.zeros((3)), torch.ones((3))],
            self.L, n_features_per_level=self.dim,
            log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution,
            finest_resolution=base_resolution*per_level_scale**self.L)

        self.network = nn.Sequential(
            nn.Linear(self.encoding.n_output_dims, hidden_dim),
            nn.SELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(inplace=True),
            nn.Linear(hidden_dim, 1+12)
        )
        gain = nn.init.calculate_gain('relu')  # for example, if using ReLU activations
        self.network.apply(lambda m: init_weights(m, gain))

    def forward(self, x, cr):
        x = x.detach()
        output = self.encoding(x).float()
        output = output.reshape(-1, self.dim, self.L)
        cr = cr.float() * self.scale_multi
        n = torch.arange(self.L, device=x.device).reshape(1, 1, -1)
        erf_x = safe_div(torch.tensor(1.0, device=x.device), safe_sqrt(self.per_level_scale * 4*n*cr.reshape(-1, 1, 1)))
        scaling = torch.erf(erf_x)

        output = output * scaling
        output = self.network(output.reshape(-1, self.L * self.dim))
        return output

class Model(nn.Module):
    def __init__(self,
                 vertices: torch.Tensor,
                 ext_vertices: torch.Tensor,
                 vertex_lights: torch.Tensor,
                 center: torch.Tensor,
                 scene_scaling: float,
                 contract_vertices=True,
                 density_offset=-1,
                 max_lights=2,
                 num_lights=2,
                 light_offset=-3,
                 **kwargs):
        super().__init__()
        self.device = vertices.device
        self.density_offset = density_offset
        self.num_lights = num_lights
        self.max_lights = max_lights
        self.light_offset = light_offset
        self.dir_offset = torch.tensor([
            [0, 0],
            [math.pi, 0],
        ], device=self.device)
        self.backbone = torch.compile(iNGPDW(**kwargs)).to(self.device)
        self.chunk_size = 408576

        self.vertex_lights = nn.Parameter(vertex_lights)

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
        circumcenter, cv, cr, normalized =  pre_calc_cell_values(
            self.vertices, self.indices, self.center, self.scene_scaling)
        return cv

    def get_cell_values(self, camera: Camera, mask=None,
                        all_circumcenters=None, radii=None):
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
            circumcenters, normalized, output = self.compute_batch_features(vertices, indices, start, end, circumcenters=all_circumcenters)
            dvrgbs = activate_output(output, indices[start:end],
                                     vertex_color_raw, circumcenters,
                                     vertices, self.density_offset)
            normed_cc.append(normalized)
            outputs.append(dvrgbs)
        features = torch.cat(outputs, dim=0)
        normed_cc = torch.cat(normed_cc, dim=0)
        return normed_cc, features

    @staticmethod
    def init_from_pcd(point_cloud, cameras, device, num_lights, ext_convex_hull, **kwargs):
        torch.manual_seed(2)
        N = point_cloud.points.shape[0]
        # N = 1000
        vertices = torch.as_tensor(point_cloud.points)[:N]

        ccenters = torch.stack([c.camera_center.reshape(3) for c in cameras], dim=0).to(device)
        center = ccenters.mean(dim=0)
        scaling = torch.linalg.norm(ccenters - center.reshape(1, 3), dim=1, ord=torch.inf).max()

        repeats = 3
        vertices = vertices.reshape(-1, 1, 3).expand(-1, repeats, 3)
        vertices = vertices + torch.randn(*vertices.shape) * 1e-1
        vertices = vertices.reshape(-1, 3)

        # add sphere
        pcd_scaling = torch.linalg.norm(vertices - center.cpu().reshape(1, 3), dim=1, ord=2).max()

        if ext_convex_hull:
            ext_vertices = expand_convex_hull(vertices, 5, device=vertices.device)
            num_ext = ext_vertices.shape[0]
        else:
            new_radius = 2* pcd_scaling.cpu()
            within_sphere = sample_uniform_in_sphere(10000, 3, radius=new_radius.item(), device='cpu') + center.reshape(1, 3).cpu()
            vertices = torch.cat([vertices, within_sphere], dim=0)
            num_ext = 1000
            ext_vertices = fibonacci_spiral_on_sphere(num_ext, new_radius, device='cpu') + center.reshape(1, 3).cpu()
        num_ext = ext_vertices.shape[0]

        # num_ext = 1000
        # ext_vertices = fibonacci_spiral_on_sphere(num_ext, 2* pcd_scaling.cpu(), device='cpu') + center.reshape(1, 3).cpu()
        vertex_lights = torch.zeros((vertices.shape[0] + num_ext, ((num_lights+1)**2-1)*3)).to(device)

        vertices = nn.Parameter(vertices.cuda())
        model = Model(vertices, ext_vertices, vertex_lights, center, scaling, **kwargs)
        return model

    def compute_batch_features(self, vertices, indices, start, end, circumcenters=None):
        if circumcenters is None:
            circumcenter, cv, cr, normalized =  pre_calc_cell_values(
                vertices, indices[start:end], self.center, self.scene_scaling)
        else:
            circumcenter = circumcenters[start:end]
            normalized = (circumcenter - self.center) / self.scene_scaling
            # ic(circumcenter.shape, vertices[indices[start:end, 0]].shape, circumcenters.shape, start, end)
            radius = torch.linalg.norm(circumcenter - vertices[indices[start:end, 0]], dim=-1)
            cv, cr = contract_mean_std(normalized, radius / self.scene_scaling)
        x = (cv/2 + 1)/2
        output = checkpoint(self.backbone, x, cr, use_reentrant=True).float()
        return circumcenter, normalized, output

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
        ext_vertices = ckpt['ext_vertices']
        model = Model(vertices.to(device), ext_vertices, lights, ckpt['center'], ckpt['scene_scaling'], **config.as_dict())
        model.load_state_dict(ckpt)
        model.contract_vertices = temp
        model.min_t = model.scene_scaling * config.base_min_t
        model.indices = torch.as_tensor(indices).cuda()
        model.boundary_tets = torch.zeros((indices.shape[0]), dtype=bool, device='cuda')
        return model

    @torch.no_grad
    def perturb_vertices(self, perturbation):
        self.contracted_vertices.data += perturbation[:self.contracted_vertices.shape[0]]

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

    def calc_vert_density(self):
        verts = self.vertices
        vertex_density = torch.zeros((verts.shape[0],), device=self.device)
        indices = self.indices.long()
        for start in range(0, self.indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.indices.shape[0])
            
            _, _, output = self.compute_batch_features(verts, self.indices, start, end)

            density = safe_exp(output[:, 0]+self.density_offset)
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
            
            _, _, output = self.compute_batch_features(verts, self.indices, start, end)

            density = safe_exp(output[:, 0]+self.density_offset)
            indices_chunk = self.indices[start:end]
            v0, v1, v2, v3 = verts[indices_chunk[:, 0]], verts[indices_chunk[:, 1]], verts[indices_chunk[:, 2]], verts[indices_chunk[:, 3]]
            
            edge_lengths = torch.stack([
                torch.norm(v0 - v1, dim=1), torch.norm(v0 - v2, dim=1), torch.norm(v0 - v3, dim=1),
                torch.norm(v1 - v2, dim=1), torch.norm(v1 - v3, dim=1), torch.norm(v2 - v3, dim=1)
            ], dim=0).min(dim=0)[0]
            
            # Compute the maximum possible alpha using the largest edge length
            alpha = 1 - torch.exp(-density * edge_lengths)
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

            circumcenters, _, output = self.compute_batch_features(vertices, indices, start, end)
            radius = torch.linalg.norm(circumcenters - vertices[indices[start:end, 0]], dim=-1)
            density = safe_exp(output[:, 0:1]+self.density_offset)
            field_samples = output[:, 1:]
            base = (field_samples[:, :3]) + 0.5  # shape (T, 3)
            base = base.reshape(-1, 3, 1).clip(min=0)
            gradients = base * torch.tanh(field_samples[:, 3:12].reshape(-1, 3, 3))  # shape (T, 3, 3)
            clipped_gradients = (base+gradients).clip(min=0, max=1) - base
            # tet_var[start:end] = torch.linalg.norm(
            #     torch.linalg.norm(clipped_gradients, dim=-1), dim=-1, ord=torch.inf)
            # tet_var[start:end] = torch.linalg.norm(clipped_gradients, dim=-1).min(dim=-1).values
            tet_var[start:end] = torch.linalg.norm(clipped_gradients, dim=-1).mean(dim=-1)
        return tet_var

    @torch.no_grad
    def extract_mesh(self, path, alpha_threshold=0.5):
        verts = self.vertices
        v = Del(verts.shape[0])
        indices_np, tri_output = v.compute(verts.detach().cpu())
        indices_np = indices_np.numpy()
        inf_mask = (indices_np < verts.shape[0]).all(axis=1)
        indices_np = indices_np[inf_mask]
        
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
        lights = np.zeros((N, 4, 3), dtype=np.float32)

        vertices = self.vertices
        indices = self.indices
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])

            circumcenters, _, output = self.compute_batch_features(vertices, indices, start, end)
            density = safe_exp(output[:, 0:1]+self.density_offset)
            field_samples = output[:, 1:]
            vcolors, _ = compute_vertex_colors_from_field(
                vertices[indices[start:end]].detach(), field_samples.float(), circumcenters.float().detach())
            vcolors = vcolors.cpu().numpy().astype(np.float32)
            density = density.cpu().numpy().astype(np.float32)
            lights[start:end] = vcolors
            densities[start:end] = density.reshape(-1)

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
            verts = self.inv_contract(self.contracted_vertices)
        else:
            verts = self.contracted_vertices
        return torch.cat([verts, self.ext_vertices])

    def sh_up(self):
        self.max_lights = min(self.num_lights, self.max_lights+1)

    @torch.no_grad()
    def update_triangulation(self, high_precision=False, alpha_threshold=0.00/255):
        torch.cuda.empty_cache()
        verts = self.vertices
        if high_precision:
            simplices = Delaunay(verts.detach().cpu().numpy()).simplices
            self.indices = torch.tensor(simplices, device=verts.device).int().cuda()

        else:
            v = Del(verts.shape[0])
            indices_np, prev = v.compute(verts.detach().cpu().double())
            indices_np = indices_np.numpy()
            indices_np = indices_np[(indices_np < verts.shape[0]).all(axis=1)]
        
        self.indices = torch.as_tensor(indices_np).cuda()
        
        if alpha_threshold > 0:
            mask = self.calc_tet_alpha() > alpha_threshold
            self.indices = self.indices[mask]
            
            del prev, mask
        else:
            del prev
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
                 start_clip_multi: float=1e-4,
                 end_clip_multi: float=1e-4,
                 weight_decay=1e-10,
                 lambda_color=1e-10,
                 split_std: float = 0.5,
                 lights_lr: float=1e-4,
                 lr_delay: int = 500,
                 max_steps: int = 10000,
                 vert_lr_delay: int = 500,
                 **kwargs):
        self.weight_decay = weight_decay
        self.lambda_color = lambda_color
        self.optim = optim.CustomAdam([
            {"params": model.backbone.encoding.parameters(), "lr": encoding_lr, "name": "encoding"},
        ], ignore_param_list=["encoding", "network"], betas=[0.9, 0.99], eps=1e-15)
        self.net_optim = optim.CustomAdam([
            {"params": model.backbone.network.parameters(), "lr": network_lr, "name": "network"},
        ], ignore_param_list=["encoding", "network"], betas=[0.9, 0.99])
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
                                                max_steps=max_steps)
        self.encoder_scheduler_args = get_expon_lr_func(lr_init=encoding_lr,
                                                lr_final=final_encoding_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=max_steps)
        self.vertex_lr = self.vert_lr_multi*vertices_lr
        self.vertex_scheduler_args = get_expon_lr_func(lr_init=self.vertex_lr,
                                                lr_final=self.vert_lr_multi*final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_multi,
                                                max_steps=max_steps,
                                                lr_delay_steps=vert_lr_delay)
        self.iteration = 0
        self.clip_multi_scheduler_args = get_expon_lr_func(lr_init=start_clip_multi,
                                                lr_final=end_clip_multi,
                                                max_steps=max_steps)

    @property
    def clip_multi(self):
        return self.clip_multi_scheduler_args(self.iteration)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self.iteration = iteration
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
        mask = mask[:self.model.contracted_vertices.shape[0]]
        self.model.contracted_vertices = self.vertex_optim.prune_optimizer(mask)['contracted_vertices']
        self.model.update_triangulation()

    @torch.no_grad()
    def split(self, clone_indices, split_point, split_mode, density_threshold):
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
        elif split_mode == "split_point":
            split_point += 1e-3*torch.randn(*split_point.shape, device=self.model.device)
            new_vertex_location = split_point
            barycentric_weights = calc_barycentric(split_point, clone_vertices)
            barycentric_weights = barycentric_weights / (1e-3+barycentric_weights.sum(dim=1, keepdim=True))
            # new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights.unsqueeze(-1)).sum(dim=1)
            new_vertex_lights = (self.model.vertex_lights[clone_indices] * barycentric_weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise Exception(f"Split mode: {split_mode} not supported")
        self.add_points(new_vertex_location, new_vertex_lights)
        mask = self.model.calc_vert_density() < density_threshold
        print(f"Pruned: {mask.sum()} points")
        self.remove_points(~mask)
        del mask

    def main_step(self):
        self.optim.step()
        self.net_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad()
        self.net_optim.zero_grad()

    @property
    def sh_optim(self):
        return self.lights_optim

    def clip_gradient(self, grad_clip):
        for module in self.model.backbone.encoding.embeddings:
            torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip, error_if_nonfinite=True)

    def regularizer(self):
        weight_decay = self.weight_decay * sum([(embed.weight**2).mean() for embed in self.model.backbone.encoding.embeddings])
        mean_sh = (self.model.vertex_lights).abs().mean()
        sh_decay = self.lambda_color * mean_sh
        return sh_decay + weight_decay
