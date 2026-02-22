import torch
import math
from data.camera import Camera
from utils import optim
from torch import nn

from utils.topo_utils import calculate_circumcenters_torch, fibonacci_spiral_on_sphere, get_tet_adjacency, tet_volumes
from utils import topo_utils

from utils.train_util import get_expon_lr_func, SpikingLR
from pathlib import Path
import numpy as np
from utils.args import Args
import open3d as o3d
from models.base_model import BaseModel
from utils.eval_sh_py import eval_sh
from icecream import ic

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
        self.max_sh_deg = max_sh_deg
        self.current_sh_deg = current_sh_deg
        self.dir_offset = torch.tensor([
            [0, 0],
            [math.pi, 0],
            [math.pi/3, math.pi/3],
        ], device=self.device)
        self.sh_dim = ((1+max_sh_deg)**2)*3
        self.additional_attr = additional_attr
        self.density_offset = density_offset

        self.chunk_size = 308576
        self.mask_values = True
        self.frozen = False
        self.linear = False
        self.feature_dim = 4 + additional_attr
        self.alpha = 0
        self.ablate_circumsphere = ablate_circumsphere
        self.ablate_gradient = ablate_gradient

        self.register_buffer('ext_vertices', ext_vertices.to(self.device))
        self.register_buffer('center', center.reshape(1, 3))
        self.register_buffer('scene_scaling', torch.tensor(float(scene_scaling), device=self.device))
        self.interior_vertices = nn.Parameter(vertices.detach())
        self.update_triangulation()

    @property
    def num_int_verts(self):
        return self.interior_vertices.shape[0]

    def get_face_values(self, camera: Camera, mask=None):
        """
        Pre-computes the activated Density and RGB values per face 
        for the specific camera view.
        Returns: [M, 4 + Aux] tensor ready for the rasterizer.
        """
        M = self.face_values.shape[0]
        
        # 1. Split Parameters
        raw_density = self.face_values[:, 0:1]
        raw_sh = self.face_values[:, 1 : 1 + self.sh_dim]
        raw_aux = self.face_values[:, 1 + self.sh_dim :]

        
        # We evaluate SH at the Face Center.
        f_v = self.vertices[self.unique_faces] # [M, 3, 3]
        f_centers = f_v.mean(dim=1) # [M, 3]
        sh_reshaped = raw_sh.reshape(M, -1, 3) # [M, (deg+1)^2, 3]
        sh0 = sh_reshaped[:, 0, :]
        sh_rest = sh_reshaped[:, 1:, :] # [M, Num_Rest, 3]
        rgb_val = eval_sh(
            f_centers, sh0, sh_rest, camera.camera_center.cuda(), self.current_sh_deg) - 0.5
        # ic(raw_density.mean(), rgb_val.mean(), self.face_values.grad)
        # return torch.cat([raw_density, rgb_val, raw_aux], dim=1)
        # return self.face_areas.clip(min=0, max=1).detach() * self.hodge_ratio.unsqueeze(1).detach().clip(min=0, max=1) * torch.cat([raw_density, rgb_val, raw_aux], dim=1)
        # return self.face_areas.clip(min=0, max=1).detach() * self.hodge_ratio.unsqueeze(1).detach().clip(min=0, max=1) * torch.cat([raw_density, rgb_val, raw_aux], dim=1)
        # return self.hodge_ratio.unsqueeze(1).detach().clip(min=0, max=1) * torch.cat([raw_density, rgb_val, raw_aux], dim=1)
        return self.voronoi_gate.unsqueeze(1).detach() * torch.cat([raw_density, rgb_val, raw_aux], dim=1)
        # return self.face_areas * torch.cat([raw_density, rgb_val, raw_aux], dim=1)

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

        vertices = topo_utils.sample_uniform_in_sphere(num_points, 3, base_radius=0, radius=new_radius, device='cpu') + center.reshape(1, 3).cpu()

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

        # num_ext = 1000
        # ext_vertices = fibonacci_spiral_on_sphere(num_ext, new_radius, device='cpu') + center.reshape(1, 3).cpu()
        # num_ext = ext_vertices.shape[0]

        num_ext = 1000
        ext_vertices = topo_utils.expand_convex_hull(vertices, 1, device=vertices.device)
        if ext_vertices.shape[0] > num_ext:
            inds = np.random.default_rng().permutation(ext_vertices.shape[0])[:num_ext]
            ext_vertices = ext_vertices[inds]
        else:
            num_ext = ext_vertices.shape[0]

        vertices = torch.cat([vertices, ext_vertices], dim=0)
        ext_vertices = torch.empty((0, 3))

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
        volumes = tet_volumes(self.vertices[self.indices])
        
        # 1. Get parameters (0-form)
        raw_dens = self.face_values[:, 0:1]
        
        # 2. Activate (Intensity)
        act_dens = (raw_dens + self.density_offset).exp()
        
        # 3. Apply Hodge Star (Intensity -> Flux)
        # CRITICAL: Must scale by area before divergence!
        dens_flux = act_dens * self.face_areas 
        
        # 4. Gather for Tets
        f_ids = self.tet_face_ids.long()
        # Handle sign (packed ids)
        # Note: Your model.tet_face_ids has signs encoded. 
        # We need to unpack or use a helper that handles it.
        # Assuming you have the 'signs' tensor available or decode it:
        
        # Quick decode for python culling:
        raw_ids = self.tet_face_ids.long()
        indices = torch.abs(raw_ids)
        # If raw < 0, sign is -1. Else 1. (using bitwise NOT logic from before)
        # Actually, let's just match the Shader logic:
        # id < 0 ? index = ~id, sign = -1 : index = id, sign = 1
        
        mask_neg = (raw_ids < 0)
        indices = raw_ids.clone()
        indices[mask_neg] = ~indices[mask_neg]
        
        signs = torch.ones_like(raw_ids, dtype=torch.float32)
        signs[mask_neg] = -1.0
        
        # Gather
        fluxs = dens_flux[indices, 0] # [N, 4]
        
        # 5. Divergence
        div = (fluxs * signs).sum(dim=1)
        
        # 6. Density
        density = div / (3.0 * volumes + 1e-8)
        return density

    @property
    def vertices(self):
        verts = self.interior_vertices
        return torch.cat([verts, self.ext_vertices])

    def sh_up(self):
        self.current_sh_deg = min(self.max_sh_deg, self.current_sh_deg+1)

    def compute_adjacency(self):
        self.faces, self.side_index = get_tet_adjacency(self.indices)

    @torch.no_grad()
    def update_triangulation(self, high_precision=False, density_threshold=0.0, **kwargs):
        torch.cuda.empty_cache()
        verts = self.vertices
        device = verts.device

        # Snapshot Old
        old_face_values = self.face_values.data if hasattr(self, 'face_values') else None
        old_unique_faces = self.unique_faces if hasattr(self, 'unique_faces') else None

        # Triangulate
        indices, _ = topo_utils.calc_delaunay(verts.detach().cpu(), device, high_precision)

        # Build Topology
        unique_faces, tet_face_ids = topo_utils.build_face_topology(indices)
        
        # Compute Signs
        signs = topo_utils.compute_face_signs(verts, indices, unique_faces, tet_face_ids)

        # Transfer Data
        new_values = 1e-2*torch.randn((unique_faces.shape[0], 1 + self.sh_dim + self.additional_attr), device=device)
        src_indices, dst_indices = None, None

        if old_face_values is not None:
            # Hash faces (Sort-invariant)
            def pack_keys(f):
                return (f[:,0].long() << 42) | (f[:,1].long() << 21) | f[:,2].long()
            
            old_keys = pack_keys(old_unique_faces)
            new_keys = pack_keys(unique_faces)
            
            sorter = torch.argsort(old_keys)
            old_sorted = old_keys[sorter]
            
            idx_in_old = torch.searchsorted(old_sorted, new_keys).clamp(0, len(old_sorted)-1)
            matched_mask = (old_sorted[idx_in_old] == new_keys)
            
            dst_indices = torch.nonzero(matched_mask).squeeze()
            src_indices = sorter[idx_in_old[matched_mask]]
            
            # Direct Copy (Canonical Faces match, so values match)
            new_values[dst_indices] = old_face_values[src_indices]

        # Culling
        # if density_threshold > 0:
        #     raw_dens = new_values[:, 0:1]
        #     act_dens = (raw_dens + self.density_offset).exp()
        #     densities = topo_utils.calc_density_from_flux(act_dens, signs, vols)
        #     keep_mask = densities > density_threshold
        #     
        #     indices = indices[keep_mask]
        #     tet_face_ids = tet_face_ids[keep_mask]
        #     signs = signs[keep_mask]

        # Finalize
        packed_ids = tet_face_ids.clone()
        neg_mask = (signs < 0)
        packed_ids[neg_mask] = ~packed_ids[neg_mask]

        self.indices = indices
        self.tet_face_ids = packed_ids.int()
        self.unique_faces = unique_faces
        self.face_values = nn.Parameter(new_values)
        self.face_areas = topo_utils.calc_face_areas(verts, unique_faces)
        hodge_ratio, voronoi_lengths, _ = topo_utils.compute_hodge_ratios(
            verts, unique_faces, indices, packed_ids
        )
        self.hodge_ratio = hodge_ratio
        self.voronoi_lengths = voronoi_lengths
        median_voronoi = voronoi_lengths.median().clamp(min=1e-8)
        self.voronoi_gate = voronoi_lengths / (voronoi_lengths + 0.1 * median_voronoi)
        
        return {
            "new_face_values": new_values,
            "src_indices": src_indices,
            "dst_indices": dst_indices
        }

    def __len__(self):
        return self.vertices.shape[0]
        
class TetOptimizer:
    def __init__(self,
                 model: Model,
                 lr: float=1e-3,
                 final_lr: float=1e-4,
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
            {"params": [model.face_values], "lr": lr, "name": "flux"},
        ], betas=[0.9, 0.999], eps=1e-15)

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

        base_net_scheduler = get_expon_lr_func(lr_init=lr,
                                                lr_final=final_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=freeze_start)

        self.scheduler_args = SpikingLR(
            spike_duration, freeze_start, base_net_scheduler,
            midpoint, densify_interval, densify_end,
            lr, lr)

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
        for param_group in self.optim.param_groups:
            lr = self.scheduler_args(iteration)
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

    def main_zero_grad(self):
        self.optim.zero_grad(set_to_none=True)

    @property
    def sh_optim(self):
        return None

    def regularizer(self, render_pkg, **kwargs):
        return 0

    def update_triangulation(self, **kwargs):
        update_info = self.model.update_triangulation(**kwargs)
        self.model.face_values = self.optim.update_topology(
            new_tensor = update_info["new_face_values"],
            name = "flux",
            src_indices = update_info["src_indices"],
            dst_indices = update_info["dst_indices"]
        )
