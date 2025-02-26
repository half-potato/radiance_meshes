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
from utils.train_util import RGB2SH, safe_exp, get_expon_lr_func
import numpy as np
import os
from plyfile import PlyData, PlyElement
from pathlib import Path
from utils.contraction import contract_points, inv_contract_points

class Model:
    def __init__(self,
                 vertices: torch.Tensor,
                 vertex_s_param: torch.Tensor,
                 vertex_rgb_param: torch.Tensor,
                 vertex_sh_param: torch.Tensor,
                 active_sh: int,
                 sh_deg: int,
                 center: torch.Tensor,
                 scene_scaling: float,
                 **kwargs):
        self.contracted_vertices = nn.Parameter(contract_points((vertices.detach() - center) / scene_scaling))
        self.center = center
        self.scene_scaling = scene_scaling
        self.vertex_s_param = vertex_s_param
        self.vertex_rgb_param = vertex_rgb_param
        self.vertex_sh_param = vertex_sh_param
        self.active_sh = active_sh
        self.sh_deg = sh_deg
        self.update_triangulation()
        self.device = vertices.device

    def inv_contract(self, points):
        return inv_contract_points(points) * self.scene_scaling + self.center

    def contract(self, points):
        return contract_points((points - self.center) / self.scene_scaling)

    @property
    def vertices(self):
        return self.inv_contract(self.contracted_vertices)

    def save2ply(self, path: Path, sample_camera: Camera):
        path.parent.mkdir(exist_ok=True, parents=True)
        
        xyz = self.vertices.detach().cpu().numpy()

        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        el = PlyElement.describe(elements, 'vertex')

        print(self.indices_np, xyz)
        dtype_tets = np.dtype([
            ('vertex_indices', 'i4', (4,)),
            # ('vertex_indices', 'O'),
            ('r', 'f4'),
            ('g', 'f4'),
            ('b', 'f4'),
            ('s', 'f4')
        ])

        # Get the color/scalar values (each element should be a tuple: (r, g, b, s))
        rgbs = self.get_cell_values(sample_camera).detach().cpu().numpy()

        tet_elements = np.empty(self.indices_np.shape[0], dtype=dtype_tets)

        # tet_elements['vertex_indices'] = list(self.indices_np)
        tet_elements['vertex_indices'] = self.indices_np
        tet_elements['r'] = rgbs[:, 0]
        tet_elements['g'] = rgbs[:, 1]
        tet_elements['b'] = rgbs[:, 2]
        tet_elements['s'] = rgbs[:, 3]

        # Create the PlyElement description for tetrahedra
        inds = PlyElement.describe(tet_elements, 'tetrahedron')
        # tets = np.array([ (features_np[i][0], features_np[i][1], features_np[i][2], features_np[i][3], self.indices_np[i].tolist()) for i in range(self.indices_np.shape[0]) ],
        #         dtype=[ ('red', 'f4'), ('green', 'f4'), ('blue', 'f4'), ('density', 'f4'), ('vertex_indices', 'i4', (4,)) ])
        # tets = PlyElement.describe(tets, 'tetrahedron')

        PlyData([el, inds]).write(str(path))

    @staticmethod
    def load_ply(path, device):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]['x']),
                        np.asarray(plydata.elements[0]['y']),
                        np.asarray(plydata.elements[0]['z'])), axis=1)
        
        s_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("s_param_")]
        s_names = sorted(s_names, key=lambda x: int(x.split('_')[-1]))
        s_param = np.stack([np.asarray(plydata.elements[0][name]) for name in s_names], axis=1)
        
        rgb_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rgb_")]
        rgb_names = sorted(rgb_names, key=lambda x: int(x.split('_')[-1]))
        rgb_param = np.stack([np.asarray(plydata.elements[0][name]) for name in rgb_names], axis=1)
        
        sh_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sh_")]
        sh_names = sorted(sh_names, key=lambda x: int(x.split('_')[-1]))
        sh_param = np.stack([np.asarray(plydata.elements[0][name]) for name in sh_names], axis=1)
        
        vertices = torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True)
        vertex_s_param = torch.tensor(s_param, dtype=torch.float, device=device).requires_grad_(True)
        vertex_rgb_param = torch.tensor(rgb_param, dtype=torch.float, device=device).requires_grad_(True)
        vertex_sh_param = torch.tensor(sh_param, dtype=torch.float, device=device).requires_grad_(True)

        sh_deg = int(math.sqrt(sh_param.shape[1] // 3 + 1)) - 1

        vertices = nn.Parameter(vertices)
        vertex_s_param = nn.Parameter(vertex_s_param)
        vertex_rgb_param = nn.Parameter(vertex_rgb_param)
        vertex_sh_param = nn.Parameter(vertex_sh_param)
        model = Model(vertices, vertex_s_param, vertex_rgb_param, vertex_sh_param, sh_deg, sh_deg,
                      torch.zeros((3), device=device), 1)
        return model

    @staticmethod
    def init_from_pcd(point_cloud, cameras, device, sh_deg, **kwargs):
        dim = 1 + 3*(sh_deg+1)**2
        torch.manual_seed(2)
        N = point_cloud.points.shape[0]
        # N = 1000
        vertices = torch.as_tensor(point_cloud.points)[:N]
        minv = vertices.min(dim=0, keepdim=True).values
        maxv = vertices.max(dim=0, keepdim=True).values
        repeats = 3
        vertices = vertices.reshape(-1, 1, 3).expand(-1, repeats, 3)
        vertices = vertices + torch.randn(*vertices.shape) * 1e-1
        vertices = vertices.reshape(-1, 3)
        # N = 10000
        # vertices = torch.cat([
        #   vertices.reshape(-1, 3),
        #   (torch.rand((N, 3)) * (maxv - minv) + minv) * 2
        # ], dim=0)
        vertices = nn.Parameter(vertices.cuda())
        vertex_s_param = (2*torch.rand((vertices.shape[0], 1), device=device)-1) + math.log(0.1)
        # vertex_rgb_param = (2*torch.rand((vertices.shape[0], 3), device=device)-1)
        vertex_rgb_param = RGB2SH(torch.as_tensor(point_cloud.colors).to(device).float())
        vertex_rgb_param = vertex_rgb_param.reshape(-1, 1, 3).expand(-1, repeats, 3).reshape(-1, 3)

        vertex_sh_param = torch.zeros((vertices.shape[0], 3*(sh_deg+1)**2 - 3), device=device)
        vertex_s_param = nn.Parameter(vertex_s_param)
        vertex_rgb_param = nn.Parameter(vertex_rgb_param)
        vertex_sh_param = nn.Parameter(vertex_sh_param)

        ccenters = torch.stack([c.camera_center.reshape(3) for c in cameras], dim=0)
        minv = ccenters.min(dim=0, keepdim=True).values
        maxv = ccenters.max(dim=0, keepdim=True).values
        center = (minv + (maxv-minv)/2).to(device)
        scaling = (maxv-minv).max().to(device)
        model = Model(vertices, vertex_s_param, vertex_rgb_param, vertex_sh_param, 0, sh_deg, center, scaling, **kwargs)
        return model

    def sh_up(self):
        self.active_sh = min(self.active_sh+1, self.sh_deg)

    def update_triangulation(self):
        v = Del(self.vertices.shape[0])
        indices_np, prev = v.compute(self.vertices.detach().cpu())
        indices_np = indices_np.numpy()
        self.indices_np = indices_np[(indices_np < self.vertices.shape[0]).all(axis=1)]
        self.indices = torch.as_tensor(self.indices_np).cuda()
        
    def get_cell_values(self, camera: Camera, mask=None):
        # directions = l2_normalize_th(self.vertices - camera.camera_center.reshape(1, 3))
        net_color = eval_sh(
            self.vertices,
            self.vertex_rgb_param,
            self.vertex_sh_param.reshape(-1, (self.sh_deg+1)**2 - 1, 3),
            camera.camera_center,
            self.active_sh)
        # feline_features = torch.cat([directions, self.vertex_rgbs_param[..., 4:]], dim=1)
        # rgb_features = self.view_dep_net(feline_features)
        # # reattach the density
        # features = torch.cat([
        #     self.vertex_rgbs_param[..., :3] + 1e-2*rgb_features, self.vertex_rgbs_param[..., 3:4]], dim=1)
        vert_color = torch.nn.functional.softplus(net_color, beta=10)
        features = torch.cat([
            vert_color, safe_exp(self.vertex_s_param)], dim=1)
        # ic(net_color, self.vertex_rgbs_param[..., 1:], vert_color, features)
        if mask is not None:
            features = features[self.indices[mask].reshape(-1)]
        else:
            features = features[self.indices.reshape(-1)]
        features = features.reshape(-1, 4, 4)
        # density = features[:, :, 3:4].sum(dim=1)
        # color = (features[:, :, :3] * features[:, :, 3:4]).sum(dim=1) / (density+1e-8)
        # features = torch.cat([color, density], dim=1)
        # features = features[:, features[:, :, 3].min(dim=1).indices]
        features = features.sum(dim=1) / 4
        # features = features.min(dim=1).values
        return features

    def __len__(self):
        return self.contracted_vertices.shape[0]

    def regularizer(self):
        return 0.0
        

class TetOptimizer:
    def __init__(self,
                 model: Model,
                 s_param_lr: float=0.025,
                 rgb_param_lr: float=0.025,
                 sh_param_lr: float=0.00025,
                 vertices_lr: float=4e-4,
                 final_vertices_lr: float=4e-7,
                 vertices_lr_delay_mult: float=0.01,
                 vertices_lr_max_steps: int=5000,
                 **kwargs):
        self.optim = optim.CustomAdam([
            # {"params": net.parameters(), "lr": 1e-3},
            {"params": [model.vertex_s_param], "lr": s_param_lr, "name": "vertex_s_param"},
            {"params": [model.vertex_rgb_param], "lr": rgb_param_lr, "name": "vertex_rgb_param"},
        ], ignore_param_list=["view_net"])
        self.sh_optim = optim.CustomAdam([
            {"params": [model.vertex_sh_param], "lr": sh_param_lr, "name": "vertex_sh_param"},
        ])
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.contracted_vertices], "lr": vertices_lr, "name": "contracted_vertices"},
        ])
        self.model = model
        self.tracker_n = 0
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None

        self.vertex_scheduler_args = get_expon_lr_func(lr_init=vertices_lr,
                                                lr_final=final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_mult,
                                                max_steps=vertices_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.vertex_optim.param_groups:
            if param_group["name"] == "contracted_vertices":
                lr = self.vertex_scheduler_args(iteration)
                param_group['lr'] = lr

    def update_ema(self):
        pass

    def add_points(self,
                   new_verts: torch.Tensor,
                   new_vert_s: torch.Tensor,
                   new_vert_rgb: torch.Tensor,
                   new_vert_sh: torch.Tensor):
        new_tensors = self.optim.cat_tensors_to_optimizer(dict(
            vertex_s_param = new_vert_s,
            vertex_rgb_param = new_vert_rgb,
            vertex_sh_param = new_vert_sh,
        ))
        self.model.vertex_s_param = new_tensors["vertex_s_param"]
        self.model.vertex_rgb_param = new_tensors["vertex_rgb_param"]
        new_tensors = self.sh_optim.cat_tensors_to_optimizer(dict(
            vertex_sh_param = new_vert_sh,
        ))
        self.model.vertex_sh_param = new_tensors["vertex_sh_param"]
        self.model.contracted_vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            contracted_vertices = self.model.contract(new_verts)
        ))['contracted_vertices']
        self.model.update_triangulation()

    def remove_points(self, mask: torch.Tensor):
        new_tensors = self.optim.prune_optimizer(mask)
        self.model.vertex_s_param = new_tensors["vertex_s_param"]
        self.model.vertex_rgb_param = new_tensors["vertex_rgb_param"]
        new_tensors = self.sh_optim.prune_optimizer(mask)
        self.model.vertex_sh_param = new_tensors["vertex_sh_param"]
        self.model.contracted_vertices = self.vertex_optim.prune_optimizer(mask)['contracted_vertices']
        self.model.update_triangulation()

    def split(self, clone_indices):
        barycentric = torch.rand((clone_indices.shape[0], clone_indices.shape[1], 1), device=self.device).clip(min=0.01, max=0.99)
        barycentric_weights = barycentric / (1e-3+barycentric.sum(dim=1, keepdim=True))
        new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights).sum(dim=1)
        new_s = (self.model.vertex_s_param[clone_indices] * barycentric_weights).sum(dim=1)
        new_rgb = (self.model.vertex_rgb_param[clone_indices] * barycentric_weights).sum(dim=1)
        new_sh = (self.model.vertex_sh_param[clone_indices] * barycentric_weights).sum(dim=1)
        self.add_points(new_vertex_location, new_s, new_rgb, new_sh)

    def track_gradients(self):
        # grad = 
        if self.vertex_rgbs_param_grad is not None:
            self.vertex_rgbs_param_grad += self.model.vertex_rgbs_param.grad
            self.vertex_grad += self.model.vertices.grad
        else:
            self.vertex_rgbs_param_grad = self.model.vertex_rgbs_param.grad
            self.vertex_grad = self.model.vertices.grad

        self.tracker_n += 1

    def get_tracker_predicates(self):
        if self.vertex_rgbs_param_grad is not None:
            grads = self.vertex_rgbs_param_grad / self.tracker_n
            vgrads = self.vertex_grad / self.tracker_n
            return grads.abs().sum(dim=-1), vgrads.abs().sum(dim=-1)
        else:
            return torch.zeros((len(self.model)), dtype=bool, device=self.model.device), torch.zeros((len(self.model)), dtype=bool, device=self.model.device)


    def reset_tracker(self):
        self.tracker_n = 0
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None

    def main_step(self):
        self.optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad()