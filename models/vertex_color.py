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
from utils.train_util import RGB2SH, safe_exp

class Model:
    def __init__(self,
                 vertices: torch.Tensor,
                 vertex_s_param: torch.Tensor,
                 vertex_rgb_param: torch.Tensor,
                 vertex_sh_param: torch.Tensor,
                 active_sh: int,
                 sh_deg: int):
        self.vertices = vertices
        self.vertex_s_param = vertex_s_param
        self.vertex_rgb_param = vertex_rgb_param
        self.vertex_sh_param = vertex_sh_param
        self.active_sh = active_sh
        self.sh_deg = sh_deg
        self.update_triangulation()
        self.device = vertices.device

    @staticmethod
    def init_from_pcd(point_cloud, sh_deg, device):
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
        model = Model(vertices, vertex_s_param, vertex_rgb_param, vertex_sh_param, 0, sh_deg)
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
        features = features.reshape(-1, 4, 4).sum(dim=1) / 4
        return features

    def __len__(self):
        return self.vertices.shape[0]
        

class TetOptimizer:
    def __init__(self,
                 model: Model,
                 view_net_lr: float=1e-3,
                 s_param_lr: float=0.025,
                 rgb_param_lr: float=0.025,
                 sh_param_lr: float=0.00025,
                 vertices_lr: float=4e-4):
        self.optim = optim.CustomAdam([
            # {"params": net.parameters(), "lr": 1e-3},
            {"params": [model.vertex_s_param], "lr": s_param_lr, "name": "vertex_s_param"},
            {"params": [model.vertex_rgb_param], "lr": rgb_param_lr, "name": "vertex_rgb_param"},
        ], ignore_param_list=["view_net"])
        self.sh_optim = optim.CustomAdam([
            {"params": [model.vertex_sh_param], "lr": sh_param_lr, "name": "vertex_sh_param"},
        ])
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.vertices], "lr": vertices_lr, "name": "vertices"},
        ])
        self.model = model
        self.tracker_n = 0
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None

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
        self.model.vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            vertices = new_verts
        ))['vertices']
        self.model.update_triangulation()

    def remove_points(self, mask: torch.Tensor):
        new_tensors = self.optim.prune_optimizer(mask)
        self.model.vertex_s_param = new_tensors["vertex_s_param"]
        self.model.vertex_rgb_param = new_tensors["vertex_rgb_param"]
        new_tensors = self.sh_optim.prune_optimizer(mask)
        self.model.vertex_sh_param = new_tensors["vertex_sh_param"]
        self.model.vertices = self.vertex_optim.prune_optimizer(mask)['vertices']
        self.model.update_triangulation()

    def split(self, clone_indices, barycentric_weights):
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