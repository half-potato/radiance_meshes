import torch
import torch.nn as nn
import numpy as np
import math
from pathlib import Path
from plyfile import PlyData, PlyElement
from data.camera import Camera
from icecream import ic


class Model:
    def __init__(self, 
                 vertices: torch.Tensor,
                 indices: torch.Tensor,
                 tet_s_param: torch.Tensor,
                 tet_rgb_param: torch.Tensor,
                 tet_sh_param: torch.Tensor,
                 active_sh: int,
                 sh_deg: int,
                 center: torch.Tensor,
                 scene_scaling: float):
        self.vertices = nn.Parameter(vertices.detach())
        self.indices = indices
        self.tet_s_param = tet_s_param
        self.tet_rgb_param = tet_rgb_param
        self.tet_sh_param = tet_sh_param
        self.active_sh = active_sh
        self.sh_deg = sh_deg
        self.center = center
        self.scene_scaling = scene_scaling
        self.device = vertices.device

    def get_cell_values(self, camera: Camera, mask=None,
                        circumcenters=None, radii=None):
        if mask is not None:
            return self.tet_rgb_param[mask]
        else:
            return self.tet_rgb_param

    @staticmethod
    def load_ply(path, device):
        plydata = PlyData.read(path)
        ic(plydata.elements[1]['vertex_indices'])

        xyz = np.stack((np.asarray(plydata.elements[0]['x']),
                        np.asarray(plydata.elements[0]['y']),
                        np.asarray(plydata.elements[0]['z'])), axis=1)
        
        indices = np.stack(plydata.elements[1]['vertex_indices'], axis=0)
        
        # Load tetrahedral attributes
        s_param = np.asarray(plydata.elements[1]['s'])[:, np.newaxis]
        rgb = np.stack((np.asarray(plydata.elements[1]['r']),
                        np.asarray(plydata.elements[1]['g']),
                        np.asarray(plydata.elements[1]['b']),
                        np.asarray(plydata.elements[1]['s'])
                        ), axis=1)
        
        sh_names = [p.name for p in plydata.elements[1].properties if p.name.startswith("sh_")]
        sh_names = sorted(sh_names, key=lambda x: int(x.split('_')[-1]))
        if len(sh_names) > 0:
            sh_param = np.stack([np.asarray(plydata.elements[1][name]) for name in sh_names], axis=1)
            sh_deg = int(math.sqrt(sh_param.shape[1] // 3 + 1)) - 1
            tet_sh_param = torch.tensor(sh_param, dtype=torch.float, device=device).requires_grad_(True)
        else:
            sh_param = None
            sh_deg = 0
            tet_sh_param = None
        
        vertices = torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True)
        indices = torch.tensor(indices, device=device)
        tet_s_param = torch.tensor(s_param, dtype=torch.float, device=device).requires_grad_(True)
        tet_rgb_param = torch.tensor(rgb, dtype=torch.float, device=device).requires_grad_(True)


        center = torch.zeros((3), device=device)
        scaling = 1.0
        
        model = Model(vertices, indices, tet_s_param, tet_rgb_param, tet_sh_param, sh_deg, sh_deg, center, scaling)
        return model

    def __len__(self):
        return self.indices.shape[0]

    def regularizer(self):
        return 0.0