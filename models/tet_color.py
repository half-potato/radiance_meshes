import torch
import torch.nn as nn
import numpy as np
import math
from pathlib import Path
from plyfile import PlyData, PlyElement
from data.camera import Camera
from icecream import ic
import tinyplypy
from models.ingp_color import to_sphere, l2_normalize_th, light_function


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
        self.dir_offset = torch.tensor([
            [0, 0],
            [math.pi, 0],
        ], device=self.device)
        self.num_lights = 2

    def get_cell_values(self, camera: Camera, mask=None,
                        circumcenters=None, radii=None):
        if mask is not None:
            indices = self.indices[mask]
            output = self.tet_rgb_param[mask]
        else:
            indices = self.indices
            output = self.tet_rgb_param

        base_color = output[:, :3]
        density = output[:, 3:4]
        # lights = output[:, 4:].reshape(-1, self.num_lights, 6)
        lights = output[:, 4:].reshape(-1, 6, self.num_lights).permute(0, 2, 1)
        light_colors = lights[:, :, :3]
        light_roughness = lights[:, :, 3:4]
        reflection_dirs = lights[:, :, 4:6] + self.dir_offset.reshape(1, -1, 2)

        reflection_dirs = to_sphere(reflection_dirs)
        barycenters = self.vertices[indices].mean(dim=1)
        # ic(output[0])
        # ic(light_colors.min(), light_colors.max(), light_roughness, reflection_dirs)
        # ic(barycenters.shape, self.vertices.shape, indices.shape)
        view_dirs = l2_normalize_th(camera.camera_center.reshape(1, 3) - barycenters).reshape(-1, 1, 3)
        color =  light_function(base_color, reflection_dirs, light_colors, light_roughness, view_dirs)
        features = torch.cat([
            color, density], dim=1)
        return features

    @staticmethod
    def load_ply(path, device):
        """
        Load a .ply file using tinyply_numpy_binding, mirroring the old load_ply logic.
        Returns (vertices, indices, tet_s_param, tet_rgb_param, tet_sh_param, sh_deg).
        """

        # 1. Read the file with tinyply
        data_dict = tinyplypy.read_ply(str(path))
        # Expecting something like:
        # {
        #    "vertex": {
        #       "x": np.array(...),
        #       "y": np.array(...),
        #       "z": np.array(...),
        #       ...
        #    },
        #    "tetrahedron": {
        #       "vertex_indices": np.array(... shape=(N,4)),
        #       "r": ...,
        #       "g": ...,
        #       "b": ...,
        #       "s": ...,
        #       "sh_0": ...,
        #       "sh_1": ...,
        #       ...
        #    }
        # }

        # 2. Gather vertex positions (x,y,z)
        vx = data_dict["vertex"]["x"]
        vy = data_dict["vertex"]["y"]
        vz = data_dict["vertex"]["z"]
        xyz = np.stack([vx, vy, vz], axis=1)  # shape (num_vertices, 3)

        # 3. Gather tetrahedron element properties
        tet_data = data_dict["tetrahedron"]
        indices = tet_data["vertex_indices"]  # shape (N,4)
        r = tet_data["r"]
        g = tet_data["g"]
        b = tet_data["b"]
        s = tet_data["s"]  # shape (N,)

        # Create a 4-channel array (r,g,b,s)
        rgb = np.stack([r, g, b, s], axis=1)  # shape (N,4)

        # 4. Detect any SH properties (named "sh_0", "sh_1", etc.)
        sh_names = [k for k in tet_data.keys() if k.startswith("sh_")]
        sh_names = sorted(sh_names, key=lambda x: int(x.split('_')[-1]))  # sort by index suffix
        if len(sh_names) > 0:
            sh_param = np.stack([tet_data[name] for name in sh_names], axis=1)  # shape (N, numSH)
            # e.g., if this is an RGB SH, numSH is 3 * (#coeffs)
            sh_deg = int(math.sqrt(sh_param.shape[1] // 3 + 1)) - 1
            tet_sh_param = torch.tensor(sh_param, dtype=torch.float, device=device, requires_grad=True)
        else:
            num_lights = 2
            l = []
            for i in range(num_lights):
                l.extend([
                    tet_data[f"l{i}_r"],
                    tet_data[f"l{i}_g"],
                    tet_data[f"l{i}_b"],
                    tet_data[f"l{i}_roughness"],
                    tet_data[f"l{i}_phi"],
                    tet_data[f"l{i}_theta"]])
            l = np.stack(l, axis=1)
            rgb = np.concatenate([rgb, l], axis=1)
            sh_param = None
            sh_deg = 0
            tet_sh_param = None


        # 5. Convert everything to torch Tensors
        vertices = torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True)
        indices = torch.tensor(indices.astype(np.int32), dtype=torch.int32, device=device)  # typically int64 for indexing
        tet_s_param = torch.tensor(s[:, np.newaxis], dtype=torch.float, device=device).requires_grad_(True)
        tet_rgb_param = torch.tensor(rgb, dtype=torch.float, device=device).requires_grad_(True)

        center = torch.zeros((3), device=device)
        scaling = 1.0
        
        model = Model(vertices, indices, tet_s_param, tet_rgb_param, tet_sh_param, sh_deg, sh_deg, center, scaling)
        return model

    # @staticmethod
    # def load_ply(path, device):
    #     plydata = PlyData.read(path)
    #     ic(plydata.elements[1]['vertex_indices'])

    #     xyz = np.stack((np.asarray(plydata.elements[0]['x']),
    #                     np.asarray(plydata.elements[0]['y']),
    #                     np.asarray(plydata.elements[0]['z'])), axis=1)
        
    #     indices = np.stack(plydata.elements[1]['vertex_indices'], axis=0)
        
    #     # Load tetrahedral attributes
    #     s_param = np.asarray(plydata.elements[1]['s'])[:, np.newaxis]
    #     rgb = np.stack((np.asarray(plydata.elements[1]['r']),
    #                     np.asarray(plydata.elements[1]['g']),
    #                     np.asarray(plydata.elements[1]['b']),
    #                     np.asarray(plydata.elements[1]['s'])
    #                     ), axis=1)
        
    #     sh_names = [p.name for p in plydata.elements[1].properties if p.name.startswith("sh_")]
    #     sh_names = sorted(sh_names, key=lambda x: int(x.split('_')[-1]))
    #     if len(sh_names) > 0:
    #         sh_param = np.stack([np.asarray(plydata.elements[1][name]) for name in sh_names], axis=1)
    #         sh_deg = int(math.sqrt(sh_param.shape[1] // 3 + 1)) - 1
    #         tet_sh_param = torch.tensor(sh_param, dtype=torch.float, device=device).requires_grad_(True)
    #     else:
    #         sh_param = None
    #         sh_deg = 0
    #         tet_sh_param = None
        
    #     vertices = torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True)
    #     indices = torch.tensor(indices, device=device)
    #     tet_s_param = torch.tensor(s_param, dtype=torch.float, device=device).requires_grad_(True)
    #     tet_rgb_param = torch.tensor(rgb, dtype=torch.float, device=device).requires_grad_(True)


    #     center = torch.zeros((3), device=device)
    #     scaling = 1.0
        
    #     model = Model(vertices, indices, tet_s_param, tet_rgb_param, tet_sh_param, sh_deg, sh_deg, center, scaling)
    #     return model

    def __len__(self):
        return self.indices.shape[0]

    def regularizer(self):
        return 0.0