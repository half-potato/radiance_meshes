#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from data.types import ProjectionType
from icecream import ic
import math

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, fovx, fovy, image, gt_alpha_mask=None,
                 image_name=None, uid=0, cx=-1, cy=-1,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 model=ProjectionType.PERSPECTIVE, distortion_params=None,
                 exposure=1, iso=100, aperature=1):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.fovx = fovx
        self.fovy = fovy
        self.cx = cx
        self.cy = cy
        self.image_name = image_name
        self.model = model
        self.distortion_params = torch.as_tensor(distortion_params).float().clone() if distortion_params is not None else torch.zeros((4))
        self.glo_vector = None
        self.exposure = exposure
        self.iso = iso
        self.aperature = aperature

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.update()
        self.data_device = self.world_view_transform.device

    def update(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.fovx, fovY=self.fovy).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def to_dict(self, device):
        fy = fov2focal(self.fovy, self.image_height)
        fx = fov2focal(self.fovx, self.image_width)
        return dict(
            world_view_transform=self.world_view_transform.T.to(device),
            cam_pos=self.camera_center.to(device),
            # fx=fx,
            # fy=fy,
            K = torch.tensor([
                [fx, 0, self.image_width/2],
                [0, fy, self.image_height/2],
                [0, 0, 1],
            ]).to(device),
            image_height=self.image_height,
            image_width=self.image_width,
            fovy=self.fovy,
            fovx=self.fovx,
            distortion_params=self.distortion_params.to(device),
            camera_type=self.model.value,
        )

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.fovy = fovy
        self.fovx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


