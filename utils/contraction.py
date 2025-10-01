import torch
import numpy as np
from utils import safe_math
from icecream import ic

def contract_mean_std(x, std, eps:float = 1.1920928955078125e-07):#torch.finfo(torch.float).eps):
    # eps = 1e-3
    # Clamping to eps prevents non-finite gradients when x == 0.
    x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
    x_mag_sqrt = torch.sqrt(x_mag_sq)
    mask = x_mag_sq <= 1
    z = torch.where(mask, x, (safe_math.safe_div(2 * torch.sqrt(x_mag_sq) - 1, x_mag_sq)) * x)
    # z = torch.where(mask, x, (2 * torch.sqrt(x_mag_sq) - 1 / x_mag_sq) * x)
    # det_13 = ((1 / x_mag_sq) * ((2 / x_mag_sqrt - 1 / x_mag_sq) ** 2)) ** (1 / 3)
    top_det = safe_math.safe_pow(2 * x_mag_sqrt - 1, 1/3)
    det_13 = (top_det / x_mag_sqrt) ** 2
    # ic(det_13.min(), det_13.max(), x_mag_sqrt.min(), x_mag_sqrt.max(), top_det.min(), top_det.max())

    std = torch.where(mask[..., 0], std, det_13[..., 0] * std)
    return z, std


# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from icecream import ic
from utils.safe_math import safe_div

def l2_normalize_th(x, eps=torch.finfo(torch.float32).eps, dim=-1):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(
        torch.clip(torch.sum(x**2, dim=dim, keepdim=True), eps, None)
    )


def inv_contract_points(z, eps=torch.finfo(torch.float32).eps):
    z_mag_sq = torch.sum(z**2, dim=-1, keepdims=True)
    z_mag_sq = torch.maximum(torch.ones_like(z_mag_sq), z_mag_sq)
    inv_scale = 2 * torch.sqrt(z_mag_sq.clip(min=eps)) - z_mag_sq
    x = safe_div(z, inv_scale)
    return x

def contract_points(x, eps=torch.finfo(torch.float32).eps, dim=-1):
    mag = torch.sqrt(
        torch.clip(torch.sum(x**2, dim=dim, keepdim=True), eps, None)
    )
    return torch.where(mag <= 1, x, (2 - (1 / mag)) * (x / mag))