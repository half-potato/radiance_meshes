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
import time
from pathlib import Path
from typing import *

import slangpy
import torch
from torch.autograd import Function
from icecream import ic

kernels = slangpy.loadModule(
    str(Path(__file__).parent / "slang/sh_kernel.slang")
)

class EvalSH(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        sh0: torch.Tensor,
        features: torch.Tensor,
        rayo: torch.Tensor,
        sh_degree: int
    ):
        block_size = 64
        rayo = rayo.reshape(3).contiguous()
        means = means.contiguous()
        sh0 = sh0.contiguous()
        features = features.contiguous()
        color = torch.zeros_like(means)
        ctx.sh_degree = sh_degree
        num_prim = means.shape[0]
        kernels.sh_kernel(
            means=means, sh0=sh0, features=features, ray_origin=rayo, colors=color, sh_degree=sh_degree
        ).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=(num_prim // block_size + 1, 1, 1),
        )

        ctx.save_for_backward(
            means, sh0, features, rayo, color
        )
        return color

    @staticmethod
    def backward(ctx, dL_dcolor: torch.Tensor):
        block_size = 64
        means, sh0, features, rayo, color = ctx.saved_tensors
        num_prim = means.shape[0]
        dL_dfeat = torch.zeros_like(features)
        dL_dsh0 = torch.zeros_like(sh0)
        dL_dcolor = dL_dcolor.contiguous()
        kernels.bw_sh_kernel(means=means, sh0=sh0, features=features, dL_dsh0=dL_dsh0, dL_dfeatures=dL_dfeat, ray_origin=rayo, dL_dcolors=dL_dcolor, sh_degree=ctx.sh_degree).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=((num_prim + block_size - 1) // block_size, 1, 1),
        )
        # return torch.zeros_like(means), dL_dfeat, torch.zeros_like(rayo), torch.zeros_like(front_dir), None
        return None, dL_dsh0, dL_dfeat, None, None, None


def eval_sh(
        means,
        sh0,
        features,
        rayo,
        sh_degree):
    out = EvalSH.apply(
        means,
        sh0,
        features,
        rayo,
        sh_degree
    )
    return out

