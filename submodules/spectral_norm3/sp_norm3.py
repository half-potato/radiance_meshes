# coding=utf-8
# Copyright 2025 Alexander Mai
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

import slangtorch
import torch
from torch.autograd import Function
from icecream import ic

kernels = slangtorch.loadModule(
    str(Path(__file__).parent / "sp_norm3.slang"),
)

class ComputeSpectralNorm(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(
        ctx: Any,
        matrices: torch.Tensor,
    ):
        block_size = 64
        num_prim = matrices.shape[0]
        norm = torch.zeros((num_prim), device=matrices.device)
        kernels.sp_norm3(
            matrices=matrices, norm=norm
        ).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=(num_prim // block_size + 1, 1, 1),
        )

        ctx.save_for_backward(
            matrices, norm
        )
        return norm

    @staticmethod
    def backward(ctx, norm_grad: torch.Tensor):
        block_size = 64
        matrices, norm = ctx.saved_tensors
        matrices_grad = torch.zeros_like(matrices)
        num_prim = matrices.shape[0]
        kernels.sp_norm3.bwd(matrices=(matrices, matrices_grad), norm=(norm, norm_grad)).launchRaw(
            blockSize=(block_size, 1, 1),
            gridSize=((num_prim + block_size - 1) // block_size, 1, 1),
        )
        # return torch.zeros_like(means), dL_dfeat, torch.zeros_like(rayo), torch.zeros_like(front_dir), None
        return matrices_grad

compute_spectral_norm3 = ComputeSpectralNorm.apply