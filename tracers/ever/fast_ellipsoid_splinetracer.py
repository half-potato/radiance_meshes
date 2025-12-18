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

import slangtorch
import torch
from torch.autograd import Function
from icecream import ic

import sys
sys.path.append(str(Path(__file__).parent))

from build.splinetracer.extension import fast_ellipsoid_splinetracer_cpp_extension as sp
kernels = slangtorch.loadModule(
    str(Path(__file__).parent / "fast_ellipsoid_splinetracer/slang/backwards_kernel.slang"),
    includePaths=[str(Path(__file__).parent / 'slang')]
)

otx = sp.OptixContext(torch.device("cuda:0"))


# Inherit from Function
class SplineTracer(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(
        ctx: Any,
        mean: torch.Tensor,
        scale: torch.Tensor,
        quat: torch.Tensor,
        density: torch.Tensor,
        color: torch.Tensor,
        rayo: torch.Tensor,
        rayd: torch.Tensor,
        tmin: float,
        tmax: float,
        max_prim_size: float,
        mean2D: torch.Tensor,
        wcts: torch.Tensor,
        max_iters: int,
        return_extras: bool = False,
    ):
        ctx.device = rayo.device
        st = time.time()
        ctx.prims = sp.Primitives(ctx.device)
        assert mean.device == ctx.device
        mean = mean.contiguous()
        scale = scale.contiguous()
        density = density.contiguous()
        quat = quat.contiguous()
        color = color.contiguous()
        half_attribs = torch.cat([mean, scale, quat], dim=1).half().contiguous()
        ctx.prims.add_primitives(mean, scale, quat, half_attribs, density, color)

        ctx.gas = sp.GAS(otx, ctx.device, ctx.prims, True, False, True)

        ctx.forward = sp.Forward(otx, ctx.device, ctx.prims, True)
        st = time.time()
        ctx.max_iters = max_iters
        out = ctx.forward.trace_rays(ctx.gas, rayo, rayd, tmin, tmax, ctx.max_iters, max_prim_size)
        ctx.saved = out["saved"]
        ctx.max_prim_size = max_prim_size
        ctx.tmin = tmin
        ctx.tmax = tmax
        tri_collection = out["tri_collection"]

        states = ctx.saved.states.reshape(rayo.shape[0], -1)
        distortion_pt1 = states[:, 0]
        distortion_pt2 = states[:, 1]
        distortion_loss = (distortion_pt1 - distortion_pt2)
        color_and_loss = torch.cat([out["color"], distortion_loss.reshape(-1, 1)], dim=1)

        initial_inds = out['initial_touch_inds'][:out['initial_touch_count'][0]]

        ctx.save_for_backward(
            mean, scale, quat, density, color, rayo, rayd, tri_collection, wcts, out['initial_drgb'], initial_inds, half_attribs
        )

        if return_extras:
            return color_and_loss, dict(
                tri_collection=tri_collection,
                iters=ctx.saved.iters,
                opacity=out["color"][:, 3],
                touch_count=ctx.saved.touch_count,
                distortion_loss=distortion_loss,
                saved=ctx.saved,
            )
        else:
            return color_and_loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, return_extras=False):
        (
            mean,
            scale,
            quat,
            density,
            features,
            rayo,
            rayd,
            tri_collection,
            wcts,
            initial_drgb,
            initial_inds,
            half_attribs
        ) = ctx.saved_tensors
        device = ctx.device

        start_ids = torch.cumsum(ctx.saved.iters, dim=0).int()
        num_prims = mean.shape[0]
        num_rays = rayo.shape[0]
        dL_dmeans = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
        dL_dscales = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
        dL_dquats = torch.zeros((num_prims, 4), dtype=torch.float32, device=device)
        dL_ddensities = torch.zeros((num_prims), dtype=torch.float32, device=device)
        dL_dfeatures = torch.zeros_like(features)
        dL_drayo = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)
        dL_drayd = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)

        dL_dmeans2D = torch.zeros((num_prims, 2), dtype=torch.float32, device=device)

        touch_count = torch.zeros((num_prims), dtype=torch.int32, device=device)

        dL_dinital_drgb = torch.zeros((num_rays, 4), dtype=torch.float32, device=device)

        block_size = 16
        st = time.time()
        if ctx.saved.iters.sum() > 0:

            dual_model = (
                mean,
                scale,
                quat,
                density,
                features,

                dL_dmeans,
                dL_dscales,
                dL_dquats,
                dL_ddensities,
                dL_dfeatures,
                dL_drayo,
                dL_drayd,
                dL_dmeans2D,
            )

            kernels.backwards_kernel(
                last_state=ctx.saved.states,
                last_dirac=ctx.saved.diracs,
                iters=ctx.saved.iters,
                tri_collection=tri_collection,
                ray_origins=rayo,
                ray_directions=rayd,
                model=dual_model,
                initial_drgb=initial_drgb,
                dL_dinital_drgb=dL_dinital_drgb,
                touch_count=touch_count,
                dL_doutputs=grad_output.contiguous(),
                wcts=wcts if wcts is not None else torch.ones((1, 4, 4), device=device, dtype=torch.float32),
                tmin=ctx.tmin,
                tmax=ctx.tmax,
                max_prim_size=ctx.max_prim_size,
                max_iters=ctx.max_iters,
            ).launchRaw(
                blockSize=(block_size, 1, 1),
                gridSize=(num_rays // block_size + 1, 1, 1),
            )
            if initial_inds.shape[0] > 0:
                ray_block_size = 64;
                second_block_size = 16;
                kernels.backwards_initial_drgb_kernel(
                    ray_origins=rayo,
                    ray_directions=rayd,
                    model=dual_model,
                    initial_drgb=initial_drgb,
                    initial_inds=initial_inds,
                    dL_dinital_drgb=dL_dinital_drgb,
                    touch_count=touch_count,
                    tmin=ctx.tmin,
                ).launchRaw(
                    blockSize=(ray_block_size, second_block_size, 1),
                    gridSize=(
                        rayo.shape[0] // ray_block_size + 1,
                        initial_inds.shape[0] // second_block_size + 1,
                        1),
                )
        v = 1e+3
        mean_v = 1e+3
        dL_dmeans2D = None if wcts is None else dL_dmeans2D
        return (
            dL_dmeans.clip(min=-mean_v, max=mean_v),
            dL_dscales.clip(min=-v, max=v),
            dL_dquats.clip(min=-v, max=v),
            dL_ddensities.clip(min=-50, max=50).reshape(density.shape),
            dL_dfeatures.clip(min=-v, max=v),
            dL_drayo.clip(min=-v, max=v),
            dL_drayd.clip(min=-v, max=v),
            None,
            None,
            None,
            dL_dmeans2D,
            None,
            None,
            None,
        )


def trace_rays(
    mean: torch.Tensor,
    scale: torch.Tensor,
    quat: torch.Tensor,
    density: torch.Tensor,
    features: torch.Tensor,
    rayo: torch.Tensor,
    rayd: torch.Tensor,
    tmin: float = 0.0,
    tmax: float = 1000,
    max_prim_size: float = 3,
    dL_dmeans2D=None,
    wcts=None,
    max_iters: int = 500,
    return_extras: bool = False,
):
    out = SplineTracer.apply(
        mean,
        scale,
        quat,
        density,
        features,
        rayo,
        rayd,
        tmin,
        tmax,
        max_prim_size,
        dL_dmeans2D,
        wcts,
        max_iters,
        return_extras,
    )
    return out

trace_rays.uses_density = True
