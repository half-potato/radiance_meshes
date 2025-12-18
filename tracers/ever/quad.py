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

import jax
import jax.numpy as jnp
import torch
from jax import config
config.update("jax_enable_x64", True)
import numpy as np

from utils import math_util
from jaxutil import safe_math, quadrature

def query_tetra( tdist, rayo, rayd, params):
    xs = rayo.reshape(1, 3) + tdist.reshape(-1, 1) * rayd.reshape(1, 3)
    R = math_util.jquatToMat3(math_util.jl2_normalize(params['quat']))
    sc = (((xs - params['mean'].reshape(1, 3)) @ R.T) / jnp.maximum(params['scale'].reshape(1, 3), 1e-8))
    d = jnp.linalg.norm(sc, axis=-1, ord=1).reshape(-1, 1) < 1
    mask = d & ((sc[..., 0] > 0) & (sc[..., 1] > 0) & (sc[..., 2] > 0)).reshape(-1, 1)
    densities = jnp.where(mask, params['density'], 0)
    colors = jnp.where(mask, params['density']*params['features'].reshape(1, 3), 0)
    return densities, colors

def query_l1( tdist, rayo, rayd, params):
    xs = rayo.reshape(1, 3) + tdist.reshape(-1, 1) * rayd.reshape(1, 3)
    R = math_util.jquatToMat3(math_util.jl2_normalize(params['quat']))
    sc = (((xs - params['mean'].reshape(1, 3)) @ R.T) / jnp.maximum(params['scale'].reshape(1, 3), 1e-8))
    d = jnp.linalg.norm(sc, axis=-1, ord=1)
    densities = params['density'] * jnp.clip(1-d, 0, None).reshape(-1, 1)
    colors = params['features'].reshape(1, 3) * densities
    return densities, colors


def query_d8( tdist, rayo, rayd, params):
    xs = rayo.reshape(1, 3) + tdist.reshape(-1, 1) * rayd.reshape(1, 3)
    R = math_util.jquatToMat3(math_util.jl2_normalize(params['quat']))
    sc = (((xs - params['mean'].reshape(1, 3)) @ R.T) / jnp.maximum(params['scale'].reshape(1, 3), 1e-8))
    d = jnp.linalg.norm(sc, axis=-1, ord=1).reshape(-1, 1)
    densities = jnp.where(d < 1, params['density'], 0)
    colors = jnp.where(d < 1, params['density']*params['features'].reshape(1, 3), 0)
    return densities, colors

def query_ellipsoid( tdist, rayo, rayd, params):
    xs = rayo.reshape(1, 3) + tdist.reshape(-1, 1) * rayd.reshape(1, 3)
    R = math_util.jquatToMat3(math_util.jl2_normalize(params['quat']))
    sc = (((xs - params['mean'].reshape(1, 3)) @ R.T) / jnp.maximum(params['scale'].reshape(1, 3), 1e-8))
    d = jnp.linalg.norm(sc, axis=-1, ord=2).reshape(-1, 1)
    densities = jnp.where(d < 1, params['density'], 0)
    colors = jnp.where(d < 1, params['density']*params['features'].reshape(1, 3), 0)
    return densities, colors


def SH2RGB(x):
    return 0.28209479177387814 * x + 0.5

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
    kernel = query_ellipsoid
):

    vquery_ellipsoid = jax.vmap(kernel, in_axes=(None, None, None, {
        'mean': 0,
        'quat': 0,
        'density': 0,
        'scale': 0,
        'features': 0,
    }))

    def sum_vquery_ellipsoid( tdist, rayo, rayd, params):
        densities, colors = vquery_ellipsoid(tdist, rayo, rayd, params)
        density = densities.sum(axis=0)
        colors = safe_math.safe_div(colors.sum(axis=0), density)
        return density.reshape(-1), colors.clip(min=0)

    num_quad = 2**16
    tdist = jnp.linspace(*(tmin, tmax), num_quad + 1)
    params = {
        'mean': mean.detach().cpu().numpy().astype(np.float64).reshape(-1, 3),
        'scale': scale.detach().cpu().numpy().astype(np.float64).reshape(-1, 3),
        'quat': quat.detach().cpu().numpy().astype(np.float64).reshape(-1, 4),
        'density': density.detach().cpu().numpy().astype(np.float64).reshape(-1, 1),
        'features': SH2RGB(features).detach().cpu().numpy().astype(np.float64).reshape(-1, 3),
    }
    rayo = rayo.detach().cpu().numpy().astype(np.float64)
    rayd = rayd.detach().cpu().numpy().astype(np.float64)
    out, extras = quadrature.render_quadrature(
        tdist,
        lambda t: sum_vquery_ellipsoid(t, rayo, rayd, params),
        return_extras=return_extras,
    )
    return out, extras

trace_rays.uses_density = True
