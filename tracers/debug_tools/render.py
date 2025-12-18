import torch

import build.splinetracer.extension.splinetracer_cpp_extension as sp
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import math

import imageio
import matplotlib.pyplot as plt
from splinetracer import trace_rays
from utils.math_util import *

DIM = 3
QDIM = 1 if DIM == 2 else 4

"""
def render_quadrature(tdist, query_fn, return_extras=False):
    t_avg = 0.5 * (tdist[..., 1:] + tdist[..., :-1])
    t_delta = jnp.diff(tdist)
    total_density, avg_colors = query_fn(t_avg)
    weights = compute_alpha_weights_helper(total_density * t_delta)
    rendered_color = jnp.sum(
        weights[..., None] * avg_colors, axis=-2
    )  # Assuming the bg color is 0.

    if return_extras:
        return rendered_color, {
            "tdist": tdist,
            "avg_colors": avg_colors,
            "weights": weights,
            "total_density": total_density,
        }
    else:
        return rendered_color
"""


@jax.jit
def linear_ellipsoid(pos, scales, rot, peak, x):
    stretch_coords = (rot.T @ (x - pos)) / scales
    d = jnp.linalg.norm(stretch_coords, axis=-1, ord=1)
    # d = jnp.linalg.norm(stretch_coords, axis=-1, ord=np.inf)
    # return peak * (1+jnp.cos(2 * jnp.clip(d**2, 0, np.pi/2)))/2
    # return peak * (1-jnp.sin(jnp.clip(d**2, 0, np.pi/2)))/2
    return peak*np.clip(1 - d, 0, None)
# return jnp.maximum(1 - d, 0)

def transform(pos, scales, rot, x):
    stretch_coords = (rot.T @ (x - pos)) / scales
    return stretch_coords

vtransform = jax.vmap(transform, in_axes=[None, None, None, 0])

# return peak*np.clip(1 - stretch_coords.T @ stretch_coords, 0, None)
# return peak*np.clip(1 - jnp.linalg.norm(stretch_coords, axis=-1, ord=2), 0, None)
# return jnp.linalg.norm(stretch_coords, axis=-1, ord=2)

vlinear_ellipsoid = jax.vmap(linear_ellipsoid, in_axes=[None, None, None, None, 0])


T = np.eye(4)
# Init rays
res = 3
fov = 0.1 * math.pi / 180
fx = res /2 /math.tan(fov/2)
radius = 2
dx, dy = np.meshgrid(
    np.linspace(-res/2, res/2, res) / fx,
    np.linspace(-res/2, res/2, res) / fx,
)
crayd = l2_normalize(np.stack([ dx, dy, np.ones_like(dx)], axis=-1)).reshape(-1, 3)
rayo, rayd = get_rays(crayd, T)
rayo = rayo.reshape(res, res, 3)
rayd = rayd.reshape(res, res, 3)

# TODO REMOVE
rayo = rayo[1:2, 1:2]
rayd = rayd[1:2, 1:2]
res = 1

all_xs = np.linspace(0, 3, 10000)
xs = (all_xs[:-1] + all_xs[1:])/2
dt = all_xs[1:] - all_xs[:-1]


quat = l2_normalize(np.random.rand(4))
# quat = l2_normalize(np.array([1, 0.1, 0, 0]))
rot = quatToMat3(quat)
print(f"Quat: {quat.tolist()}")
mean = np.array([0, 1e-4, 1])
s = 1e-3
scale = s*np.array([1,1,1])
density = 1

color = np.array([0.1, 0.1, 1]).reshape(1, 3)
im = np.zeros((res, res, 3), dtype=np.float32)
for i in range(res):
    for j in range(res):
        ts = xs.reshape(-1, 1) * rayd[i, j][None, :] + rayo[i, j][None, :]
        sigma = vlinear_ellipsoid(mean, scale, rot, density, ts)
        
        weight = compute_alpha_weights_helper(sigma.reshape(1, -1) * dt.reshape(1, -1))
        # print(weight.sum(), 1-np.exp(-(sigma * dt).sum()))
        # weight, transmit = raw2alpha(sigma.reshape(-1, 1), dt.reshape(-1, 1))
        final_col = (weight.reshape(-1, 1) * color).sum(axis=0)
        im[i, j] = final_col
byte_im = np.clip(im*255, 0, 255).astype(np.uint8)
imageio.imwrite("gt.png", byte_im)

ts = all_xs.reshape(-1, 1) * rayd[res//2, res//2, :] + rayo[res//2, res//2, :]
T = np.eye(4)
T[:3, :3] = np.diag(1 / scale) @ rot.T
T[:3, 3] = -T[:3, :3] @ mean

Trd = T[:3, :3] @ rayd[res//2, res//2]
stretch_coords = (T[:3, :3] @ ts.reshape(-1, 3).T + T[:3, 3].reshape(3, 1)).T
raw_sigma = 1 - np.linalg.norm(stretch_coords, axis=-1, ord=1)
dsigma_pred = np.where(
    raw_sigma > 0, np.sum(-np.sign(stretch_coords) * Trd.reshape(1, 3), axis=-1), 0
)
sigma = vlinear_ellipsoid(mean, scale, rot, density, ts)
dsigma = (sigma[1:] - sigma[:-1]) / dt
plt.plot(xs, dsigma)

device = torch.device(0)
rayo_th = torch.as_tensor(rayo.reshape(-1, 3)).float().to(device)
rayd_th = torch.as_tensor(rayd.reshape(-1, 3)).float().to(device)

#"""
# out = trace_rays(
#     torch.as_tensor(mean).reshape(1, 3).float(),
#     torch.as_tensor(scale).reshape(1, 3).float(),
#     torch.as_tensor(quat).reshape(1, 4).float(),
#     torch.tensor([density]).reshape(1).float(),
#     torch.as_tensor(color).reshape(1, 3).float(),
#     rayo_th, rayd_th)
# out = out.reshape(res, res, 3).cpu().numpy()

"""
device = torch.device(0)
ctx = sp.OptixContext(device)
prims = sp.Primitives(device)
# prims = sp.Primitives("/usr/local/google/home/alexmai/optix-examples/models/lego.ply", device)
means_th = torch.as_tensor(mean).reshape(1, 3).float().to(device)
scale_th = torch.as_tensor(scale).reshape(1, 3).float().to(device)
quat_th = torch.as_tensor(quat).reshape(1, 4).float().to(device)
density_th = torch.tensor([density]).reshape(1).float().to(device)
color_th = torch.as_tensor(color).reshape(1, 3).float().to(device)
prims.add_primitives(means_th, scale_th, quat_th, density_th, color_th)
    
    
gas = sp.GAS(ctx, device, prims)

forward = sp.Forward(ctx, device, prims, True)
backward = sp.Backward(ctx, device, prims)
out = forward.trace_rays(gas, rayo_th, rayd_th)
print("Backward")
bwout = backward.trace_rays(gas, rayo_th, rayd_th, out['saved'], -torch.ones_like(out['color'][:, :3]))
print(bwout)

out = out['color'].reshape(res, res, 4).cpu().numpy()

# print(f"Density under curve: {(sigma[:-1] * dt).sum()}")
# print(f"Total transmittance: {1-np.exp(-(sigma[:-1] * dt).sum())}")
print(f"pred: {out[res//2, res//2]}")
print(f"gt: {im[res//2, res//2]}")
mse = ((im - out[:, :, :3])**2).mean()
print(f"MSE: {mse}")
#"""
args = (
    torch.as_tensor(mean).reshape(1, 3).float().to(device),
    torch.as_tensor(scale).reshape(1, 3).float().to(device),
    torch.as_tensor(quat).reshape(1, 4).float().to(device),
    torch.tensor([density]).reshape(1).float().to(device),
    torch.as_tensor(color).reshape(1, 3).float().to(device),
    rayo_th, rayd_th
)
for arg in args[3:]:
    arg.requires_grad = True

torch.autograd.gradcheck(trace_rays, args, eps=1e-03, atol=1e-05, rtol=0.001)

imageio.imwrite("pred.png", np.clip(out[:, :, :3]*255, 0, 255).astype(np.uint8))

# rayo = torch.tensor([0, 0, 0], device=device, dtype=torch.float32)
# rayd = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
# out = forward.trace_rays(gas, rayo, rayd)
# print(out)
plt.show()
plt.plot(all_xs, sigma)
plt.show()
