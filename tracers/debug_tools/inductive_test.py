
from splinetracer import trace_rays
import numpy as np
import math
from util import *
from torch.nn import Parameter
import torch
import imageio
import jax_render

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

np.random.seed(0)
#============================
# Camera SETUP
#============================
T = np.eye(4)
# Init rays
res = 800
# fov = 10 * math.pi / 180
fov = 0.5 * math.pi / 180
fx = res /2 /math.tan(fov/2)
radius = 2
dx, dy = np.meshgrid(
    np.linspace(-res/2, res/2, res) / fx,
    np.linspace(-res/2, res/2, res) / fx,
)
crayd = l2_normalize(np.stack([ dx, dy, np.ones_like(dx)], axis=-1)).reshape(-1, 3)
rayo, rayd = get_rays(crayd, T)
rayo = rayo.reshape(res, res, 3).reshape(-1, 3)
rayd = rayd.reshape(res, res, 3).reshape(-1, 3)

# TODO REMOVE
# rayo = rayo[1:2, 1:2]
# rayd = rayd[1:2, 1:2]
# res = 1

all_xs = np.linspace(0, 3, 10000)
xs = (all_xs[:-1] + all_xs[1:])/2
dt = all_xs[1:] - all_xs[:-1]

device = torch.device(0)
rayo_th = torch.as_tensor(rayo.reshape(-1, 3)).float().to(device)
rayd_th = torch.as_tensor(rayd.reshape(-1, 3)).float().to(device)

#============================
# SCENE SETUP
#============================
quat = l2_normalize(np.random.rand(4))
rot = quatToMat3(quat)
# print(f"Quat: {quat.tolist()}")
mean = np.array([0, 2e-3, 1])
s = np.log(1e-1)
scale = s*np.array([1.0,0.8,0.6], dtype=np.float32)
# scale = s*np.array([1,1,1])
# density = -13
density = 2
color = np.array([0.1, 0.1, 1]).reshape(1, 3)

means = Parameter(torch.as_tensor(mean).reshape(1, 3).float().to(device))
scales = Parameter(torch.as_tensor(scale).reshape(1, 3).float().to(device))
quats = Parameter(torch.as_tensor(quat).reshape(1, 4).float().to(device))
densities = Parameter(torch.tensor([density]).reshape(1).float().to(device))
colors = Parameter(torch.as_tensor(color).reshape(1, 3).float().to(device))

# optim = torch.optim.Adam([means, scales, quats, densities, colors], lr=1e-3)
optim = torch.optim.SGD([means, scales, quats, densities, colors], lr=1e+2, momentum=0.9)

n = 2
ascales = scales.exp()
aquats = l2_normalize_th(quats)
area = (ascales[:, 0]**n + ascales[:, 1]**n + ascales[:, 2]**n)**(1/n)
adensities = (densities).exp() / area
# adensities = (densities - area).exp()
acolors = torch.sigmoid(colors)
print(acolors)
params = dict(
    mean=means.cpu().detach().numpy().reshape(-1),
    scale=ascales.cpu().detach().numpy().reshape(-1),
    quat=aquats.cpu().detach().numpy().reshape(-1),
    density=adensities.cpu().detach().numpy().reshape(-1),
    feature=acolors.cpu().detach().numpy().reshape(-1),
)
key_order = ['mean', 'scale', 'quat', 'density', 'feature']

M = 20
N = 1
vjps = []
for i in range(M, M+N):
    args = (means, ascales, aquats, adensities, acolors, rayo_th[i:i+1], rayd_th[i:i+1])
    color, vjp = torch.autograd.functional.vjp(trace_rays, args, torch.ones((1, 3), device=device))
    (color, extras) = trace_rays(*args, return_extras=True)
    print(color)
    print(params, rayo[i:i+1].reshape(-1), rayd[i:i+1].reshape(-1), extras['tri_collection'].cpu().numpy())
    color2, bwd_diff = jax.vjp(jax_render.render, params, rayo[i:i+1].reshape(1, -1), rayd[i:i+1].reshape(1, -1), extras['tri_collection'].cpu().numpy().tolist())
    vjp2, drayo, drayd, _ = bwd_diff(jnp.ones((3)))
    print(vjp2)
    print(f"{bcolors.BOLD}COLOR: {bcolors.ENDC}{color.detach().cpu().numpy()}, {color2}")
    vjp2 = [torch.as_tensor(np.array(vjp2[k])) for k in key_order] + [drayo, drayd]
    vjps.append(vjp2)

args = (means, ascales, aquats, adensities, acolors, rayo_th[M:M+N], rayd_th[M:M+N])
out, total_vjp = torch.autograd.functional.vjp(trace_rays, args, torch.ones((N, 3), device=device))
print(total_vjp)
pred_total_vjp = []
for j in range(len(vjps[0])):
    s = sum([vjps[i][j] for i in range(N)])
    pred_total_vjp.append(s)
print(pred_total_vjp)
for i, k in enumerate(key_order):
    err = (pred_total_vjp[i].cpu() - total_vjp[i].cpu()).abs().max()
    print(f"{k} err: {err}")
