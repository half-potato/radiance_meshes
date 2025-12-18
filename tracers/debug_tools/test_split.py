# from splinetracer import trace_rays
from const_splinetracer import trace_rays
import numpy as np
import math
from util import *
from torch.nn import Parameter
import torch
import imageio
from utils import primitives

np.random.seed(3)
#============================
# Camera SETUP
#============================
T = np.eye(4)
# Init rays
res = 800
# fov = 10 * math.pi / 180
fov = 30 * math.pi / 180
fx = res / 2 / math.tan(fov/2)
radius = 2
dx, dy = np.meshgrid(
    (np.linspace(-res/2, res/2, res) / fx),
    (np.linspace(-res/2, res/2, res) / fx),
)
crayd = l2_normalize(np.stack([ dx, dy, np.ones_like(dx)], axis=-1)).reshape(-1, 3)
shift = 0
rayo, rayd = get_rays(crayd, T)
rayo = rayo.reshape(res, res, 3) + shift
rayd = rayd.reshape(res, res, 3)

# rayo = rayo[res//2:res//2+1, res//2:res//2+1]
# rayd = rayd[res//2:res//2+1, res//2:res//2+1]
# res = 1

# rayo = rayo[297, 551]
# rayd = rayd[297, 551]
# res = 1

device = torch.device(0)
rayo_th = torch.as_tensor(rayo.reshape(-1, 3)).float().to(device)
rayd_th = torch.as_tensor(rayd.reshape(-1, 3)).float().to(device)

N = 1

scene_scale = 1.00
# densities = Parameter(-torch.tensor(-0.0 + 1*np.random.rand(N), dtype=torch.float32, device=device))
densities = 3.25+torch.tensor(0.2*np.random.rand(N), dtype=torch.float32, device=device)
# densities = Parameter(np.exp(3.25)+torch.tensor(0.2*np.random.rand(N), dtype=torch.float32, device=device))
# densities = Parameter(-0.05+torch.tensor(0.2*np.random.rand(N), dtype=torch.float32, device=device))
means = torch.tensor(0.5 * (2*np.random.rand(N, 3) - 1) + np.array([0.1, 0, 6]), dtype=torch.float32, device=device) * scene_scale + shift
# scales = Parameter((-1.5-1*torch.tensor(np.random.rand(N, 3), dtype=torch.float32, device=device)).exp())
scales = torch.ones((N, 3), dtype=torch.float32, device=device) * scene_scale
# quats = l2_normalize_th(2*torch.tensor(np.random.rand(N, 4), dtype=torch.float32, device=device)-1)
quats = torch.tensor([[0.0,0,0,1]], device=device)
# means = torch.nextafter(means, torch.inf*torch.ones_like(means))
# quats[:, :3] = 0
# quats[:, 3] = 1
# quats = Parameter(l2_normalize_th(torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)))
feats = torch.zeros((N, 1, 3), dtype=torch.float32, device=device)
feats[:, 0:1, :] = torch.tensor(np.ones((N, 1, 3)), dtype=torch.float32, device=device)

model = primitives.Primitives(means, scales, quats, densities, feats)

# i = 414
# j = 368
# s = 3
target_color = trace_rays(model.means, model.scales, model.quats, model.densities, model.features,
                          # rayo_th.reshape(res, res, 3)[414, 368].reshape(-1, 3),
                          # rayd_th.reshape(res, res, 3)[414, 368].reshape(-1, 3))
                          # rayo_th.reshape(res, res, 3)[i-s:i+s+1, j-s:j+s+1].reshape(-1, 3),
                          # rayd_th.reshape(res, res, 3)[i-s:i+s+1, j-s:j+s+1].reshape(-1, 3))
                          rayo_th,
                          rayd_th)
ind = target_color[:, 0].argmax()
print(ind // res, ind % res)
print(f"Max val: {target_color.max()} vs {1-(-(densities.reshape(-1, 1) * scales).max()).exp()}")
print(densities.reshape(-1, 1) * scales, densities.reshape(-1, 1))
print(scales)
def inv_opacity(y):
    x = (y / (1 - y).clip(min=1 / 255)).clip(min=1e-4)
    return x
print(inv_opacity(target_color.max()) / densities)

test_im = np.clip(target_color.reshape(res, res, 3).detach().cpu().numpy()*255, 0, 255).astype(np.uint8)
imageio.imwrite(f"target.png", test_im)
imageio.imwrite(f"target.exr", target_color.reshape(res, res, 3).detach().cpu())

# """

optim = torch.optim.Adam([
    dict(name="means", params=[model._means], lr=1), 
    dict(name="scales", params=[model._scales], lr=1), 
    dict(name="quats", params=[model._quats], lr=1), 
    dict(name="densities", params=[model._densities], lr=1), 
    dict(name="features", params=[model._features], lr=1),
    dict(name="base_color", params=[model._base_color], lr=1)
], lr=0)

model.split(optim, torch.ones((N), device=device, dtype=bool))
print(model.means.shape)
split_color = trace_rays(model.means, model.scales, model.quats, model.densities, model.features, rayo_th, rayd_th)
# model.save_ply("outputs/split.ply")
print((split_color*255).clip(min=0, max=255).byte().float() / 255)

split_im = np.clip(split_color.reshape(res, res, 3).detach().cpu().numpy()*255, 0, 255).astype(np.uint8)
imageio.imwrite(f"split.png", split_im)

# """
