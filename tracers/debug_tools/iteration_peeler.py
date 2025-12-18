from absl.testing import absltest
from absl.testing import parameterized
# from utils.test_utils import METHODS, SYM_METHODS, ALL_QUAD_PAIRS, QUAD_PAIRS
# from splinetracers import ellipsoid_splinetracer as method
from splinetracers import tetra_splinetracer as method
import numpy as np
import torch
from icecream import ic
from utils.math_util import l2_normalize_th
from splinetracers import quad

torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)

SH_C0 = 0.28209479177387814
N = 8
device = torch.device('cuda')
density_multi = 1

torch.manual_seed(1)
ic(torch.initial_seed())
np.random.seed(2)

state_names = [
  "distortion_parts_x",
  "distortion_parts_y",
  "cum_sum_x",
  "cum_sum_y",
  "padding_x",
  "padding_y",
  "padding_z",
  "t",
  "drgb_d",
  "drgb_r",
  "drgb_g",
  "drgb_b",
  "logT",
  "r",
  "g",
  "b",
]

rayo = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(device)
rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(device)

scale = 0.5*torch.tensor(
          np.random.rand(N, 3), dtype=torch.float32
).to(device)
# TODO REMOVE
scale = 1.0*torch.tensor(
          np.ones((N, 3)), dtype=torch.float32
).to(device)
mean = 2*torch.rand(N, 3, dtype=torch.float32).to(device)-1
# TODO REMOVE
mean[:, 0] *= 0.0
mean[:, 1] *= 0.0
mean[:, 0] += 0.01
mean[:, 1] += 0.01
mean[:, 2] += 1.5

quat = l2_normalize_th(2*torch.rand(N, 4, dtype=torch.float32).to(device)-1)
# TODO REMOVE
quat = torch.tensor([0, 0, 0, 1],
                    dtype=torch.float32, device=device).reshape(1, -1).expand(N, -1)
density = density_multi*torch.rand(N, 1, dtype=torch.float32).to(device)
features = torch.rand(N, 1, 3, dtype=torch.float32).to(device)
features[:, 0, :] = (1 - 0.5) / SH_C0

color, extras = method.trace_rays(
          mean, scale, quat, density, features, rayo, rayd,
          0, 3, return_extras=True)
M = extras['iters'].max()
print(M)
states = []
for i in range(M+1):
    color, extras = method.trace_rays(
              mean, scale, quat, density, features, rayo, rayd,
              0, 3, return_extras=True, max_iters=i)
    states.append(extras['saved'].states.reshape(-1))
states = torch.stack(states, dim=0)
for i, name in enumerate(state_names):
    print(f"{name}: {states[:, i]}")

q_color, q_extras = quad.trace_rays(
          mean, scale, quat, density, features, rayo, rayd,
          0, 3, return_extras=True, kernel=quad.query_tetra)
ic(color, q_color)
