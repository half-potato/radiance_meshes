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
from absl.testing import absltest
from absl.testing import parameterized
from utils.test_utils import METHODS, SYM_METHODS, ALL_QUAD_PAIRS, QUAD_PAIRS
import numpy as np
import torch
from icecream import ic
from utils.math_util import l2_normalize_th
torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)
from splinetracers import quad
import random

device = torch.device('cuda')

class QuadratureTest(parameterized.TestCase):
    # @parameterized.parameters(QUAD_PAIRS)
    @parameterized.product(
        method_kernel=ALL_QUAD_PAIRS,
        N = [1, 5, 10, 20, 40],
        density_multi = [0.01, 0.1, 1],
    )
    def test_ray_origin_outside(self, method_kernel, N, density_multi):
        method, kernel = method_kernel
        rayo = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(device)
        rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(device)

        scale = 0.5*torch.tensor(
                  np.random.rand(N, 3), dtype=torch.float32
        ).to(device)
        mean = torch.rand(N, 3, dtype=torch.float32).to(device)
        mean[:, 0] *= 0.5
        mean[:, 1] *= 0.5
        mean[:, 2] += 0.5

        N = scale.shape[0]

        quat = l2_normalize_th(2*torch.rand(N, 4, dtype=torch.float32).to(device)-1)
        # quat2 = torch.tensor([0, 0, 0, 1],
        #                     dtype=torch.float32, device=device).reshape(1, -1).expand(N, -1)
        density = density_multi*torch.rand(N, 1, dtype=torch.float32).to(device)
        features = torch.rand(N, 1, 3, dtype=torch.float32).to(device)

        color1, extras1 = quad.trace_rays(
                  mean, scale, quat, density, features, rayo, rayd,
                  0, 3, return_extras=True, kernel=kernel)
        color1 = color1[:, :4].reshape(-1)

        color2, extras2 = method.trace_rays(
                  mean, scale, quat, density, features, rayo, rayd,
                  0, 100, return_extras=True)

        color2 = color2[:, :4].reshape(-1)

        np.testing.assert_allclose(np.array(color1), color2.cpu().numpy(), atol=1e-4, rtol=1e-4)

    @parameterized.product(
        method_kernel=QUAD_PAIRS,
        N = [1, 5, 10, 20, 40],
        density_multi = [0.01, 0.1, 1],
    )
    def test_ray_origin_inside(self, method_kernel, N, density_multi):
        method, kernel = method_kernel
        rayo = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(device)
        rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(device)

        scale = 0.5*torch.tensor(
                  np.random.rand(N, 3), dtype=torch.float32
        ).to(device)
        mean = 1.2*torch.rand(N, 3, dtype=torch.float32).to(device)-0.2
        mean[:, 0] *= 0.5
        mean[:, 1] *= 0.5

        N = scale.shape[0]

        quat = l2_normalize_th(2*torch.rand(N, 4, dtype=torch.float32).to(device)-1)
        # quat2 = torch.tensor([0, 0, 0, 1],
        #                     dtype=torch.float32, device=device).reshape(1, -1).expand(N, -1)
        density = density_multi*torch.rand(N, 1, dtype=torch.float32).to(device)
        features = torch.rand(N, 1, 3, dtype=torch.float32).to(device)

        tmin = random.random()*0.3
        color1, extras1 = quad.trace_rays(
                  mean, scale, quat, density, features, rayo, rayd,
                  tmin, 3, return_extras=True, kernel=kernel)
        color1 = color1[:, :4].reshape(-1)

        color2, extras2 = method.trace_rays(
                  mean, scale, quat, density, features, rayo, rayd,
                  tmin, 100, return_extras=True)

        color2 = color2[:, :4].reshape(-1)

        np.testing.assert_allclose(np.array(color1), color2.cpu().numpy(), atol=1e-4, rtol=1e-4)

if __name__ == "__main__":
    absltest.main()
