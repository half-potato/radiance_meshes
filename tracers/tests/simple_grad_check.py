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
from utils.test_utils import METHODS, SYM_METHODS
# from gaussian_splinetracer import trace_rays
# from splinetracers.ellipsoid_splinetracer import trace_rays
# from gaussian_alphacomp import trace_rays
# from ellipsoid_alphacomp import trace_rays
import torch
import numpy as np
from icecream import ic


SH_C0 = 0.28209479177387814
N = 3
device = torch.device("cuda")

def l2_normalize_th(x, eps=torch.finfo(torch.float32).eps):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(
        torch.clip(torch.sum(x**2, dim=-1, keepdim=True), eps, None)
    )

@torch.jit.script
def inv_opacity(y):
    x = (-(1 - y).clip(min=1e-10).log()).clip(min=0)
    return x

class SimpleTest(parameterized.TestCase):

    @parameterized.parameters(SYM_METHODS)
    def test_grad(self, method):
        scene_scale = 1.00
        # opacities = Parameter(-torch.tensor(-0.0 + 1*np.random.rand(N), dtype=torch.float32, device=device))
        # opacities = 0.005+torch.tensor(0.0*np.ones((N)), dtype=torch.float32, device=device)
        # opacities = torch.tensor([0.1, 10, 0.1], dtype=torch.float32, device=device)
        opacities = torch.tensor([0.1, 0.9, 0.1], dtype=torch.float32, device=device)
        ic(opacities)
        # opacities = Parameter(np.exp(3.25)+torch.tensor(0.2*np.random.rand(N), dtype=torch.float32, device=device))
        # opacities = Parameter(-0.05+torch.tensor(0.2*np.random.rand(N), dtype=torch.float32, device=device))
        # means = torch.tensor(np.array([[0.0, 0.0, 6], [0.0, 0.0, 5.1]]), dtype=torch.float32, device=device)
        means = torch.tensor(np.array(
            [[0.0, 0.0, 40.0],
             [0.0, 0.0, 25.0],
             [0.0, 0.0, 3.2]]), dtype=torch.float32, device=device)
        # means = torch.tensor(np.array([[0.0, 0.89, 6]]), dtype=torch.float32, device=device)
        # scales = (-1.5-1*torch.tensor(np.random.rand(N, 3), dtype=torch.float32, device=device)).exp()
        scales = 1.00*torch.ones((N, 3), dtype=torch.float32, device=device)
        quats = l2_normalize_th(2*torch.tensor(np.random.rand(N, 4), dtype=torch.float32, device=device)-1)
        # quats = torch.tensor([[0.0,0,0,1]], device=device).expand(N, -1)
        # means = torch.nextafter(means, torch.inf*torch.ones_like(means))
        # quats[:, :3] = 0
        # quats[:, 3] = 1
        # quats = Parameter(l2_normalize_th(torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)))
        feats = torch.zeros((N, 1, 3), dtype=torch.float32, device=device)
        # feats[:, 0:1, :] = torch.tensor(np.ones((N, 1, 3)), dtype=torch.float32, device=device)
        a = torch.tensor(np.eye(3), dtype=torch.float32, device=device)
        feats[:, 0, :] = (a - 0.5) / SH_C0
        # feats[:, 0, :] = a

        means = torch.nn.Parameter(means)
        scales = torch.nn.Parameter(scales)
        quats = torch.nn.Parameter(quats)
        feats = torch.nn.Parameter(feats)
        opacities = torch.nn.Parameter(opacities)

        input_densities = inv_opacity(opacities) / 2 if method.trace_rays.uses_density else opacities
        input_densities.retain_grad()
        rayo = torch.tensor([
            [0.0, 0.0, 0],
        ], dtype=torch.float32, device=device)
        rayd = torch.tensor([
            [0.0, 0, 1],
        ], dtype=torch.float32, device=device)

        target_color = method.trace_rays(
            means, scales, quats,
            input_densities,
            feats,
            rayo,
            rayd)
        ic(target_color)

        loss = target_color[..., :3].sum()
        loss.backward()
        # ic(means.grad)
        # ic(quats.grad)
        # ic(scales.grad)
        # ic(feats.grad)
        # ic(opacities.grad)
        # ic(input_densities.grad)
        # self.assertTrue(torch.allclose(opacities.grad,
        #                       torch.tensor([0.0900, 0.8100, 0.0900], device='cuda:0'), 1e-4))
        feat_grad_gt = torch.diag(
            torch.tensor([0.0025, 0.2285, 0.0282], device='cuda:0')).reshape(feats.grad.shape)
        # self.assertTrue(torch.allclose(feats.grad, feat_grad_gt, atol=1e-4))
        torch.testing.assert_allclose(
            opacities.grad,
            torch.tensor([0.0900, 0.8100, 0.0900], device='cuda:0'),
            rtol=1e-5, atol=1e-4)
        torch.testing.assert_allclose(feats.grad, feat_grad_gt, rtol=1e-5, atol=1e-4)

if __name__ == "__main__":
    absltest.main()
