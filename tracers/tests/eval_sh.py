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
from utils.test_utils import METHODS, SYM_METHODS, QUAD_PAIRS
import numpy as np
import torch
from icecream import ic
from utils.math_util import l2_normalize_th
import random
torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)

import eval_sh

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5


@torch.jit.script
def eval_sh_py(deg: int, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = 0.28209479177387814 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                0.4886025119029199 * y * sh[..., 1] +
                0.4886025119029199 * z * sh[..., 2] -
                0.4886025119029199 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    1.0925484305920792 * xy * sh[..., 4] +
                    -1.0925484305920792 * yz * sh[..., 5] +
                    0.31539156525252005 * (2.0 * zz - xx - yy) * sh[..., 6] +
                    -1.0925484305920792 * xz * sh[..., 7] +
                    0.5462742152960396 * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                -0.5900435899266435 * y * (3 * xx - yy) * sh[..., 9] +
                2.890611442640554 * xy * z * sh[..., 10] +
                -0.4570457994644658 * y * (4 * zz - xx - yy)* sh[..., 11] +
                0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                -0.4570457994644658 * x * (4 * zz - xx - yy) * sh[..., 13] +
                1.445305721320277 * z * (xx - yy) * sh[..., 14] +
                -0.5900435899266435 * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + 2.5033429417967046 * xy * (xx - yy) * sh[..., 16] +
                            -1.7701307697799304 * yz * (3 * xx - yy) * sh[..., 17] +
                            0.9461746957575601 * xy * (7 * zz - 1) * sh[..., 18] +
                            -0.6690465435572892 * yz * (7 * zz - 3) * sh[..., 19] +
                            0.10578554691520431 * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            -0.6690465435572892 * xz * (7 * zz - 3) * sh[..., 21] +
                            0.47308734787878004 * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            -1.7701307697799304 * xz * (xx - 3 * yy) * sh[..., 23] +
                            0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

device = torch.device('cuda')
class SHGradCheckTest(parameterized.TestCase):

    @parameterized.product(
        N = [1],
        sh_degree = [0, 1],#, 2, 3],
        # sh_degree = [0],
    )
    def test_deriv_sph(self, N, sh_degree):
        features = torch.rand((N, (sh_degree+1)**2, 3), device=device)
        means = l2_normalize_th(2*torch.rand((N, 3), device=device)-1)

        rayo = torch.tensor([[0, 0, -2]], dtype=torch.float32).to(device)
        rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(device)

        features = torch.nn.Parameter(features)
        
        def l2_loss(features):
            return eval_sh.eval_sh(means, features, rayo, sh_degree)
        def l2_loss_th(features):
            shs_view = features.transpose(1, 2).view(
                -1, 3, (sh_degree + 1) ** 2
            )
            dir_pp = means - rayo.repeat(
                N, 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            rgb1 = (eval_sh_py(sh_degree, shs_view, dir_pp_normalized) + 0.5).clip(min=0)
            return rgb1
        shs_view = features.transpose(1, 2).view(
            -1, 3, (sh_degree + 1) ** 2
        )
        dir_pp = means - rayo.repeat(
            N, 1
        )
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        rgb1 = (eval_sh_py(sh_degree, shs_view, dir_pp_normalized) + 0.5).clip(min=0)
        for i in range(3):
            grad = torch.nn.functional.one_hot(torch.tensor([i]), 3).to(device)
            out, total_vjp = torch.autograd.functional.vjp(l2_loss, (features), grad) 
            out_th, total_vjp_th = torch.autograd.functional.vjp(l2_loss_th, (features), grad)
            np.testing.assert_allclose(total_vjp.cpu().numpy(), total_vjp_th.cpu().numpy(), atol=1e-6, rtol=1e-6)
        torch.autograd.gradcheck(l2_loss, (features), eps=1e-4, atol=1e-2)

    # @parameterized.product(
    #     N = [1, 1, 1, 1, 5],
    #     sh_degree = [0, 1, 2, 3],
    # )
    # def test_matches_python(self, N, sh_degree):
    #     features = torch.rand((N, (sh_degree+1)**2, 3), device=device)
    #     means = l2_normalize_th(2*torch.rand((N, 3), device=device)-1)
    #
    #     rayo = torch.tensor([[0, 0, -2]], dtype=torch.float32).to(device)
    #     rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(device)
    #
    #     shs_view = features.transpose(1, 2).view(
    #         -1, 3, (sh_degree + 1) ** 2
    #     )
    #     dir_pp = means - rayo.repeat(
    #         N, 1
    #     )
    #     dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    #     rgb1 = (eval_sh_py(sh_degree, shs_view, dir_pp_normalized) + 0.5).clip(min=0)
    #
    #     rgb2 = eval_sh.eval_sh(means, features, rayo, sh_degree)
    #     np.testing.assert_allclose(rgb1.cpu().numpy(), rgb2.cpu().numpy(), atol=1e-5)




if __name__ == "__main__":
    absltest.main()
