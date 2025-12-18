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

N = 1
sh_degree = 1
device = torch.device('cuda')


features = torch.rand((N, (sh_degree+1)**2, 3), device=device)
means = l2_normalize_th(2*torch.rand((N, 3), device=device)-1)

rayo = torch.tensor([[0, 0, -2]], dtype=torch.float32).to(device)
rayd = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(device)

features = torch.nn.Parameter(features)

def l2_loss(features):
    return eval_sh.eval_sh(means, features, rayo, rayd, sh_degree).sum()

# ic(eval_sh.eval_sh(means, features, rayo, rayd, sh_degree))
# features[:, 1] += 1
# ic(eval_sh.eval_sh(means, features, rayo, rayd, sh_degree))
torch.autograd.gradcheck(l2_loss, (features), eps=1e-4, atol=1e-2)
# loss = l2_loss(features)
# loss.backward()

