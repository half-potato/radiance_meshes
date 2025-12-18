import numpy as np
import torch
from utils import primitives
device = torch.device("cpu")
from util import *

# model = primitives.Primitives(
#     torch.zeros((1, 3)),
#     0.5*torch.ones((1, 3)),
#     torch.tensor([[0, 0, 0, 1]], dtype=float),
#     10*torch.ones((1)),
#     torch.tensor([[0.5, 0, 0.5]]),
# )

N = 2
np.random.seed(2)

# densities = Parameter(-torch.tensor(-0.0 + 1*np.random.rand(N), dtype=torch.float32, device=device))
densities = 3.25+torch.tensor(0.2*np.random.rand(N), dtype=torch.float32, device=device)
# densities = Parameter(np.exp(3.25)+torch.tensor(0.2*np.random.rand(N), dtype=torch.float32, device=device))
# densities = Parameter(-0.05+torch.tensor(0.2*np.random.rand(N), dtype=torch.float32, device=device))
means = torch.tensor(0.5 * (2*np.random.rand(N, 3) - 1) + np.array([0, 0, 0]), dtype=torch.float32, device=device)
# scales = Parameter((-1.5-1*torch.tensor(np.random.rand(N, 3), dtype=torch.float32, device=device)).exp())
scales = 0.5*torch.ones((N, 3), dtype=torch.float32, device=device)
quats = l2_normalize_th(2*torch.tensor(np.random.rand(N, 4), dtype=torch.float32, device=device)-1)
# quats[:, :3] = 0
# quats[:, 3] = 1
# quats = Parameter(l2_normalize_th(torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)))
feats = torch.zeros((N, 1, 3), dtype=torch.float32, device=device)
feats[:, 0:1, :] = torch.tensor(np.random.rand(N, 1, 3)*0.8+0.1, dtype=torch.float32, device=device)
model = primitives.Primitives(means, scales, quats, densities, feats)

model.save_ply("outputs/double.ply")
