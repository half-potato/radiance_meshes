import os
from pathlib import Path
import sys
sys.path.append(str(Path(os.path.abspath('')).parent))
print(str(Path(os.path.abspath('')).parent))
import numpy as np
import time
import torch
from gDel3D.build.gdel3d import Del
from scipy.spatial import Voronoi, Delaunay
from icecream import ic

# Generate random 3D points
num_points = 100
torch.manual_seed(189710236)
vertices = torch.rand((num_points, 3), device='cuda')

# Initialize Delaunay triangulation
v = Del(vertices.shape[0])
st = time.time()
oindices_np, prev = v.compute(vertices.detach().cpu())
prev.check_correctness(vertices.detach().cpu())

oindices_np = oindices_np.numpy()
indices_np = oindices_np[(oindices_np < vertices.shape[0]).all(axis=1)]
indices = torch.as_tensor(indices_np.astype(np.int32)).cuda()
print(f'Initial Delaunay computation time: {1 / (time.time() - st):.4f} Hz')
ic(indices)
# d = Delaunay(vertices.cpu().numpy())
# print(d.simplices.shape)
d = torch.randn_like(vertices)
del v

for i in range(1):
    # Perturb vertices slightly
    vertices += (d * 0.1)  # Small perturbation

    # Recompute Delaunay using previous indices
    st = time.time()
    v = Del(vertices.shape[0])
    oindices_np, prev = v.compute_from_prev(vertices.detach().cpu(), prev)
    oindices_np = oindices_np.numpy()
    indices_np = oindices_np[(oindices_np < vertices.shape[0]).all(axis=1)]
    indices_np = np.sort(indices_np, axis=1)
    indices_np = indices_np[np.lexsort(indices_np.T)].astype(np.int32)
    indices = torch.as_tensor(indices_np.astype(np.int32)).cuda()
    ic(indices, indices.shape)
    prev.check_correctness(vertices.detach().cpu())
    print(f'Perturbed Delaunay computation time: {1 / (time.time() - st):.4f} Hz')

