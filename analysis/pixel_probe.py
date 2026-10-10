"""Per-pixel FD vs torch-backward density gradient probe (iter-3 scene)."""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from tests.multi_tet_test import generate_point_cloud, create_view_matrix  # noqa: E402
from utils.test_util import compute_delaunay, generate_color_palette  # noqa: E402
from utils.topo_utils import calculate_circumcenters_torch  # noqa: E402
from utils.model_util import offset_normalize  # noqa: E402
from utils.compare_quad import setup_camera  # noqa: E402
from utils.train_util import render as trender  # noqa: E402
from data.camera import Camera  # noqa: E402

torch.manual_seed(189710234)
RADIUS, OFFSET, NPTS = 10, 1, 20
for i in range(4):
    points = generate_point_cloud(NPTS, RADIUS)
    direction = torch.randn(3, device=points.device)
    direction = direction / torch.norm(direction)
    camera_pos = direction * (RADIUS + OFFSET)
    center = torch.mean(points, dim=0)
    viewmat = create_view_matrix(camera_pos, center)

indices = compute_delaunay(points)
N = indices.shape[0]
allc = generate_color_palette(N * 2)[:, :3]
base_colors = allc[:N].reshape(-1, 3)
raw_grd = (2 * allc[N:].reshape(-1, 3) - 1) / math.sqrt(3)
tets = points[indices]
cc, _ = calculate_circumcenters_torch(tets)
nc, ng = offset_normalize(base_colors, raw_grd, cc, tets)
J = 1

vm, K, cam_pos2, fovy, fovx, fx, fy = setup_camera(16, 16, 90, viewmat)
camera = Camera(colmap_id=0, R=vm[:3, :3].T.cpu().numpy(), T=vm[:3, 3].cpu().numpy(),
                fovx=fovx, fovy=fovy, image=torch.zeros((3, 16, 16), device="cuda"),
                uid=0, cx=-1, cy=-1, data_device="cuda")


class M:
    pass


m = M()
m.vertices = points
m.indices = indices
m.vertex_normals = torch.zeros_like(points)
m.additional_attr = 0
m.device = "cuda"
m.mask_values = False
m.feature_dim = 7


def torch_img(dv):
    td = torch.ones((N, 1), device="cuda")
    td[J, 0] = dv
    cv = torch.cat([td, nc, ng.reshape(-1, 3)], dim=1)
    with torch.no_grad():
        pkg = trender(camera, m, cell_values=cv, tile_size=8, min_t=0.01)
    return pkg["render"].sum(0)  # [H, W] rgb sum


eps = 3e-3
fd_px = (torch_img(1.0 + eps) - torch_img(1.0 - eps)) / (2 * eps)

order = fd_px.abs().ravel().argsort(descending=True)[:6]
for flat in order.tolist():
    py, px = divmod(flat, 16)
    td = torch.ones((N, 1), device="cuda")
    cv = torch.cat([td, nc, ng.reshape(-1, 3)], dim=1).detach().requires_grad_(True)
    pkg = trender(camera, m, cell_values=cv, tile_size=8, min_t=0.01)
    pkg["render"][:, py, px].sum().backward()
    print(f"pix ({py:2d},{px:2d}): FD={fd_px[py, px].item():9.5f}  torch_bwd={cv.grad[J, 0].item():9.5f}")
