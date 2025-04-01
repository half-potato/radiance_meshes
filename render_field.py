import os
VERSION = 9
if VERSION is not None:
    os.environ["CC"] = f"/usr/bin/gcc-{VERSION}"
    os.environ["CXX"] = f"/usr/bin/g++-{VERSION}"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
import sys
sys.path.append(str(Path(os.path.abspath('')).parent))
print(str(Path(os.path.abspath('')).parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import topo_utils

from utils.contraction import contract_mean_std
from gDel3D.build.gdel3d import Del
import plotly.graph_objects as go
import numpy as np
from utils.compare_quad import setup_camera
from utils.train_util import render
from data.camera import Camera
from models.ingp_color import pre_calc_cell_values, compute_vertex_colors_from_field, calculate_circumcenters_torch
from utils.graphics_utils import l2_normalize_th
import mediapy
from scipy.spatial import Delaunay
import math

torch.set_printoptions(precision=10)

tile_size = 16
height = 1000
width = 1000
fov = 90
viewmat = torch.eye(4)

viewmat, projection_matrix, cam_pos, fovy, fovx, fx, fy = setup_camera(height, width, fov, viewmat)
# Now extract R,T from viewmat
# If viewmat is truly "World->View", then R is top-left 3x3, T is top-right 3x1
# V = torch.inverse(viewmat)
V = viewmat
R = V[:3, :3].T
T = V[:3, 3]

# Create a blank image for the camera
blank_image = torch.zeros((3, height, width), device="cuda")

# Instantiate the camera
camera = Camera(
    colmap_id = 0,
    R = R.cpu().numpy(),
    T = T.cpu().numpy(),
    fovx = fovx,
    fovy = fovy,
    image = blank_image,
    gt_alpha_mask = None,
    uid = 0,
    cx = -1,
    cy = -1,
    trans = np.array([0.0, 0.0, 0.0]), # or any translation offset you need
    scale = 1.0,
    data_device = "cuda",
    # You can add any extra distortions, exposure, etc. you might need
)

def field(xyz, scale=10.5):
    # Unpack coordinates
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    
    # Base color intercepts
    c0 = 0.3*torch.sin(scale * x)
    c1 = 0.3*torch.cos(scale * y)
    c2 = 0.3*torch.sin(scale * z)
    
    # Gradients: 9 coefficients (to be reshaped as (3,3) per point if needed)
    c3  = torch.cos(scale * x)
    c4  = torch.sin(scale * (x + y))
    c5  = torch.cos(scale * (x + z))
    c6  = torch.sin(scale * (y - x))
    c7  = torch.cos(scale * y)
    c8  = torch.sin(scale * (y + z))
    c9  = torch.cos(scale * (z - x))
    c10 = torch.sin(scale * (z - y))
    c11 = torch.cos(scale * z)
    
    # Extra coefficient
    c12 = torch.sin(scale * (x * y * z))
    
    # Stack all coefficients along the last dimension
    return torch.stack([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12], dim=-1)

def project_points_to_tetrahedra(points, tets):
    """
    Projects each point in `points` (shape (N, 3)) onto the corresponding tetrahedron in `tets` (shape (N, 4, 3))
    by clamping negative barycentrics to zero and renormalizing them so that they sum to 1.
    
    The barycentrics for a tetrahedron with vertices v0, v1, v2, v3 are computed as:
      w0 = 1 - (x0+x1+x2)
      w1, w2, w3 = x0, x1, x2, where x solves T x = (p - v0) with T = [v1-v0, v2-v0, v3-v0]
    """
    N = points.shape[0]
    v0 = tets[:, 0, :]             # shape (N, 3)
    T = tets[:, 1:, :] - v0.unsqueeze(1)  # shape (N, 3, 3)
    
    # Solve for x: T x = (p - v0)
    p_minus_v0 = points - v0       # shape (N, 3)
    x = torch.linalg.solve(T, p_minus_v0.unsqueeze(2)).squeeze(2)  # shape (N, 3)
    
    # Compute full barycentrics: weight for v0 and for v1,v2,v3.
    w0 = 1 - x.sum(dim=1, keepdim=True)  # shape (N, 1)
    bary = torch.cat([w0, x], dim=1)      # shape (N, 4)
    
    # Clamp negative values and renormalize to sum to 1.
    # bary = torch.clamp(bary, min=0, max=1)
    # norm = (bary.sum(dim=1, keepdim=True)).clip(min=1e-8)
    # mask = (norm > 1).reshape(-1)
    # bary[mask] = bary[mask] / norm[mask]
    
    # Reconstruct the point as the weighted sum of the tetrahedron vertices.
    p_proj = (T * bary[:, 1:].unsqueeze(1)).sum(dim=2) + v0
    return p_proj

M= 200
N = 800

### Create motion:
# Save initial positions as centers
centers = 2 * torch.rand((N, 3), device='cuda') - 1
centers[:, 2] += 5

# Choose random radii for circles (for instance, between 0.1 and 0.5)
radii = 0.1 + 0.4 * torch.rand((N,), device='cuda')

# Pick a random normal for each vertex (which defines the plane of its circle)
normals = torch.randn((N, 3), device='cuda')
normals = normals / normals.norm(dim=1, keepdim=True)  # normalize to unit vectors

# Create an arbitrary vector that is not parallel to the normal.
# Start with (1,0,0) for all vertices.
arbitrary = torch.tensor([1, 0, 0], device='cuda', dtype=torch.float32).expand(N, 3).contiguous()

# If any normal is too close to (1,0,0), replace with (0,1,0)
dot = (normals * arbitrary).sum(dim=1)
mask = dot.abs() > 0.99
if mask.any():
    arbitrary[mask] = torch.tensor([0, 1, 0], device='cuda', dtype=torch.float32).reshape(1, 3)

# Construct two orthonormal vectors in the plane:
e1 = torch.cross(normals, arbitrary)
e1 = e1 / e1.norm(dim=1, keepdim=True)
e2 = torch.cross(normals, e1)

# vertices = 2*torch.rand((N, 3), device='cuda')-1
# vertices[:, 2] += 3
# velocity = l2_normalize_th(torch.randn((N, 3), device='cuda'))

frames = []

for i in range(M):
    theta = 2 * math.pi * i / M  # complete circle by frame M
    # Compute the offset for each vertex along its circle
    offset = radii.unsqueeze(1) * (math.cos(theta) * e1 + math.sin(theta) * e2)
    # The new positions are the centers plus the circular offset
    vertices = centers + offset
    # vertices += velocity * 0.005

    model = lambda x: x
    model.vertices = vertices

    # Convert to tensor and move to CUDA
    points_cpu = vertices.detach().cpu().numpy()

    delaunay = Delaunay(points_cpu)
    indices = torch.tensor(delaunay.simplices, device=vertices.device).int()

    # v = Del(vertices.shape[0])
    # indices_np, prev = v.compute(vertices.detach().cpu())
    # indices_np = indices_np.numpy()
    # indices_np = indices_np[(indices_np < vertices.shape[0]).all(axis=1)]
    # indices = torch.as_tensor(indices_np).cuda()
    
    model.vertex_color = torch.ones((1), device='cuda')

    tets = vertices[indices]
    circumcenter, radius = calculate_circumcenters_torch(tets.double())
    # clipped_circumcenter = project_points_to_tetrahedra(circumcenter.float(), tets)
    clipped_circumcenter = circumcenter.clone()
    max_norm = torch.linalg.norm(vertices, dim=1).max()
    norm = torch.linalg.norm(circumcenter, dim=-1, keepdim=True)
    mask = (norm > max_norm)[:, 0]
    clipped_circumcenter[mask] = clipped_circumcenter[mask] / norm[mask] * max_norm
    output = field(clipped_circumcenter)
    # cv, cr = contract_mean_std(circumcenter, radius)
    # output = field(cv)
    density = (output[:, 0:1]-1).exp()
    field_samples = output[:, 1:]
    vcolors = compute_vertex_colors_from_field(
        vertices[indices].detach(), field_samples.float(), circumcenter.float().detach())
    vcolors = torch.nn.functional.softplus(vcolors, beta=10)
    vcolors = vcolors.reshape(-1, 12)
    features = torch.cat([
        density, vcolors], dim=1)
    model.tet_density = features.float()
    
    model.indices = indices
    model.scene_scaling = 1
    def get_cell_values(camera, mask=None):
        if mask is not None:
            return model.vertex_color, model.tet_density[mask]
        else:
            return model.vertex_color, model.tet_density
    model.get_cell_values = get_cell_values
    
    render_pkg = render(camera, model, tile_size=tile_size, min_t=0, ladder_p=1, pre_multi=1)
    torch_image = render_pkg['render'].permute(1, 2, 0)
    image = (torch_image.detach().cpu().numpy() * 255).clip(min=0, max=255).astype(np.uint8)
    frames.append(image)

mediapy.write_video('frames.mp4', frames)

