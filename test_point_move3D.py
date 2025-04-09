import torch
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.spatial import Delaunay
import math
from utils import topo_utils
from utils.contraction import contraction_jacobian, contract_points
from gDel3D.build.gdel3d import Del

# Suppose you have computed frames of clipped circumcenters stored in a list
# For illustration, we create dummy data.
M = 500  # total frames
N = 100  # number of points per frame
device = 'cpu'

def calculate_circumcenters_torch(vertices: torch.Tensor):
    """
    Compute the circumcenter and circumradius of a tetrahedron using PyTorch.
    
    Args:
        vertices: Tensor of shape (..., 4, 3) containing the vertices of the tetrahedron(s).
    
    Returns:
        circumcenter: Tensor of shape (..., 3) containing the circumcenter coordinates.
        radius: Tensor of shape (...) containing the circumradius.
    """
    # Compute vectors from v0 to other vertices
    a = vertices[..., 1, :] - vertices[..., 0, :]  # v1 - v0
    b = vertices[..., 2, :] - vertices[..., 0, :]  # v2 - v0
    c = vertices[..., 3, :] - vertices[..., 0, :]  # v3 - v0

    # Compute squared lengths
    aa = torch.sum(a * a, dim=-1, keepdim=True)  # |a|^2
    bb = torch.sum(b * b, dim=-1, keepdim=True)  # |b|^2
    cc = torch.sum(c * c, dim=-1, keepdim=True)  # |c|^2

    # Compute cross products
    cross_bc = torch.cross(b, c, dim=-1)
    cross_ca = torch.cross(c, a, dim=-1)
    cross_ab = torch.cross(a, b, dim=-1)

    # Compute denominator
    denominator = 2.0 * torch.sum(a * cross_bc, dim=-1, keepdim=True) + 1e-3

    # Create mask for small denominators
    mask = torch.abs(denominator) < 1e-6

    # Compute circumcenter relative to verts[0]
    relative_circumcenter = (
        aa * cross_bc +
        bb * cross_ca +
        cc * cross_ab
    ) / torch.where(mask, torch.ones_like(denominator), denominator)

    # Compute circumradius
    radius = torch.norm(a - relative_circumcenter, dim=-1)

    # Return absolute position
    return vertices[..., 0, :] + relative_circumcenter, radius

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
    T = T.permute(0,2,1)

    # Solve for x: T x = (p - v0)
    p_minus_v0 = points - v0       # shape (N, 3)
    x = torch.linalg.solve(T, p_minus_v0.unsqueeze(2)).squeeze(2)  # shape (N, 3)

    # Compute full barycentrics: weight for v0 and for v1,v2,v3.
    w0 = 1 - x.sum(dim=1, keepdim=True)  # shape (N, 1)
    bary = torch.cat([w0, x], dim=1)      # shape (N, 4)
    bary = bary.clip(min=0)

    norm = (bary.sum(dim=1, keepdim=True)).clip(min=1e-8)
    # Clamp negative values and renormalize to sum to 1.
    bary = torch.clamp(bary, min=0)
    mask = (norm > 1).reshape(-1)
    bary[mask] = bary[mask] / norm[mask]

    # Reconstruct the point as the weighted sum of the tetrahedron vertices.
    p_proj = (T * bary[:, 1:].unsqueeze(1)).sum(dim=2) + v0
    return p_proj

### Create motion:
# Save initial positions as centers
S = 5
centers = 2*S * torch.randn((N, 3), device='cpu')
# centers[:, 2] += 5

# vel = torch.randn((N, 3))
# vel = torch.linalg.norm(vel, dim=1, keepdim=True)

# Choose random radii for circles (for instance, between 0.1 and 0.5)
radii = S * (0.1 + 0.4 * torch.rand((N,), device=device))

# Pick a random normal for each vertex (which defines the plane of its circle)
normals = torch.randn((N, 3), device=device)
normals = normals / normals.norm(dim=1, keepdim=True)  # normalize to unit vectors

# Create an arbitrary vector that is not parallel to the normal.
# Start with (1,0,0) for all vertices.
arbitrary = torch.tensor([1, 0, 0], device=device, dtype=torch.float32).expand(N, 3).contiguous()

# If any normal is too close to (1,0,0), replace with (0,1,0)
dot = (normals * arbitrary).sum(dim=1)
mask = dot.abs() > 0.99
if mask.any():
    arbitrary[mask] = torch.tensor([0, 1, 0], device=device, dtype=torch.float32).reshape(1, 3)

# Construct two orthonormal vectors in the plane:
e1 = torch.cross(normals, arbitrary)
e1 = e1 / e1.norm(dim=1, keepdim=True)
e2 = torch.cross(normals, e1)

frames = []
theta = 0

offset = radii.unsqueeze(1) * (math.cos(theta) * e1 + math.sin(theta) * e2)

vertices = centers
thetas = torch.zeros((N, 1), device=device)

points_cpu = vertices.detach().cpu()
v = Del(points_cpu.shape[0])
indices, prev = v.compute(points_cpu)
mask = (indices < points_cpu.shape[0]).all(axis=1)
indices = torch.as_tensor(indices[mask])
border = prev.get_boundary_tets(points_cpu.double())
border_mask = torch.zeros_like(torch.as_tensor(mask))
border_mask[border] = True
border_mask = border_mask[mask]

for i in range(M):

    tets = vertices[indices]
    circumcenter, radius = calculate_circumcenters_torch(tets.double())
    # clipped_circumcenter = project_points_to_tetrahedra(circumcenter.float(), tets)
    clipped_circumcenter = circumcenter
    mask = torch.linalg.norm(vertices, dim=1) < 1
    _, sensitivity = topo_utils.compute_vertex_sensitivity(indices.cuda(), vertices.cuda(), clipped_circumcenter.cuda(), border_mask)
    scaling = 5e-3/sensitivity.cpu().reshape(-1, 1).clip(min=1)/ radii.reshape(-1, 1).cpu()

    thetas += 2 * math.pi * scaling
    # Compute the offset for each vertex along its circle
    offset = radii.unsqueeze(1) * (torch.cos(thetas) * e1 + torch.sin(thetas) * e2)
    vertices = centers + offset


    model = lambda x: x
    model.vertices = vertices

    # tets = vertices[indices]
    # circumcenter, radius = calculate_circumcenters_torch(tets.double())
    cc = contract_points(clipped_circumcenter)
    # cc = circumcenter
    # frames.append(circumcenter)
    frames.append(cc)
    points_cpu = vertices.detach().cpu()
    # delaunay = Delaunay(points_cpu)
    # indices = torch.tensor(delaunay.simplices, device=vertices.device).int()
    v = Del(points_cpu.shape[0])
    indices, prev = v.compute(points_cpu.double())
    mask = (indices < points_cpu.shape[0]).all(axis=1)
    indices = indices[mask]
    border = prev.get_boundary_tets(points_cpu)
    border_mask = torch.zeros_like(mask)
    border_mask[border] = True
    border_mask = border_mask[mask]

# Convert each frame to numpy arrays for plotting
frames_np = [frame.cpu().numpy() for frame in frames]

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter([], [], [], s=5)  # initial empty scatter

# Set axis limits (adjust as needed)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

def update(frame_num):
    data = frames_np[frame_num]
    # Update scatter plot data; need to update the _offsets3d attribute for 3D scatter
    scat._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
    ax.set_title(f"Frame {frame_num+1}/{M}")
    return scat,

# Create the animation
ani = FuncAnimation(fig, update, frames=M, interval=1, blit=False)

# Save the animation to a file (requires ffmpeg)
ani.save('clipped_circumcenters.mp4', writer='ffmpeg', fps=24)
