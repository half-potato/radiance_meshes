import numpy as np
from scipy.spatial import KDTree, Delaunay
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import time
import torch
from torch.autograd.functional import jacobian
from icecream import ic
from submodules.spectral_norm3 import compute_spectral_norm3
from utils.contraction import contract_points, contraction_jacobian, contraction_jacobian_d_in_chunks
import math
from scipy.spatial import ConvexHull


@torch.jit.script
def calc_barycentric(points, tets):
    """
    points: (N, 3)
    tets: (N, 4, 3)
    """
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
    return bary

@torch.jit.script
def project_points_to_tetrahedra(points, tets):
    """
    points: (N, 3)
    tets: (N, 4, 3)
    """
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

def calculate_circumcenters(vertices):
    """
    Compute the circumcenter of a tetrahedron.
    
    Args:
        vertices: Tensor of shape (..., 4, 3) containing the vertices of the tetrahedron(a).
                 The first dimension can be batched.
    
    Returns:
        circumcenter: Tensor of shape (..., 3) containing the circumcenter coordinates
    """
    # Compute vectors from v0 to other vertices
    a = vertices[..., 1, :] - vertices[..., 0, :]  # v1 - v0
    b = vertices[..., 2, :] - vertices[..., 0, :]  # v2 - v0
    c = vertices[..., 3, :] - vertices[..., 0, :]  # v3 - v0
    
    # Compute squares of lengths
    aa = np.sum(a * a, axis=-1, keepdims=True)  # |a|^2
    bb = np.sum(b * b, axis=-1, keepdims=True)  # |b|^2
    cc = np.sum(c * c, axis=-1, keepdims=True)  # |c|^2
    
    # Compute cross products
    cross_bc = np.cross(b, c, axis=-1)
    cross_ca = np.cross(c, a, axis=-1)
    cross_ab = np.cross(a, b, axis=-1)
    
    # Compute denominator
    denominator = 2.0 * np.sum(a * cross_bc, axis=-1, keepdims=True)
    
    # Create mask for small denominators
    mask = np.abs(denominator) < 1e-6
    
    # Compute circumcenter relative to verts[0]
    relative_circumcenter = (
        aa * cross_bc +
        bb * cross_ca +
        cc * cross_ab
    ) / np.where(mask, np.ones_like(denominator), denominator)
    

    radius = np.linalg.norm(a - relative_circumcenter, axis=-1)
    
    # Return absolute position
    return vertices[..., 0, :] + relative_circumcenter, radius


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
    denominator = 2.0 * torch.sum(a * cross_bc, dim=-1, keepdim=True)

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


def compute_circumsphere_jacobian(points: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian of circumcenter with respect to tetrahedron vertices.
    
    Args:
        points: (M, 4, 3) tensor of tetrahedron vertex coordinates
    Returns:
        jacobian_matrix: (M, 3, 4, 3) Jacobian tensor
    """
    a = points[..., 1:, :] - points[..., 0:1, :]  # Shape: (3, 3)
    # b = torch.sum(a * a, dim=1) / 2  # Shape: (3,)

    # # Solve for circumcenter relative to points[0]
    # center = torch.linalg.solve(a, b)  # Shape: (3,)
    # center = center + points[0]
    jacobian_matrix = torch.cat([torch.zeros((points.shape[0], 1, 3), device=a.device), torch.linalg.inv(a)], dim=1)
    
    return jacobian_matrix  # Sensitivity of circumcenter w.r.t. tetrahedron vertices

def circumcenter_jacobian(vertices):
    def wrapper(v):
        return torch.linalg.norm(calculate_circumcenters_torch(v)[0], dim=-1)  # Only get circumcenters

    jacobian_numerical = jacobian(wrapper, vertices).squeeze(0)  # (3, 4, 3)
    return jacobian_numerical

# @torch.jit.script
def compute_vertex_sensitivity(indices: torch.Tensor, vertices: torch.Tensor,
                               normalized_circumcenter: torch.Tensor) -> torch.Tensor:
    """
    Compute mean sensitivity for each vertex by vectorizing the computation.
    
    Args:
        indices: (M, 4) tensor of tetrahedron vertex indices
        vertices: (V, 3) tensor of vertex positions
    Returns:
        vertex_sensitivities: (V,) tensor of per-vertex sensitivity
    """
    # Gather all tetrahedra vertex positions
    tetra_points = vertices[indices]  # Shape: (M, 4, 3)
    a = tetra_points[..., 1:, :] - tetra_points[..., 0:1, :]  # Shape: (3, 3)

    # the absolute value of the determinant of the jacobian of the contraction
    # J_d is lower the further from the center it is.
    # sensitivity is lower the further we are from the origin
    J_d = contraction_jacobian_d_in_chunks(normalized_circumcenter).float()

    # we actually find the min eigen value for A, instead of max eigen of A^-1
    # the spectral norm grows as A^-1 becomes more unstable. Our inverse one shrinks
    sp_norm = compute_spectral_norm3(a)

    # jacobian_matrix_sens = J_d.clip(min=1e-3)/compute_spectral_norm3(a).clip(min=1e-5)
    # jacobian_matrix_sens = J_d.clip(min=1e-5)*sp_norm.clip(min=1e-5)
    # jacobian_matrix_sens = sp_norm.clip(min=1e-5) / J_d.clip(min=1e-5)
    jacobian_matrix_sens = sp_norm.clip(min=1e-5)# / J_d.clip(min=1e-5)
    num_vertices = vertices.shape[0]

    vertex_sensitivity = torch.full((num_vertices,), 0.0, device=vertices.device)
    indices = indices.long()

    reduce_type = "sum"
    # reduce_type = "amax"
    vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 0], src=jacobian_matrix_sens, reduce=reduce_type)
    vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 1], src=jacobian_matrix_sens, reduce=reduce_type)
    vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 2], src=jacobian_matrix_sens, reduce=reduce_type)
    vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 3], src=jacobian_matrix_sens, reduce=reduce_type)

    # only use sp_norm here because the perturbations are applied in contracted space
    # multiply by J_d to cancel out contracted perturbations
    return J_d*sp_norm, vertex_sensitivity.reshape(num_vertices, -1)

def fibonacci_spiral_on_sphere(n_points: int, 
                               radius: float = 1.0, 
                               device: str = 'cpu') -> torch.Tensor:
    """
    Generate points on a sphere (approximately evenly) via a Fibonacci spiral.

    Args:
        n_points (int): Number of points to generate on the sphere.
        radius (float): Radius of the sphere. Default = 1.0 (unit sphere).
        device (str): PyTorch device (e.g., 'cpu' or 'cuda'). Default = 'cpu'.

    Returns:
        torch.Tensor: A (n_points x 3) tensor of 3D coordinates on the sphere.
    """
    # Golden angle in radians
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # ~2.39996323

    # Create an index tensor [0, 1, 2, ..., n_points-1]
    idx = torch.arange(n_points, device=device, dtype=torch.float)

    # y ranges from +1 to -1
    y = 1.0 - (idx * 2.0 / (n_points - 1.0))
    # Radius in the plane for each y
    r = torch.sqrt(1.0 - y * y)

    # Fibonacci spiral angle
    theta = golden_angle * idx

    # Project to Cartesian coordinates
    x = r * torch.cos(theta)
    z = r * torch.sin(theta)

    # Stack into an (n_points x 3) tensor and scale by 'radius'
    points = torch.stack([x, y, z], dim=1) * radius

    return points

def expand_convex_hull(points: torch.Tensor, expand_distance: float, device='cpu'):
    points_np = points.cpu().numpy()
    hull = ConvexHull(points_np)
    hull_vertices = points_np[hull.vertices]
    centroid = hull_vertices.mean(axis=0, keepdims=True)
    directions = hull_vertices - centroid
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    expanded_vertices = hull_vertices + expand_distance * directions
    return torch.tensor(expanded_vertices, device=device)

def sample_uniform_in_sphere(batch_size, dim, radius=0.0, device=None):
    """
    Generate samples uniformly distributed inside a sphere.

    Parameters:
        batch_size (int): Number of samples to generate.
        dim (int): Dimensionality of the sphere.
        radius (float): Radius of the sphere (default is 0.0).
        device (torch.device, optional): Device to perform computation on.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, dim) with samples from inside the sphere.
    """
    if device is None:
        device = torch.device("cpu")

    # Sample from a normal distribution
    samples = torch.randn(batch_size, dim, device=device)
    
    # Normalize each vector to lie on the unit sphere
    samples = samples / samples.norm(dim=0, keepdim=True)
    
    # Sample radii uniformly with proper weighting for volume
    radii = torch.rand(batch_size, device=device).pow(1 / dim) * radius
    
    # Scale samples by the radii
    samples = samples * radii.unsqueeze(-1)

    return samples
