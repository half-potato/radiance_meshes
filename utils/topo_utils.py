import numpy as np
from scipy.spatial import KDTree, Delaunay
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import time
import torch
from torch.autograd.functional import jacobian
from icecream import ic
from submodules.spectral_norm3 import compute_spectral_norm3

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

import torch

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

def circumsphere_center_and_jacobian(batch_points):
    """
    Compute the circumsphere center c and its derivatives dc/dp_i
    for a batch of 4 points in 3D.

    Parameters
    ----------
    batch_points : np.ndarray of shape (B,4,3)
        batch_points[b,0,:] = p0  (shape (3,))
        batch_points[b,1,:] = p1
        batch_points[b,2,:] = p2
        batch_points[b,3,:] = p3
        for b = 0..B-1 in the batch.

    Returns
    -------
    c : np.ndarray of shape (B,3)
        The circumsphere center for each batch item.

    dc_dp : np.ndarray of shape (B,4,3,3)
        The 4 Jacobians for each batch item:
            dc_dp[b,i,:,:] = partial c / partial p_i
        which is a 3x3 matrix for each i=0..3.

    Notes
    -----
    The formula is:
        c = p0 + A_inv * b,
    with
        A = [ (p1 - p0); (p2 - p0); (p3 - p0) ]  (3x3)
        b = 0.5 * [ ||p1 - p0||^2; ||p2 - p0||^2; ||p3 - p0||^2 ]  (3,)

    Then
        dc/dp_i = delta_{i0} * I  +  A_inv * (db/dp_i)  -  A_inv * (dA/dp_i) * x
    where x = A_inv * b.

    We implement (db/dp_i) and the operator (dA/dp_i)(·) by direct inspection.
    """
    B = batch_points.shape[0]

    # Unpack the 4 points in each batch
    p0 = batch_points[:, 0, :]  # shape (B, 3)
    p1 = batch_points[:, 1, :]
    p2 = batch_points[:, 2, :]
    p3 = batch_points[:, 3, :]

    #--------------------------------------------------------------------------
    # 1) Construct A and b for each batch item
    #--------------------------------------------------------------------------
    # A[b,:,:] is the 3x3 matrix = [ (p1-p0); (p2-p0); (p3-p0) ]
    #   so A[b,i,:] = p_{i+1} - p0, for i=0..2
    A = np.stack((p1 - p0, p2 - p0, p3 - p0), axis=1)  # shape (B, 3, 3)

    # b[b,:] = 0.5 * [ ||p1 - p0||^2, ||p2 - p0||^2, ||p3 - p0||^2 ]
    def sqnorm(u):
        return np.sum(u * u, axis=-1)  # shape (B,) if u.shape=(B,3)

    b_vals = 0.5 * np.stack((
        sqnorm(p1 - p0),
        sqnorm(p2 - p0),
        sqnorm(p3 - p0)
    ), axis=1)  # shape (B, 3)

    #--------------------------------------------------------------------------
    # 2) Solve for x = A_inv * b, then c = p0 + x
    #--------------------------------------------------------------------------
    # We'll solve A x = b for each batch item.
    # x.shape = (B,3)
    # c.shape = (B,3)
    x_vals = np.linalg.solve(A, b_vals[:, :, None])  # shape (B,3,1)
    x_vals = x_vals[:, :, 0]  # shape (B,3)

    c = p0 + x_vals  # shape (B,3)

    # For repeated use below:
    A_inv = np.linalg.inv(A)  # shape (B,3,3)

    #--------------------------------------------------------------------------
    # 3) Build a small helper to multiply "dA/dp_i" by a vector
    #--------------------------------------------------------------------------
    # Because "dA/dp_i" is a linear operator from R^3 -> R^3, but each row
    # can shift differently, it's often simpler to define a function that:
    #    y = (dA/dp_i)(v)
    # for a given 3-vector v, *without* constructing a normal 3x3 in memory.
    #
    # We'll define these as Python functions that return y.shape = (B,3).
    #
    #   - dA/dp0 : each row i = d/dp0 of (p_{i+1} - p0) = -I
    #              so (dA/dp0)(v) = -v repeated in each row => y = -v
    #
    #   - dA/dp_i for i=1,2,3 : only row (i-1) changes by +I,
    #              so (dA/dp_i)(v) is zero except in row (i-1), which = +v
    #              => y has all zeros except y[i-1] = v
    #
    # That means as a function of v:
    #   (dA/dp0)(v) = -v  (the same for all 3 rows => final is shape (3,) = -v)
    #   (dA/dp1)(v) = [v, 0, 0]
    #   (dA/dp2)(v) = [0, v, 0]
    #   (dA/dp3)(v) = [0, 0, v]
    #
    # But we want the result as a length-3 vector. Each row i is (d row_i)/dp * v.
    # So for dA/dp1, the 0th row is +v, the others are 0 => y = [ (v dot e0?), ...
    # Actually we must keep track carefully. We'll do it explicitly below.
    #
    # Implementation detail: we do batch wise, so v is shape (B,3).

    def dA_dp0_times_v(v):
        # For each batch item b, y[b] = -v[b]
        return -v  # shape (B,3)

    def dA_dp1_times_v(v):
        # For each batch, the row 0 is +v, row 1 & 2 are zero => final is [v0, 0, 0]^T?
        # Actually in standard "row form", the 0th row is derivative w.r.t. p1 => +I
        # => (p1 - p0) row changes => row index=0. So the output is shape(3,)
        # with y[0] = row0 . v = v^T v? That’s not correct. We want "row0 times v" = v·v?
        #
        # More directly from the derivative formula:
        #   The i-th row of A is (p_{i+1}-p0). So for i=0 => p1-p0.
        #   derivative wrt p1 => row0 changes by +I, row1, row2 remain 0.
        # So (dA/dp1)(v) is a 3D vector whose i=0 entry is v[0], i=1 entry=0, i=2 entry=0.
        # => that vector is simply [v_0, v_1, v_2] in the 0th row? Wait, careful:
        #   "row0" is 1×3, times v(3×1) => a scalar, so the 0th component is row0·v.
        #   row0=+I => that dot v => v[0]? Actually row0= [1,0,0], row1=[0,0,0], row2=[0,0,0]? 
        # But we want it in the 0th *component*? The simplest route is:
        #     => (dA/dp1)(v) = ( v, 0, 0 ) as a 3-vector. 
        # In code:
        y = np.zeros_like(v)
        y[:, 0] = v[:, 0]  # x-component => row 0
        y[:, 1] = v[:, 1]  # y-component => row 0? Actually we want the entire v in row 0...
        y[:, 2] = v[:, 2]
        # But that lumps v into the same row 0. We actually want a single dimension?
        # Let’s be more direct: row0 = I, so row0 dot v = v0. We want that in the
        # 0-th component of the result. row1=0 => 1-st component=0, row2=0 => 2-nd=0.
        # So the result is [v0, 0, 0] if v=(v0,v1,v2).  But we want a 3D vector...
        #
        # Actually we want: y0 = row0·v = v0, y1= row1·v=0, y2=row2·v=0 => y=[v0,0,0].
        # So let's do:
        y = np.zeros_like(v)
        y[:, 0] = v[:, 0]  # So if v=(v0,v1,v2), y=(v0,0,0).
        return y

    def dA_dp2_times_v(v):
        # Similarly, row1 changes by +I => y=[0,v0,0], etc.
        y = np.zeros_like(v)
        y[:, 1] = v[:, 1]
        return y

    def dA_dp3_times_v(v):
        # row2 changes by +I => y=[0,0,v0], etc.
        y = np.zeros_like(v)
        y[:, 2] = v[:, 2]
        return y

    #--------------------------------------------------------------------------
    # 4) Build also the partial derivatives of b:
    #
    #   db/dp0   = -A
    #
    #   db/dp_i  = a 3x3 matrix whose only nonzero row is row i,
    #              which is (p_i - p0)^T.  In code, it means:
    #                row i => (p_i - p0), others => 0
    #--------------------------------------------------------------------------
    # We'll store them as actual 3x3 matrices for each i, so we can batch-multiply.
    db_dp0 = -A  # shape (B,3,3)

    # For i=1,2,3: only row (i-1) is (p_i - p0)
    # We'll construct them in a loop or by direct definition:
    db_dp = np.zeros((B, 4, 3, 3), dtype=A.dtype)
    # i=0 => db/dp0
    db_dp[:, 0, :, :] = db_dp0

    # i=1 => place (p1-p0) in row0
    db_dp[:, 1, 0, :] = p1 - p0
    # i=2 => place (p2-p0) in row1
    db_dp[:, 2, 1, :] = p2 - p0
    # i=3 => place (p3-p0) in row2
    db_dp[:, 3, 2, :] = p3 - p0

    #--------------------------------------------------------------------------
    # 5) Put it all together to get dc/dp_i
    #
    #    dc/dp_i = delta_{i0} * I + A_inv * db/dp_i - A_inv * (dA/dp_i)( x )
    #
    # We will store results in a big array dcdp of shape (B,4,3,3).
    #--------------------------------------------------------------------------
    I3 = np.eye(3, dtype=A.dtype)
    dcdp = np.zeros((B, 4, 3, 3), dtype=A.dtype)

    # We'll define a small helper that does the matrix multiply for a batch:
    #   batch_matmul_3x3x3( (B,3,3), (B,3,3) ) -> (B,3,3)
    def batch_matmul(M, N):
        # M.shape = (B,3,3), N.shape = (B,3,3), output (B,3,3)
        return np.einsum('bij,bjk->bik', M, N)

    # We'll also define a helper that does (A_inv)*(dA/dp_i)(x).
    # The result is (B,3).  Then we embed it as needed.
    def Ainv_dA_dp_i_x(dA_dp_i_times_v_func):
        # First compute v = (dA/dp_i)( x ), shape (B,3).
        v = dA_dp_i_times_v_func(x_vals)
        # Then multiply by A_inv to get shape (B,3).
        # We'll treat v as (B,3,1) for the matmul:
        v_3x1 = v[..., None]  # shape (B,3,1)
        Av = np.einsum('bij,bjk->bik', A_inv, v_3x1)  # (B,3,1)
        return Av[..., 0]  # shape (B,3)

    # Loop i=0..3 and fill dcdp[:, i, :, :]
    for i in range(4):
        # 1) delta_{i0} * I
        if i == 0:
            dcdp[:, i, :, :] = I3  # shape (3,3) broadcast to (B,3,3)

        # 2) + A_inv * (db/dp_i)
        #    shape after multiply => (B,3,3)
        dcdp[:, i, :, :] += batch_matmul(A_inv, db_dp[:, i, :, :])

        # 3) - A_inv * (dA/dp_i)(x)
        if i == 0:
            # dA/dp0 => each row = -I => (dA/dp0)(x) = -x
            # so shape (B,3)
            tmp = -x_vals
        elif i == 1:
            tmp = dA_dp1_times_v(x_vals)
        elif i == 2:
            tmp = dA_dp2_times_v(x_vals)
        else:
            tmp = dA_dp3_times_v(x_vals)
        # Then multiply by A_inv => shape (B,3)
        tmp_3x1 = tmp[..., None]  # shape (B,3,1)
        minus_term = np.einsum('bij,bjk->bik', A_inv, tmp_3x1)[:, :, 0]  # (B,3)
        # Subtract it from dcdp:
        # remember that each column of dcdp[:,i,:,:] corresponds to how c changes
        # if we nudge p_i in that coordinate direction.  So we subtract "minus_term"
        # from each column. The easiest way: reshape minus_term to (B,3,1) and broadcast:
        dcdp[:, i, :, :] -= minus_term[:, :, None]

    #--------------------------------------------------------------------------
    # Done!  Return c and all partial derivatives
    #--------------------------------------------------------------------------
    return c, dcdp

# def circumcenter_jacobian(vertices):
#     _, jac = circumsphere_center_and_jacobian(vertices.detach().numpy())
#     jac = torch.as_tensor(jac)
#     return jac.permute(2, 0, 1, 3)

def circumcenter_jacobian(vertices):
    def wrapper(v):
        return torch.linalg.norm(calculate_circumcenters_torch(v)[0], dim=-1)  # Only get circumcenters

    jacobian_numerical = jacobian(wrapper, vertices).squeeze(0)  # (3, 4, 3)
    return jacobian_numerical

def compute_vertex_sensitivity(indices: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
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

    # Compute Jacobian of circumcenters (M, 3, 4, 3)
    # jacobian_matrix = compute_circumsphere_jacobian(tetra_points)
    # jacobian_matrix_sens = (torch.linalg.norm(jacobian_matrix, dim=-1)+1e-5)**2
    # jacobian_matrix_sens = 1/torch.linalg.matrix_norm(a, ord=-2).clip(min=1e-5)
    # jacobian_matrix_sens = 1/torch.linalg.matrix_norm(a, ord='fro').clip(min=1e-5)
    jacobian_matrix_sens = 1/compute_spectral_norm3(a).clip(min=1e-5)
    # ic(jacobian_matrix_sens.mean(), jacobian_matrix_sens)
    # Compute sensitivity per vertex (Frobenius norm across Jacobian dimensions)
    num_vertices = vertices.shape[0]

    # Scatter sensitivity back to vertices
    # vertex_sensitivity = torch.full((num_vertices,3), float('-inf'), device=sensitivity_per_tetra.device)
    vertex_sensitivity = torch.full((num_vertices,), 0.0, device=vertices.device)
    indices = indices.long()
    # jacobian_matrix_sens = 1/(torch.linalg.det(a)+1e-8) + jacobian_matrix_sens.sum(dim=-1)
    # jacobian_matrix_sens = jacobian_matrix_sens.sum(dim=-1)

    reduce_type = "amax"
    # reduce_type = "sum"
    # reduce_type = "amin"
    vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 0], src=jacobian_matrix_sens, reduce=reduce_type)
    vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 1], src=jacobian_matrix_sens, reduce=reduce_type)
    vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 2], src=jacobian_matrix_sens, reduce=reduce_type)
    vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 3], src=jacobian_matrix_sens, reduce=reduce_type)
    # vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 0], src=jacobian_matrix_sens[:, 0], reduce=reduce_type)
    # vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 1], src=jacobian_matrix_sens[:, 1], reduce=reduce_type)
    # vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 2], src=jacobian_matrix_sens[:, 2], reduce=reduce_type)
    # vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 3], src=jacobian_matrix_sens[:, 3], reduce=reduce_type)
    # vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 0], src=sensitivity_per_tetra, reduce="amax")
    # vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 1], src=sensitivity_per_tetra, reduce="amax")
    # vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 2], src=sensitivity_per_tetra, reduce="amax")
    # vertex_sensitivity.scatter_reduce_(dim=0, index=indices[..., 3], src=sensitivity_per_tetra, reduce="amax")

    return vertex_sensitivity.reshape(num_vertices, -1)
