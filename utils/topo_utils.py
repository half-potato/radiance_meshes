import numpy as np
import torch
from torch.autograd.functional import jacobian
from icecream import ic
import math
from scipy.spatial import ConvexHull
from utils.safe_math import safe_div

def tet_volumes(tets):
    v0 = tets[:, 0]
    v1 = tets[:, 1]
    v2 = tets[:, 2]
    v3 = tets[:, 3]

    a = v1 - v0
    b = v2 - v0
    c = v3 - v0
    
    mat = torch.stack((a, b, c), dim=1)
    det = torch.det(mat)
    
    vol = det / 6.0
    return vol

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
    mask = torch.abs(denominator) < 1e-10

    # Compute circumcenter relative to verts[0]
    relative_circumcenter = safe_div(aa * cross_bc + bb * cross_ca + cc * cross_ab, denominator)
    # relative_circumcenter = (
    #     aa * cross_bc +
    #     bb * cross_ca +
    #     cc * cross_ab
    # ) / torch.where(mask, torch.ones_like(denominator), denominator)

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

def sample_uniform_in_sphere(batch_size, dim, radius=0.0, base_radius=0.0, device=None):
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
    samples = samples / samples.norm(dim=1, keepdim=True)
    
    # Sample radii uniformly with proper weighting for volume
    radii = torch.rand(batch_size, device=device).pow(1 / dim) * radius + base_radius
    
    # Scale samples by the radii
    samples = samples * radii.unsqueeze(-1)

    return samples

def build_tv_struct(verts, tets, device=None):
    """
    verts : (V,3)  float32/64
    tets  : (T,4)  int64
    returns:
        pairs  : (M,2) int64   neighbouring-tet indices
        areas  : (M,)  float32
    """
    if device is None: device = verts.device
    T = tets.shape[0]

    # ---- all faces -------------------------------------------------
    face_lists = torch.stack([
        tets[:, (1,2,3)],
        tets[:, (0,2,3)],
        tets[:, (0,1,3)],
        tets[:, (0,1,2)],
    ], dim=1).reshape(-1, 3)                                    # (4T,3)

    face_sorted, _ = face_lists.sort(dim=1)                     # canonical key
    owner_tet = torch.arange(T, device=device).repeat_interleave(4)

    # ---- group identical faces ------------------------------------
    uniq, inv, counts = torch.unique(face_sorted, dim=0,
                                     return_inverse=True,
                                     return_counts=True)
    interior = counts == 2                                      # only faces with 2 owners
    mask      = interior[inv]                                   # (4T,) boolean
    face_rows = torch.nonzero(mask, as_tuple=False)[:,0]        # rows belonging to int. faces

    # bucket rows by face id, then pick the two tet owners
    face_id   = inv[face_rows]
    sort_idx  = torch.argsort(face_id)
    face_rows = face_rows[sort_idx]
    face_id   = face_id[sort_idx]

    owners    = owner_tet[face_rows].reshape(-1,2)              # (M,2) neighbouring tets

    # ---- geometric weight -----------------------------------------
    fverts = verts[face_lists[face_rows]]                        # (M,3,3)
    v01 = fverts[:,1] - fverts[:,0]
    v02 = fverts[:,2] - fverts[:,0]
    areas = 0.5 * torch.linalg.norm(torch.cross(v01, v02, dim=1), dim=1)

    return owners, areas[::2]

def max_density_contrast(vertices, indices, density,
                          mode: str = "diff",
                          eps: float = 1e-8) -> torch.Tensor:
    """
    Maximum *one-way* contrast with respect to face-neighbours.

    Parameters
    ----------
    vertices : (V,3)  — float32/64   (only needed so build_tv_struct can compute faces)
    indices  : (T,4)  — int64        tetrahedron index list
    density  : (T,)   — float32/64   per-tet density
    mode     : "diff" | "ratio"
        "diff"  – max(d_self − d_neigh)             (only neighbours with lower density)
        "ratio" – max(d_self / d_neigh)             (            »               )
    eps      : small constant to stabilise division in "ratio".

    Returns
    -------
    contrast : (T,)  – max contrast value for each tet, 0 if it is never the higher side
    """
    assert mode in ("diff", "ratio")

    # ---- neighbouring tet pairs --------------------------------------------
    pairs, _ = build_tv_struct(vertices, indices, device=vertices.device)  # (M,2)

    d0 = density[pairs[:, 0]]
    d1 = density[pairs[:, 1]]

    # Determine which tet in each pair has the higher density
    first_is_higher = d0 >= d1
    src_idx   = torch.where(first_is_higher, pairs[:, 0], pairs[:, 1])  # winning tet
    neigh_idx = torch.where(first_is_higher, pairs[:, 1], pairs[:, 0])  # lower-density neighbour

    if mode == "diff":
        edge_val = (density[src_idx] - density[neigh_idx])            # positive
    else:  # "ratio"
        edge_val = density[src_idx] / (density[neigh_idx] + eps)

    # ---- per-tet maximum over its outgoing edges ---------------------------
    contrast = torch.zeros_like(density)
    # PyTorch ≥ 2.1
    contrast.scatter_reduce_(0, src_idx, edge_val, reduce="amax")

    return contrast


def build_adj_matrix(num_tets, owners):
    """
    Helper function to build a sparse adjacency matrix from owner pairs.
    adj[i, j] is True if tet i and tet j are neighbors.
    """
    device = owners.device
    # Build a symmetric adjacency list from owner pairs
    adj_indices = torch.cat([owners.T, owners.T[[1, 0]]], dim=1)

    # Coalesce to handle any potential duplicate edges and create a valid sparse tensor
    adj_sparse = torch.sparse_coo_tensor(
        adj_indices,
        torch.ones(adj_indices.shape[1], device=device),
        (num_tets, num_tets)
    ).coalesce()

    # The values don't matter, only the connectivity.
    # Return a boolean sparse tensor for efficient logical operations.
    return torch.sparse_coo_tensor(
        adj_sparse.indices(),
        torch.ones_like(adj_sparse.values(), dtype=torch.bool),
        adj_sparse.size(),
        dtype=torch.bool
    )

def build_adj(verts, tets, device=None):
    """
    Builds a (T, 4) adjacency map for a tetrahedron mesh.

    verts : (V,3)
    tets  : (T,4)
    returns:
        tet_adj : (T,4) int64
                  tet_adj[i, j] = index of tet adjacent to face j of tet i
                  (face j is opposite vertex j)
                  Value is -1 if it's a boundary face.
    """
    if device is None: device = verts.device
    T = tets.shape[0]

    face_lists = torch.stack([
        tets[:, (0, 2, 1)], 
        tets[:, (1, 2, 3)], 
        tets[:, (0, 3, 2)], 
        tets[:, (3, 0, 1)], 
        # tets[:, (1, 2, 3)], 
        # tets[:, (0, 3, 2)], 
        # tets[:, (0, 1, 3)], 
        # tets[:, (0, 2, 1)], 
    ], dim=1).reshape(-1, 3)                         # (4T,3)
    face_sorted, _ = face_lists.sort(dim=1)          # canonical key
    
    # owner_tet[i] = tet index for global face i
    owner_tet = torch.arange(T, device=device).repeat_interleave(4) # (4T,)
    # local_face[i] = local face index (0-3) for global face i
    local_face = torch.arange(4, device=device).repeat(T)           # (4T,)


    # ---- group identical faces ------------------------------------
    uniq, inv, counts = torch.unique(face_sorted, dim=0,
                                     return_inverse=True,
                                     return_counts=True)
    interior = counts == 2
    mask     = interior[inv]                         # (4T,) boolean
    
    # Get global face indices (0..4T-1) for all interior faces
    face_rows = torch.nonzero(mask, as_tuple=False)[:,0] # (2*M,)

    # ---- build (T,4) map ------------------------------------------
    
    # 1. Sort face_rows by their unique face id
    face_id   = inv[face_rows]
    sort_idx  = torch.argsort(face_id)
    face_rows = face_rows[sort_idx] # Shape (2*M,)

    # 2. Get tet owners and local faces for each pair
    tet_owners  = owner_tet[face_rows]  # (2*M,)
    local_faces = local_face[face_rows] # (2*M,)

    # 3. Reshape to get pairs
    # tet_A[i] and tet_B[i] are neighbors
    # face_A[i] is local face of tet_A leading to tet_B
    # face_B[i] is local face of tet_B leading to tet_A
    tet_A  = tet_owners[::2]
    tet_B  = tet_owners[1::2]
    face_A = local_faces[::2]
    face_B = local_faces[1::2]

    # 4. Create and populate the (T, 4) adjacency map
    # Initialize all faces to -1 (boundary)
    tet_adj = torch.full((T, 4), -1, device=device, dtype=torch.int64)
    
    tet_adj[tet_A, face_A] = tet_B
    tet_adj[tet_B, face_B] = tet_A
    
    return tet_adj

def get_tet_adjacency(tets: torch.Tensor):
    """
    Takes a tensor of N tetrahedra indices and finds all M unique
    faces, returning the oriented faces and a map of their neighboring tets.

    Args:
        tets: A (N, 4) long tensor of tetrahedra indices.

    Returns:
        A tuple of (faces, side_index):
        - faces: (M, 3) long tensor of unique, oriented face indices.
        - side_index: (M, 2) long tensor.
            - side_index[i, 0] is the index of the tet for the "front"
              face (faces[i]).
            - side_index[i, 1] is the index of the "back" face tet,
              or -1 if it's a boundary face.
    """
    if not (tets.ndim == 2 and tets.shape[1] == 4):
        raise ValueError(f"Input tensor must have shape (N, 4), "
                         f"but got {tets.shape}")
        
    N = tets.shape[0]
    device = tets.device

    # ---
    # 1. Define all 4 faces for all N tets
    # We assume a standard winding order:
    # face 0: [v0, v1, v2] (base)
    # face 1: [v0, v3, v1] (side)
    # face 2: [v1, v3, v2] (side)
    # face 3: [v2, v3, v0] (side)
    # ---
    f0 = tets[:, [0, 1, 2]]
    f1 = tets[:, [0, 3, 1]]
    f2 = tets[:, [1, 3, 2]]
    f3 = tets[:, [2, 3, 0]]
    
    # Stack into a (4*N, 3) tensor
    all_faces = torch.stack([f0, f1, f2, f3], dim=1).reshape(-1, 3)

    # ---
    # 2. Create a (4*N) tensor to map each face back to its tet index
    # ---
    # [0, 0, 0, 0, 1, 1, 1, 1, ... N-1, N-1, N-1, N-1]
    tet_idx_map = torch.arange(N, device=device).unsqueeze(1).expand(N, 4).reshape(-1)

    # ---
    # 3. Create a unique, hashable key for each face by sorting its indices
    # ---
    # (4*N, 3)
    sorted_faces, _ = torch.sort(all_faces, dim=1)

    # ---
    # 4. Find unique sorted faces and group all 4*N faces by them
    # ---
    # unique_sorted_keys: (M, 3) tensor, where M is num unique faces
    # inverse_map: (4*N) tensor mapping each of the 4N faces to its
    #                unique ID (from 0 to M-1)
    unique_sorted_keys, inverse_map = torch.unique(
        sorted_faces, dim=0, return_inverse=True
    )
    
    M = unique_sorted_keys.shape[0]

    # ---
    # 5. Sort by the unique ID. This groups matching faces (pairs) together.
    # ---
    # perm will be a (4*N) tensor of indices [0...4N-1]
    perm = torch.argsort(inverse_map)
    
    # Re-order all our data using this permutation
    sorted_inverse_map = inverse_map[perm]  # [0, 0, 1, 1, 2, 3, 3, ...]
    sorted_tet_idx = tet_idx_map[perm]
    sorted_all_faces = all_faces[perm]      # This has the original winding

    # ---
    # 6. Find the *first* instance of each unique face
    # ---
    # We find where the ID changes in sorted_inverse_map
    # [True, False, True, False, True, True, False, ...]
    change_mask = torch.cat(
        (torch.tensor([True], device=device), 
         sorted_inverse_map[1:] != sorted_inverse_map[:-1])
    )
    # first_indices will be (M) tensor giving the index into the
    # sorted_* tensors for the first instance of each unique face.
    first_indices = torch.where(change_mask)[0]

    # ---
    # 7. Build the final (M, 3) faces and (M, 2) side_index
    # ---
    
    # Initialize outputs
    # The (M, 3) oriented faces
    faces = torch.zeros((M, 3), dtype=torch.long, device=device)
    # The (M, 2) adjacency map, default to -1 (boundary)
    side_index = torch.full((M, 2), -1, dtype=torch.long, device=device)

    # Fill the "front" face (slot 0) for ALL M faces
    # This is the oriented face from the first tet we found
    faces = sorted_all_faces[first_indices]
    side_index[:, 0] = sorted_tet_idx[first_indices]

    # ---
    # 8. Find internal faces (duplicates) and fill the "back" (slot 1)
    # ---
    
    # A count of 2 means it's an internal face
    counts = torch.bincount(sorted_inverse_map, minlength=M)
    internal_mask = (counts == 2)
    
    # Get the M-indices of just the internal faces
    internal_face_indices_M = torch.where(internal_mask)[0]
    
    # The *second* instance of these faces is at `first_indices + 1`
    # in the sorted tensors
    second_indices_sorted = first_indices[internal_mask] + 1
    
    # Get the tet ID for the second face in the pair
    second_tet_idx = sorted_tet_idx[second_indices_sorted]
    
    # Scatter these tet IDs into slot 1 of the side_index
    side_index[internal_face_indices_M, 1] = second_tet_idx

    return faces, side_index
