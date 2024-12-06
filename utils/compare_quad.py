import numpy as np
import torch
import torch.nn as nn
import math
from pathlib import Path

from jaxutil import tetra_quad
from delaunay_rasterization.internal.alphablend_tiled_slang import AlphaBlendTiledRender
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from pyquaternion import Quaternion
from icecream import ic
from jax import jacrev
import jax.numpy as jnp


def get_projection_matrix(znear, zfar, fy, fx, height, width, device):
    """Calculate projection matrix for given camera parameters."""
    tanHalfFovX = width/(2*fx)
    tanHalfFovY = height/(2*fy)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.tensor([
       [2.0 * znear / (right - left),     0.0,                          (right + left) / (right - left), 0.0],
       [0.0,                              2.0 * znear / (top - bottom), (top + bottom) / (top - bottom), 0.0],
       [0.0,                              0.0,                          zfar / (zfar - znear),          -(zfar * znear) / (zfar - znear)],
       [0.0,                              0.0,                          1.0,                             0.0]
    ], device=device)

    return P

def focal2fov(focal, pixels):
    """Convert focal length to field of view."""
    return 2 * math.atan(pixels/(2*focal))

def setup_camera(height, width, fov_degrees, viewmat):
    """Set up camera parameters."""
    f = height / math.tan(fov_degrees * math.pi / 180 / 2.0)
    K = torch.tensor([
        [f, 0, width/2],
        [0, f, height/2],
        [0, 0, 1],
    ])
    
    
    # Camera properties
    zfar = 100.0
    znear = 0.01
    fx = K[0,0]
    fy = K[1,1]
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)
    
    projection_matrix = get_projection_matrix(znear, zfar, fy, fx, height, width, K.device)
    cam_pos = viewmat.inverse()[:3, 3]
    
    return viewmat, K, cam_pos, fovy, fovx, fx, fy

def test_tetrahedra_rendering(vertices, indices, rgbs, viewmat, n_samples=10000, height=3, width=3,
                              tile_size=16, fov=90, check_gradients=True):
    """
    Test tetrahedra rendering by comparing JAX and PyTorch implementations.
    
    Args:
        vertices (torch.Tensor): Vertex positions (N x 3)
        indices (torch.Tensor): Tetrahedra indices (M x 4)
        rgbs (torch.Tensor): RGBA values for each tetrahedra (M x 4)
        height (int): Image height
        width (int): Image width
        tile_size (int): Size of rendering tiles
        fov (float): Field of view in degrees
        
    Returns:
        dict: Dictionary containing test results and error metrics
    """
    # Setup camera
    viewmat, projection_matrix, cam_pos, fovy, fovx, fx, fy = setup_camera(height, width, fov, viewmat)
    # ic(projection_matrix)
    
    # Setup rendering grid
    render_grid = RenderGrid(height, width, tile_height=tile_size, tile_width=tile_size)
    if check_gradients:
        # Detach and enable gradients
        vertices = vertices.detach().requires_grad_(True)
        rgbs = rgbs.detach().requires_grad_(True)
    
    # Get sorted tetrahedra and other rendering parameters
    sorted_tetra_idx, tile_ranges, radii, vs_tetra, circumcenter, mask = vertex_and_tile_shader(
        indices,
        vertices,
        viewmat.cuda(),
        projection_matrix.cuda(),
        cam_pos.cuda(),
        fovy,
        fovx,
        render_grid
    )
    # ic(vs_tetra)
    
    # Render using PyTorch implementation
    torch_image = AlphaBlendTiledRender.apply(
        sorted_tetra_idx,
        tile_ranges,
        indices,
        vertices,
        rgbs,
        render_grid,
        viewmat.cuda(),
        projection_matrix.cuda(),
        cam_pos.cuda(),
        fovy,
        fovx
    )
    
    jax_image, extras = tetra_quad.render_camera(
        vertices.detach().cpu().numpy(), indices.cpu().numpy(),
        rgbs.detach().cpu().numpy(),
        height, width, viewmat.cpu().numpy(),
        fx.item(), fy.item(), np.linspace(0, 1, n_samples))

    # Compare results
    torch_image_np = torch_image[..., :3].cpu().detach().numpy()
    diff = jax_image[..., :3] - torch_image_np
    mean_error = np.abs(diff).mean()
    max_error = np.abs(diff).max()
    
    results = {
        'torch_image': torch_image_np,
        'jax_image': jax_image[..., :3],
        'difference': diff,
        'mean_error': mean_error,
        'max_error': max_error,
        'extras': extras
    }

    if check_gradients:
        # Compute gradients through both implementations
        torch_loss = torch_image[..., :3].mean()
        torch_loss.backward()
        
        # Store PyTorch gradients
        results['torch_vertex_grad'] = vertices.grad.clone().cpu().numpy()
        results['torch_rgbs_grad'] = rgbs.grad.clone().cpu().numpy()
        
        def render_fn(verts_and_rgbs):
            verts, colors = verts_and_rgbs
            img, _ = tetra_quad.render_camera(
                verts,
                indices.cpu().numpy(),
                colors,
                height, width, viewmat.cpu().numpy(),
                fx.item(), fy.item(),
                jnp.linspace(0, 1, n_samples)
            )
            return img[..., :3].mean()

        # Compute JAX gradients using jacrev
        jax_verts_grad, jax_rgbs_grad = jacrev(render_fn)((
            vertices.detach().cpu().numpy(),
            rgbs.detach().cpu().numpy()
        ))
            
        # Compute JAX gradients (assuming tetra_quad.render_camera returns gradients)
        results['jax_vertex_grad'] = np.array(jax_verts_grad)
        results['jax_rgbs_grad'] = np.array(jax_rgbs_grad)

        results['vertex_err'] = np.abs(results['torch_vertex_grad'] - results['jax_vertex_grad'])
        results['rgbs_err'] = np.abs(results['torch_rgbs_grad'] - results['jax_rgbs_grad'])

    return results

def generate_face_view(vertices, indices):
    """
    Generate a view matrix looking at a randomly selected face of a tetrahedron.
    Returns the view matrix and the selected face point.
    """
    # Randomly select one face (3 vertices) from the tetrahedron
    face_idx = torch.randint(0, 4, (1,)).item()  # 4 faces in a tetrahedron
    face_vertices = []
    if face_idx == 0:
        face_vertices = [1, 2, 3]
    elif face_idx == 1:
        face_vertices = [0, 2, 3]
    elif face_idx == 2:
        face_vertices = [0, 1, 3]
    else:
        face_vertices = [0, 1, 2]
    
    # Get vertices of the selected face
    v1 = vertices[indices[0][face_vertices[0]]]
    v2 = vertices[indices[0][face_vertices[1]]]
    v3 = vertices[indices[0][face_vertices[2]]]
    
    # Generate random barycentric coordinates for the face
    face_barycentric = torch.rand(3).cuda()
    face_barycentric = face_barycentric / face_barycentric.sum()
    
    # Calculate point on face
    face_point = (v1 * face_barycentric[0] + 
                 v2 * face_barycentric[1] + 
                 v3 * face_barycentric[2])
    
    # Calculate face normal using cross product
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = torch.cross(edge1, edge2)
    normal = normal / torch.norm(normal)
    
    # Generate random offset from face along normal
    offset_distance = torch.rand(1).cuda().item() * 10  # Adjust scale as needed
    camera_position = face_point + normal * offset_distance
    
    # Create orthonormal basis for view matrix
    # Forward direction (z) - points from camera to face_point
    forward = face_point - camera_position
    forward = forward / torch.norm(forward)
    
    # Generate random vector for up direction
    random_vec = torch.randn(3).cuda()
    # Right direction (x)
    right = torch.cross(forward, random_vec)
    right = right / torch.norm(right)
    # Up direction (y)
    up = torch.cross(right, forward)
    up = up / torch.norm(up)
    
    # Construct view matrix
    viewmat = torch.eye(4).cuda()
    viewmat[:3, 0] = right
    viewmat[:3, 1] = up
    viewmat[:3, 2] = forward
    viewmat[:3, 3] = camera_position
    
    return viewmat, face_point, normal

def random_rot():
    # Generate random quaternion components
    q = np.random.randn(4)
    q = q / np.linalg.norm(q)  # Normalize to unit quaternion
    quat = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    return quat.rotation_matrix

# Example usage
if __name__ == "__main__":
    # Create sample tetrahedra
    torch.manual_seed(0)
    N = 4
    vertices = (torch.rand((N, 3)).cuda() * 2 - 1) * 100
    indices = torch.tensor([[0, 1, 2, 3]]).int().cuda()
    rgbs = torch.rand(1, 4).cuda()  # Random RGBA values

    # Generate random barycentric coordinates that sum to 1
    barycentric = torch.rand(4).cuda()
    barycentric = barycentric / barycentric.sum()

    # Use barycentric coordinates to get a point inside the tetrahedron
    origin = vertices[indices[0]].T @ barycentric  # Shape: (3,)
    print(origin, vertices)

    # Update viewmat with new origin
    viewmat = torch.eye(4)
    viewmat[:3, 3] = origin  # Set translation to sampled point
    viewmat = torch.linalg.inv(viewmat)
    
    # Run test
    results = test_tetrahedra_rendering(
        vertices.cuda(), indices, rgbs, viewmat,
        # height=32, width=32)
        height=8, width=8)
    
    print(f"Max value: {results['torch_image'].max()}")
    print(f"Mean Error: {results['mean_error']}")
    print(f"Max Error: {results['max_error']}")