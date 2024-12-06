from absl.testing import absltest
from absl.testing import parameterized
import torch
from pyquaternion import Quaternion
import numpy as np
from utils.compare_quad import test_tetrahedra_rendering
import random
import math

key_pairs = [
    ('torch_image', 'jax_image', 'Forward pass'),
    ('torch_vertex_grad', 'jax_vertex_grad', 'Vertex gradient'),
    ('torch_rgbs_grad', 'jax_rgbs_grad', 'RGB gradient')
]

def compare_dict_values(results1, results2, keys_to_compare, vertices=None, viewmat=None):
    """
    Compare values from two dictionaries, checking for error magnitude and convergence.
    
    Args:
        results1 (dict): First results dictionary (with more samples)
        results2 (dict): Second results dictionary (with fewer samples)
        keys_to_compare (list): List of tuples containing:
            - key1: Key for first value to compare
            - key2: Key for second value to compare
            - description: String description of what's being compared
        vertices (torch.Tensor, optional): Vertex values for error reporting
        viewmat (torch.Tensor, optional): View matrix for error reporting
    """
    for key1, key2, description in keys_to_compare:
        error1 = np.abs(results1[key1] - results1[key2]).mean()
        error2 = np.abs(results2[key1] - results2[key2]).mean()
        
        # Check if error is both large and non-decreasing
        error_threshold = 1e-1
        if error1 > error_threshold and error2 >= error1:
            error_message = (
                f"\n{description} error is large and non-decreasing:"
                f"\nError with n_samples=10000: {error1:.6f}"
                f"\nError with n_samples=5000: {error2:.6f}"
                f"\nRelative error increase: {(error2/error1 - 1)*100:.2f}%"
            )
            if vertices is not None:
                error_message += f"\nvertices = torch.{vertices}"
            if viewmat is not None:
                error_message += f"\nviewmat = torch.{viewmat}"
            raise AssertionError(error_message)
            
        try:
            np.testing.assert_allclose(results1[key1], results1[key2], atol=1e-1, rtol=1e-1)
        except AssertionError as e:
            error_message = f"\n{description} error: {error1:.6f}"
            if vertices is not None:
                error_message += f"\nvertices = torch.{vertices}"
            if viewmat is not None:
                error_message += f"\nviewmat = torch.{viewmat}"
            raise AssertionError(error_message) from e

class TetrahedraRenderingTest(parameterized.TestCase):
    def setUp(self):
        torch.manual_seed(189710234)
        self.height = 32
        self.width = 32
        self.indices = torch.tensor([[0, 1, 2, 3]]).int().cuda()
        
    def _create_base_tetrahedra(self, radius):
        vertices = (torch.rand((4, 3)).cuda() * 2 - 1) * radius
        return vertices
        
    def _random_rotation_matrix(self):
        q = np.random.randn(4)
        q = q / np.linalg.norm(q)
        quat = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
        return torch.tensor(quat.rotation_matrix, device='cuda').float()

    def run_test(self, vertices, viewmat, tile_size):
        rgbs = torch.ones(1, 4).cuda()
        rgbs[:, 3] = 10
        results = test_tetrahedra_rendering(vertices, self.indices, rgbs, viewmat, 
                                         height=self.height, width=self.width, tile_size=tile_size, n_samples=10000)
        results2 = test_tetrahedra_rendering(vertices, self.indices, rgbs, viewmat, 
                                         height=self.height, width=self.width, tile_size=tile_size, n_samples=5000)
        compare_dict_values(results, results2, key_pairs, vertices, viewmat)


    @parameterized.product(
        tile_size=[4]#, 8, 16],
    )
    def test_center_view(self, tile_size, N=20, radius=100):
        """Test rendering from random center with random rotation."""
        for i in range(N):
            vertices = self._create_base_tetrahedra(radius)
            
            # Generate barycentric point
            barycentric = torch.rand(4).cuda()
            barycentric = barycentric / barycentric.sum()
            origin = vertices[self.indices[0]].T @ barycentric
            
            # Create view matrix with random rotation
            viewmat = torch.eye(4).cuda()
            viewmat[:3, :3] = self._random_rotation_matrix()
            viewmat[:3, 3] = origin
            viewmat = torch.linalg.inv(viewmat)
            
            self.run_test(vertices, viewmat, tile_size)

    @parameterized.product(
        offset_mag=[0.1, 1, 5, 10, 100, 1000],
        tile_size=[4]#, 8, 16],
    )
    def test_face_view(self, offset_mag, tile_size, width=32, height=32, radius=100, N=5):
        """Test rendering from face with inward-pointing rotation."""
        for i in range(N):
            vertices = self._create_base_tetrahedra(radius)
            
            # Select random face
            face_idx = torch.randint(0, 4, (1,)).item()
            face_verts = [[1,2,3], [0,2,3], [0,1,3], [0,1,2]][face_idx]
            
            # Generate point on face
            face_barycentric = torch.rand(3).cuda()
            face_barycentric = face_barycentric / face_barycentric.sum()
            face_vertices = vertices[self.indices[0][face_verts]]
            face_point = face_vertices.T @ face_barycentric
            
            # Calculate normal and offset
            edge1 = face_vertices[1] - face_vertices[0]
            edge2 = face_vertices[2] - face_vertices[0]
            normal = torch.cross(edge1, edge2)
            normal = normal / torch.norm(normal)
            
            # Position camera
            offset = (random.random()*2-1) * offset_mag
            camera_pos = face_point + normal * offset
            
            # Create view matrix looking at face point
            forward = face_point - camera_pos
            forward = forward / torch.norm(forward)
            right = torch.cross(forward, torch.randn(3).cuda())
            right = right / torch.norm(right)
            up = torch.cross(right, forward)
            
            viewmat = torch.eye(4).cuda()
            viewmat[:3, 0] = right
            viewmat[:3, 1] = up
            viewmat[:3, 2] = forward
            viewmat[:3, 3] = camera_pos
            viewmat = torch.linalg.inv(viewmat)
            
            self.run_test(vertices, viewmat, tile_size)

    @parameterized.product(
        depth=[1, 5, 10],
        origin_radius=[1],
        tile_size=[4]#, 8, 16],
    )
    def test_frustum_point(self, depth, origin_radius, tile_size, radius=1, N=5):
        """Test rendering tetrahedra positioned in view frustum."""
        for i in range(N):
            vertices = self._create_base_tetrahedra(radius)
            
            # Sample point in view frustum at identity
            # For 90 degree FOV, frustum forms right triangle
            z = torch.rand(1).cuda().item() * depth / 2 + depth/2  # depth
            xy_range = z  # at z distance, xy bounds are [-z, z]
            x = (torch.rand(1).cuda().item() * 2 - 1) * xy_range
            y = (torch.rand(1).cuda().item() * 2 - 1) * xy_range
            frustum_point = torch.tensor([x, y, z], device='cuda')
            
            # Create random camera transform
            viewmat = torch.eye(4).cuda()
            viewmat[:3, :3] = self._random_rotation_matrix()
            viewmat[:3, 3] = torch.randn(3).cuda() * origin_radius  # random origin
            
            # Transform frustum point to camera space
            world_point = viewmat @ torch.cat([frustum_point, torch.ones(1).cuda()])
            world_point = world_point[:3] / world_point[3]
            
            # Create tetrahedra at transformed point
            vertices = vertices + world_point
            
            self.run_test(vertices, viewmat, tile_size)

if __name__ == '__main__':
    absltest.main()
