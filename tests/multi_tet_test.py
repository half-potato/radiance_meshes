from absl.testing import absltest
from absl.testing import parameterized
import torch
from pyquaternion import Quaternion
import numpy as np
from utils.compare_quad import test_tetrahedra_rendering
from utils.test_util import compare_dict_values, bcolors, compute_delaunay, generate_color_palette
import random
import math
from utils.model_util import offset_normalize
from utils.topo_utils import calculate_circumcenters_torch

key_pairs = [
    ('torch_image', 'jax_image', 'Image', 1e-1, 1e-1),
    ('torch_dist_loss', 'jax_dist_loss', 'Distortion Loss', 1e-2, 1e-2),
    ('torch_tet_density_grad', 'jax_tet_density_grad', 'Sigma gradient', 1e-1, 1e-1),
    ('torch_cell_values_grad', 'jax_cell_values_grad', 'Cell values gradient', 1e-1, 1e-1)
]

def generate_point_cloud(n_points, radius, device='cuda'):
    """Generate a random point cloud within a sphere of given radius."""
    points = (torch.rand((n_points, 3), device=device) * 2 - 1) * radius
    distances = torch.norm(points, dim=1)
    points = points[distances <= radius]
    return points

def create_view_matrix(camera_pos, look_at_point):
    """Create view matrix looking from camera_pos to look_at_point."""
    forward = look_at_point - camera_pos
    forward = forward / torch.norm(forward)
    
    right = torch.cross(forward, torch.tensor([0., 1., 0.], device=camera_pos.device))
    if torch.norm(right) < 1e-6:
        right = torch.cross(forward, torch.tensor([1., 0., 0.], device=camera_pos.device))
    right = right / torch.norm(right)
    up = torch.cross(right, forward)
    
    viewmat = torch.eye(4, device=camera_pos.device)
    viewmat[:3, 0] = right
    viewmat[:3, 1] = up
    viewmat[:3, 2] = forward
    viewmat[:3, 3] = camera_pos
    return torch.linalg.inv(viewmat)

class DelaunayRenderTest(parameterized.TestCase):
    def setUp(self):
        torch.manual_seed(189710234)
        self.height = 32
        self.width = 32
        
    def run_test(self, points, viewmat, tile_size):
        """Run rendering test with different sample counts and compare results."""
        indices = compute_delaunay(points)
        N = indices.shape[0]
        all = generate_color_palette(N*2)[:, :3]
        base_colors = all[:N].reshape(-1, 3)
        raw_grd = (2*all[N:].reshape(-1, 3)-1) / math.sqrt(3)
        tets = points[indices]
        circumcenters, _ = calculate_circumcenters_torch(tets)
        new_color, new_grd = offset_normalize(
            base_colors, raw_grd,
            circumcenters, tets)

        tet_density = torch.ones((len(indices),1), device='cuda')
        raw_cell_values = torch.cat([tet_density, new_color, new_grd.reshape(-1, 3)], dim=1)
        results = test_tetrahedra_rendering(
            points, indices, raw_cell_values, tet_density, viewmat,
            height=self.height, width=self.width, 
            tile_size=tile_size, n_samples=10000
        )
        
        results2 = test_tetrahedra_rendering(
            points, indices, raw_cell_values, tet_density, viewmat,
            height=self.height, width=self.width, 
            tile_size=tile_size, n_samples=5000
        )
        
        compare_dict_values(results, results2, key_pairs, points, viewmat, tile_size)
        return results
    
    @parameterized.product(
        n_points=[10, 20],
        radius=[10, 100],
        tile_size=[16]
    )
    def test_center_view(self, n_points, radius, tile_size, N=5):
        """Test rendering from center of point cloud with random rotation."""
        for i in range(N):
            # Generate point cloud and compute Delaunay tetrahedralization
            points = generate_point_cloud(n_points, radius)
            
            # Choose random center point
            center_idx = torch.randint(0, len(points), (1,)).item()
            center_point = points[center_idx]
            
            # Create view matrix with random rotation looking out from center
            viewmat = torch.eye(4).cuda()
            viewmat[:3, :3] = self._random_rotation_matrix()
            viewmat[:3, 3] = center_point
            viewmat = torch.linalg.inv(viewmat)
            
            self.run_test(points, viewmat, tile_size)
    
    @parameterized.product(
        n_points=[10, 20],
        radius=[10, 100],
        offset_mag=[1, 10, 100],
        tile_size=[16]
    )
    def test_outside_view(self, n_points, radius, offset_mag, tile_size, N=5):
        """Test rendering from outside the point cloud looking in."""
        for i in range(N):
            points = generate_point_cloud(n_points, radius)
            
            # Generate random direction and position camera
            direction = torch.randn(3, device=points.device)
            direction = direction / torch.norm(direction)
            camera_pos = direction * (radius + offset_mag)
            
            # Create view matrix looking at center
            center = torch.mean(points, dim=0)
            viewmat = create_view_matrix(camera_pos, center)
            
            self.run_test(points, viewmat, tile_size)
    
    def _random_rotation_matrix(self):
        """Generate random rotation matrix using quaternions."""
        q = np.random.randn(4)
        q = q / np.linalg.norm(q)
        quat = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
        return torch.tensor(quat.rotation_matrix, device='cuda').float()

if __name__ == '__main__':
    absltest.main()
