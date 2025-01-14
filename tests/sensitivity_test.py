import torch
import numpy as np
import unittest

from utils import topo_utils
from gDel3D.build.gdel3d import Del
from torch.autograd.functional import jacobian
from icecream import ic

def compute_circumcenter_shift(num_vertices=10, perturbation=0.001, device="cuda"):
    torch.manual_seed(np.random.randint(0, 10000))  # Different seed for each run
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Generate random vertices
    vertices = torch.rand((num_vertices, 3), device=device, dtype=torch.float32)

    # Compute Delaunay triangulation
    del_obj = Del(vertices.shape[0])
    indices_np = del_obj.compute(vertices.detach().cpu()).numpy()
    indices_np = indices_np[(indices_np < vertices.shape[0]).all(axis=1)]
    indices = torch.as_tensor(indices_np.astype(np.int32), device=device)

    # Compute initial circumcenters (NumPy)
    circumcenters_before, _ = topo_utils.calculate_circumcenters(vertices.detach().cpu().numpy()[indices_np])

    # Compute vertex sensitivity
    with torch.no_grad():
        sensitivity = topo_utils.compute_vertex_sensitivity(indices, vertices)
        # scaling = (1 / (sensitivity+1e-5)).detach().cpu().numpy()
        scaling = (1/(sensitivity.reshape(-1, 1)+1e-5)).clip(max=1)

    # Perturb the vertices
    direction = np.random.randn(*vertices.shape)  # Random movement direction
    direction = direction / (np.linalg.norm(direction, axis=-1, keepdims=True)+1e-5)
    movement = perturbation * scaling.detach().cpu().numpy() * direction
    moved_vertices = vertices.detach().cpu().numpy() + movement

    # Compute new circumcenters without updating triangulation
    circumcenters_after, _ = topo_utils.calculate_circumcenters(moved_vertices[indices_np])

    # Compute movement magnitude
    movement_magnitude = np.linalg.norm(circumcenters_after - circumcenters_before, axis=1)

    return movement_magnitude, vertices, movement

def test_jacobian(vertices):
    # Compute analytical Jacobian
    jacobian_analytical = topo_utils.circumcenter_jacobian(vertices)

    # Compute numerical Jacobian using PyTorch autograd
    def wrapper(v):
        return topo_utils.calculate_circumcenters_torch(v)[0]  # Only get circumcenters

    jacobian_numerical = jacobian(wrapper, vertices).squeeze(0)  # (3, 4, 3)

    # Compare
    print("Analytical Jacobian:")
    print(jacobian_analytical)
    print(jacobian_analytical.shape)

    print("\nNumerical Jacobian:")
    print(jacobian_numerical)
    print(jacobian_numerical.shape)

    # Compute error between the two
    error = torch.norm(jacobian_analytical - jacobian_numerical)
    print(f"\nJacobian Difference (Frobenius norm): {error.item()}")

    assert torch.allclose(jacobian_analytical, jacobian_numerical, atol=1e-4), "Jacobian mismatch!"


class TestCircumcenterSensitivity(unittest.TestCase):
    # def test_jacobian(self):
    #     # Generate a random tetrahedron (batch size = 1)
    #     vertices = torch.rand((1, 4, 3), dtype=torch.float32, requires_grad=True)
    #     test_jacobian(vertices)

    def test_circumcenter_shift(self, num_vertices=10, perturbation=0.01):
        for i in range(50):
            movement_magnitude, vertices, movement = compute_circumcenter_shift(num_vertices, perturbation)

            if not np.all(movement_magnitude < perturbation):
                failing_indices = np.where(movement_magnitude >= perturbation)[0]
                error_message = (
                    f"Run {i+1} failed: Some circumcenters moved too much.\n"
                    f"Exceeded limit at indices: {failing_indices}\n"
                    f"vertices = torch.{repr(vertices)}\n"
                    f"movement = torch.{repr(movement)}\n"
                    f"movement_magnitude = torch.{repr(movement_magnitude.reshape(-1, 1))}\n"
                    f"Min: {movement_magnitude.min()}, Max: {movement_magnitude.max()}, Mean: {movement_magnitude.mean()}"
                )
                self.fail(error_message)


if __name__ == '__main__':
    unittest.main()
