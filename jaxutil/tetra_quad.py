# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import math

import jax
import jax.numpy as jnp

from jaxutil import safe_math

import torch 
import numpy as np


def assert_valid_stepfun(t, y):
  """Assert that step function (t, y) has a valid shape."""
  if t.shape[-1] != y.shape[-1] + 1:
    raise ValueError(
        f'Invalid shapes ({t.shape}, {y.shape}) for a step function.'
    )

@jax.jit
def lossfun_distortion(t, w):
  """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
  assert_valid_stepfun(t, w)

  # The loss incurred between all pairs of intervals.
  ut = (t[Ellipsis, 1:] + t[Ellipsis, :-1]) / 2
  dut = jnp.abs(ut[Ellipsis, :, None] - ut[Ellipsis, None, :])
  loss_inter = jnp.sum(w * jnp.sum(w[Ellipsis, None, :] * dut, axis=-1), axis=-1)

  # The loss incurred within each individual interval with itself.
  loss_intra = jnp.sum(w**2 * jnp.diff(t), axis=-1) / 3

  return loss_inter + loss_intra


def log1mexp(x):
    """Accurate computation of log(1 - exp(-x)) for x > 0, thanks watsondaniel."""
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    return safe_math.safe_log(1-jnp.exp(-x))
    return jnp.where(x > jnp.log(2), jnp.log1p(-jnp.exp(-x)), jnp.log(-jnp.expm1(-x)))


def compute_alpha_weights_helper(density_delta):
    log_trans = -jnp.concatenate(
        [
            jnp.zeros_like(density_delta[..., :1]),
            jnp.cumsum(density_delta[..., :-1], axis=-1),
        ],
        axis=-1,
    )

    log_weights = log1mexp(density_delta) + log_trans
    # log_weights = log_trans
    weights = jnp.exp(log_weights)
    # weights = density_delta
    # weights = 1 - jnp.exp(-density_delta)
    return weights

def render_quadrature(tdist, query_fn, return_extras=False):
    """Numerical quadrature rendering of a set of colored Gaussians."""
    t_avg = 0.5 * (tdist[..., 1:] + tdist[..., :-1])
    t_delta = jnp.diff(tdist)
    total_density, avg_colors = query_fn(t_avg)
    weights = compute_alpha_weights_helper(total_density * t_delta)
    dist_loss = lossfun_distortion(tdist, weights)
    rendered_color = jnp.sum(
        weights[..., None] * avg_colors, axis=-2
    )  # Assuming the bg color is 0.
    alpha = jnp.sum(weights, axis=-1).reshape(-1, 1)
    rendered_color = jnp.concatenate([
        rendered_color.reshape(-1, 3), alpha, dist_loss.reshape(-1, 1)
    ], axis=1)

    if return_extras:
        return rendered_color, {
            "tdist": tdist,
            "avg_colors": avg_colors,
            "weights": weights,
            "total_density": jnp.sum(total_density*t_delta, axis=-1),
        }
    else:
        return rendered_color

def same_side(v1, v2, v3, v4, p):
    """
    Check if point p is on the same side of triangle (v1,v2,v3) as point v4.
    
    Args:
        v1, v2, v3: vertices of the triangle face
        v4: fourth vertex of tetrahedron
        p: query point
    Returns:
        bool indicating if p is on same side as v4
    """
    normal = jnp.cross(v2 - v1, v3 - v1)
    dot_v4 = jnp.dot(normal, v4 - v1)
    dot_p = jnp.dot(normal, p - v1)
    return jnp.sign(dot_v4) * jnp.sign(dot_p) > 0

def point_in_tetrahedron(vertices, tetrahedron, point):
    """
    Test if point is inside tetrahedron using same-side tests.
    
    Args:
        vertices: (N, 3) vertex positions
        tetrahedron: (4,) indices of tetrahedron vertices
        point: (3,) query point position
    Returns:
        bool indicating if point is inside
    """
    v1, v2, v3, v4 = [vertices[i] for i in tetrahedron]
    
    return (same_side(v1, v2, v3, v4, point) & 
            same_side(v2, v3, v4, v1, point) & 
            same_side(v3, v4, v1, v2, point) & 
            same_side(v4, v1, v2, v3, point))

@jax.jit
def query_tetrahedra_kernel(t_samples, ray_origins, ray_directions, 
                           vertices, tetrahedra, densities, colors):
    """
    Kernel function for querying tetrahedra field at sample points.
    
    Args:
        t_samples: (..., N) tensor of ray distances
        ray_origins: (..., 3) tensor of ray origins
        ray_directions: (..., 3) tensor of ray directions
        vertices: (V, 3) tensor of vertex positions
        tetrahedra: (T, 4) tensor of tetrahedra indices
        densities: (T,) tensor of density values per tetrahedra
        colors: (T, 3) tensor of RGB colors per tetrahedra
        
    Returns:
        total_density: (..., N) tensor of accumulated density at each sample
        avg_colors: (..., N, 3) tensor of interpolated colors at each sample
    """
    # Compute sample positions
    sample_points = ray_origins[..., None, :] + \
                   ray_directions[..., None, :] * t_samples[..., :, None]
    
    # Initialize outputs
    batch_shape = sample_points.shape[:-1]
    num_samples = batch_shape[-1]
    
    total_density = jnp.zeros(batch_shape)
    weighted_colors = jnp.zeros(batch_shape + (3,))
    
    def process_tetrahedron(i, accum):
        """Process single tetrahedron for all sample points."""
        density_acc, color_acc = accum
        
        # Get current tetrahedron data
        tet_indices = tetrahedra[i]
        tet_density = densities[i]
        tet_color = colors[i]
        
        is_inside = jax.vmap(lambda p: point_in_tetrahedron(vertices, tet_indices, p))(sample_points.reshape(-1, 3)).reshape(batch_shape)
        
        # Update density and color
        contrib_density = jnp.where(is_inside, tet_density, 0.0)
        contrib_color = jnp.where(
            is_inside[..., None],
            tet_color[None, None, :],
            0.0
        )
        
        return (
            density_acc + contrib_density,
            color_acc + contrib_color * contrib_density[..., None]
        )
    
    # Process all tetrahedra
    total_density, weighted_colors = jax.lax.fori_loop(
        0, len(tetrahedra),
        process_tetrahedron,
        (total_density, weighted_colors)
    )
    
    # Normalize colors by density
    avg_colors = jnp.where(
        total_density[..., None] > 1e-6,
        weighted_colors / total_density[..., None],
        0.0
    )
    
    return total_density, avg_colors

# @jax.jit
def render_tetrahedra_volume(ray_origins, ray_directions, tdist,
                            vertices, tetrahedra, densities, colors, return_extras=False):
    """
    Render volume defined by colored tetrahedra.
    
    Args:
        ray_origins: (..., 3) tensor of ray origins
        ray_directions: (..., 3) tensor of ray directions
        tdist: (..., N+1) tensor of sample distances
        vertices: (V, 3) tensor of vertex positions
        tetrahedra: (T, 4) tensor of tetrahedra indices
        densities: (T,) tensor of density values
        colors: (T, 3) tensor of RGB colors
    
    Returns:
        rendered_color: (..., 4) tensor of RGBA colors
        loss: (...) tensor of distortion loss
    """
    def query_fn(t_avg):
        return query_tetrahedra_kernel(
            t_avg, ray_origins, ray_directions,
            vertices, tetrahedra, densities, colors
        )
    
    return render_quadrature(tdist, query_fn, return_extras=return_extras)

def construct_camera_rays(viewmatrix, cam_pos, H, W, focal_x, focal_y):
    """Vectorized construction of camera rays for all pixels."""
    V = viewmatrix[:3, :3].T
    
    pixel_x, pixel_y = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
    
    ray_dirs = jnp.stack([
        (pixel_x + 0.5 - W / 2.0) / focal_x,
        (pixel_y + 0.5 - H / 2.0) / focal_y,
        jnp.ones_like(pixel_x)
    ], axis=-1)
    
    # Normalize rays
    ray_dirs = ray_dirs / jnp.linalg.norm(ray_dirs, axis=-1, keepdims=True)
    # Transform by view matrix
    ray_dirs = jnp.einsum('ij,hwj->hwi', V, ray_dirs)
    
    # Broadcast camera position to match ray directions
    ray_origins = jnp.broadcast_to(cam_pos, ray_dirs.shape)
    
    return ray_origins, ray_dirs

@functools.partial(jax.jit, static_argnums=(3, 4))
def render_camera(vertices, indices, rgbs, height, width, viewmat, fx, fy, linspace):
    """Vectorized camera renderer using JAX."""
    # Extract camera position and convert inputs
    cam_pos = jnp.linalg.inv(viewmat)[:3, 3]
    densities = rgbs[:, 3].reshape(-1, 1)
    colors = rgbs[:, :3]
    
    # Generate sample points
    # point_dist = jnp.linalg.norm(cam_pos.reshape(1, 3) - vertices, axis=1)
    point_dist = viewmat @ jnp.concatenate([vertices.T, jnp.ones_like(vertices.T[:1, :])], axis=0)
    # test_pts = point_dist[:3] / point_dist[2:3]
    # jax.debug.print("NDC: {}\ncam_space: {}", test_pts, point_dist)
    
    start = jnp.clip(point_dist[2].min(), 0, None)
    end = point_dist[2].max()*1.2
    tdist = (end - start) * linspace + start
    
    # Generate rays for all pixels
    ray_origins, ray_directions = construct_camera_rays(
        viewmat, cam_pos, height, width, fx, fy
    )
    
    # Vectorize rendering over all pixels
    batched_render = jax.vmap(
        jax.vmap(
            lambda o, d: render_tetrahedra_volume(
                o[None], d[None], tdist,
                vertices, indices, densities, colors,
                return_extras=True
            ),
            in_axes=(0, 0)
        ),
        in_axes=(0, 0)
    )
    
    image, extras = batched_render(ray_origins, ray_directions)
    return image[..., 0, :], extras

# def construct_camera_ray(viewmatrix, cam_pos, pixel_x, pixel_y, W, H, focal_x, focal_y):
#     """Construct camera rays for given pixels."""
#     V = viewmatrix[:3, :3].T
    
#     ray_dir = torch.tensor([
#         (pixel_x + 0.5 - W / 2.0) / focal_x,
#         (pixel_y + 0.5 - H / 2.0) / focal_y,
#         1.0
#     ], device=cam_pos.device).float()
    
#     ray_dir = ray_dir / torch.norm(ray_dir)
#     ray_dir = V @ ray_dir
    
#     return cam_pos, ray_dir

# def render_camera(vertices, indices, rgbs, 
#                   viewmat, height, width, fx, fy):
    
#     cam_pos = viewmat.inverse()[:3, 3]
    
#     # Render using JAX implementation
#     tdist = np.linspace(0, vertices.abs().max().item(), 10000)
#     densities = rgbs[:, 3].detach().cpu().numpy()
#     colors = rgbs[:, :3].detach().cpu().numpy()
#     ray_origins = cam_pos.cpu().numpy().reshape(1, -1)

#     pixels_x, pixels_y = np.meshgrid(np.arange(height), np.arange(width))
#     jax_image = np.zeros((height, width, 3))
    
#     for i in range(height):
#         for j in range(width):
#             _, ray_directions = construct_camera_ray(
#                 viewmat, cam_pos, pixels_x[i, j], pixels_y[i, j], 
#                 width, height, fx, fy
#             )
#             ray_directions = ray_directions.cpu().numpy()
#             color, extras = render_tetrahedra_volume(
#                 ray_origins, 
#                 ray_directions, 
#                 tdist,
#                 vertices.cpu().numpy(), 
#                 indices.cpu().numpy(), 
#                 densities.reshape(-1, 1), 
#                 colors, 
#                 return_extras=True
#             )
#             jax_image[i, j] = color[0, :3]
#     return jax_image