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
def lossfun_distortion(t, mass):
  """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
  w = mass / jnp.clip(mass.sum(axis=-1), min=1e-8, max=None)
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


def tukey_power_ladder(x, p):
    """Port of tukey_power_ladder in alphablend_shader_interp.slang."""
    xs = jnp.abs(x) / jnp.maximum(1e-20, jnp.abs(p - 1))
    return jnp.sign(x) * jnp.abs(p - 1) / p * ((xs + 1) ** p - 1)


def scale_dist(x):
    """Port of scale_dist in alphablend_shader_interp.slang."""
    return tukey_power_ladder(160 * x, 0.1)


def _warped_dist_loss_1d(s, sigma, dt, tet_id):
    """Per-ray port of the slang DistortionState5 recursion.

    The rasterizer's distortion loss is a *per-primitive* functional: one
    exact-moment term per ray-tet segment, with sigma acting on the warped
    segment width and the raw-space transmittance frozen at segment entry.
    It is NOT the limit of a per-sample rule, so plain quadrature cannot
    converge to it. Since Delaunay tets don't overlap, each sample belongs
    to at most one tet; we recover the segmentation by grouping runs of
    consecutive samples with the same tet id (NOT equal sigma — adjacent
    tets often share a density value) and apply the shader's exact moments
    per segment:
        m0   = T (1 - e^{-u}),                u = sigma (s_b - s_a)
        m1   = T (s_a alpha + (1-(1+u)e^{-u})/sigma)
        self = T^2 (1 - 2u e^{-u} - e^{-2u}) / sigma
        x += 2 m1 Z_prev ; y += 2 m0 W_prev ; Z += m0 ; v += self
        loss = relu(x - y + v) / max(Z^2, 1e-6)   (utils/train_util.py)
    """
    n = sigma.shape[0]
    od_raw = sigma * dt
    # Raw-space transmittance at each sample's left edge.
    T_edge = jnp.exp(-jnp.concatenate([jnp.zeros(1), jnp.cumsum(od_raw[:-1])]))

    # Run-length segmentation of same-tet samples (ordered along the ray).
    boundary = jnp.concatenate([jnp.ones(1, bool), tet_id[1:] != tet_id[:-1]])
    seg = jnp.cumsum(boundary) - 1
    s_a = jax.ops.segment_min(s[:-1], seg, num_segments=n)
    s_b = jax.ops.segment_max(s[1:], seg, num_segments=n)
    sg = jax.ops.segment_max(sigma, seg, num_segments=n)
    Ts = jax.ops.segment_max(T_edge, seg, num_segments=n)  # T is decreasing

    # Empty trailing segment ids get +/-inf identity values from the segment
    # reductions; sanitize them so they cannot poison gradients (0 * inf = NaN
    # in the where-VJP) before any arithmetic touches them.
    present = jax.ops.segment_sum(jnp.ones_like(sigma), seg, num_segments=n) > 0
    s_a = jnp.where(present, s_a, 0.0)
    s_b = jnp.where(present, s_b, 0.0)
    sg = jnp.where(present, sg, 0.0)
    Ts = jnp.where(present, Ts, 1.0)

    # The shader terminates the ray once accumulated log-transmittance drops
    # below -5.5, so tets starting beyond that contribute nothing to its
    # distortion state. Replicate the cutoff (a tet is processed iff T at its
    # start is above the threshold; the crossing tet itself is included).
    valid = (sg > 0) & (Ts > jnp.exp(-5.5))  # also masks empty segments
    safe_sg = jnp.where(valid, sg, 1.0)
    u = jnp.where(valid, sg * (s_b - s_a), 0.0)
    e = jnp.exp(-u)
    alpha = 1.0 - e
    m0 = jnp.where(valid, Ts * alpha, 0.0)
    m1 = jnp.where(valid, Ts * (s_a * alpha + (1 - (1 + u) * e) / safe_sg), 0.0)
    self_v = jnp.where(valid, Ts * Ts * (1 - 2 * u * e - e * e) / safe_sg, 0.0)

    Z_prev = jnp.concatenate([jnp.zeros(1), jnp.cumsum(m0[:-1])])
    W_prev = jnp.concatenate([jnp.zeros(1), jnp.cumsum(m1[:-1])])
    x = jnp.sum(2 * m1 * Z_prev)
    y = jnp.sum(2 * m0 * W_prev)
    v = jnp.sum(self_v)
    Z = jnp.sum(m0)
    return jnp.maximum(x - y + v, 0.0) / jnp.maximum(Z * Z, 1e-6)


def warped_dist_loss(tdist, total_density, tmin, tet_ids):
    """Reference for the rasterizer's distortion loss (see _warped_dist_loss_1d).

    t is distance along the unit-norm ray from the camera origin; the
    shader's rays start at origin + min_t * dir, hence the tmin shift before
    warping with scale_dist.
    """
    s = scale_dist(tdist - tmin)
    dt = jnp.diff(tdist)
    batch = total_density.shape[:-1]
    f = jnp.vectorize(_warped_dist_loss_1d, signature='(m),(n),(n),(n)->()')
    return f(jnp.broadcast_to(s, batch + s.shape[-1:]),
             total_density,
             jnp.broadcast_to(dt, total_density.shape),
             jnp.broadcast_to(tet_ids, total_density.shape))


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

def _termination_mask_1d(logT_edge, tet_id):
    """Per-sample mask replicating the rasterizer's early ray termination.

    The shader stops after the tet whose blend pushes log T below -5.5; a tet
    is processed iff log T at its entry is >= -5.5 (the crossing tet is fully
    included). Group samples into tet segments and keep those whose
    segment-entry log T is above the threshold.
    """
    n = tet_id.shape[0]
    boundary = jnp.concatenate([jnp.ones(1, bool), tet_id[1:] != tet_id[:-1]])
    seg = jnp.cumsum(boundary) - 1
    seg_entry_logT = jax.ops.segment_max(logT_edge, seg, num_segments=n)
    seg_entry_logT = jnp.where(jnp.isfinite(seg_entry_logT), seg_entry_logT, 0.0)
    keep = seg_entry_logT >= -5.5
    return keep[seg].astype(logT_edge.dtype)


def termination_mask(tdist, total_density, tet_ids):
    dt = jnp.diff(tdist)
    od = total_density * dt
    logT_edge = -jnp.concatenate(
        [jnp.zeros_like(od[..., :1]), jnp.cumsum(od[..., :-1], axis=-1)], axis=-1)
    f = jnp.vectorize(_termination_mask_1d, signature='(n),(n)->(n)')
    return f(logT_edge, jnp.broadcast_to(tet_ids, od.shape))


def render_quadrature(tdist, query_fn, return_extras=False, dist_tmin=0.0):
    """Numerical quadrature rendering of a set of colored Gaussians."""
    t_avg = 0.5 * (tdist[..., 1:] + tdist[..., :-1])
    t_delta = jnp.diff(tdist)
    total_density, avg_colors, tet_ids = query_fn(t_avg)
    weights = compute_alpha_weights_helper(total_density * t_delta)
    # Replicate the rasterizer's early termination (non-differentiable cutoff,
    # like the shader's loop break) so gradients on near-threshold rays match.
    weights = weights * jax.lax.stop_gradient(
        termination_mask(tdist, total_density, tet_ids))
    dist_loss = warped_dist_loss(tdist, total_density, dist_tmin, tet_ids)
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
            "dist_loss": dist_loss,
            # "total_density": jnp.sum(total_density*t_delta, axis=-1),
            "total_density": total_density,
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

def barycentric_coordinates_matrix(p, a, b, c, d):
    T = jnp.array([a - d, b - d, c - d]).T
    v = jnp.linalg.solve(T, p - d)
    u, v, w = v
    t = 1 - u - v - w
    return jnp.array([u, v, w, t])

def _ray_tet_interval(o, d, tet_v, eps=1e-10):
    """Analytic ray-tet intersection interval via face-plane clipping.

    Mirrors ray_tetrahedron_intersect_fused in intersect.slang (outward
    normals; entering planes raise t_enter, exiting planes lower t_exit;
    near-parallel planes reject when the origin is outside).

    Args:  o, d: (..., 3) ray origins/directions; tet_v: (4, 3)
    Returns: t_enter, t_exit: (...) — empty interval when t_enter >= t_exit.
    """
    faces = jnp.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    opposite = jnp.array([3, 2, 1, 0])
    p0 = tet_v[faces[:, 0]]  # (4, 3)
    n = jnp.cross(tet_v[faces[:, 1]] - p0, tet_v[faces[:, 2]] - p0)  # (4, 3)
    flip = jnp.where(jnp.sum(n * (tet_v[opposite] - p0), -1, keepdims=True) > 0, -1.0, 1.0)
    n = n * flip  # outward

    denom = jnp.einsum('...k,fk->...f', d, n)               # (..., 4)
    # dot(n, o - p0): subtract BEFORE the dot product, like the slang shader
    # (d = -dot(n, v0 - orig)). Subtracting two large dot products instead
    # loses ~4 digits to cancellation at scene scales of ~100.
    dist = jnp.sum(n * (o[..., None, :] - p0), -1)          # (..., 4)

    parallel = jnp.abs(denom) < eps
    tplane = -dist / jnp.where(parallel, 1.0, denom)
    t_enter = jnp.max(jnp.where(~parallel & (denom < 0), tplane, -jnp.inf), axis=-1)
    t_exit = jnp.min(jnp.where(~parallel & (denom > 0), tplane, jnp.inf), axis=-1)
    # Parallel to a plane while outside it: no intersection.
    miss = jnp.any(parallel & (dist > 0), axis=-1)
    t_enter = jnp.where(miss, jnp.inf, t_enter)
    return t_enter, t_exit


@jax.jit
def query_tetrahedra_kernel(t_samples, ray_origins, ray_directions,
                           vertices, tetrahedra, densities, vertex_color):
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
    tet_ids = jnp.zeros(batch_shape)  # 0 = empty space, i+1 = inside tet i

    def process_tetrahedron(i, accum):
        """Process single tetrahedron for all sample points."""
        density_acc, color_acc, id_acc = accum

        # Get current tetrahedron data
        tet_indices = tetrahedra[i]
        tet_density = densities[i]

        # Inclusion via analytic ray-interval clipping against the 4 face
        # planes (same construction as ray_tetrahedron_intersect_fused in
        # intersect.slang). A per-sample point-in-tet test flickers in/out
        # along grazing rays (float roundoff near faces), fragmenting the
        # tet into fake far-apart segments that corrupt the distortion
        # reference; a convex tet intersects a ray in ONE interval.
        tet_v = vertices[tet_indices]  # (4, 3)
        t_enter, t_exit = _ray_tet_interval(ray_origins, ray_directions, tet_v)

        is_inside = ((t_samples > t_enter[..., None]) &
                     (t_samples < t_exit[..., None])).reshape(batch_shape)

        # Color model identical to integrate_tetra_segment in
        # interp_version.slang: evaluate c(p) = base + dot(grd, p - v0) at the
        # segment's ENTER and EXIT points, clamp each at 0, and interpolate
        # linearly in t between them. (The shader's
        # compute_integral(c_exit, c_enter, ...) is exactly the density-
        # weighted integral of that endpoint-clamped linear ramp, so a
        # per-sample clamp would diverge wherever the ramp crosses zero
        # mid-segment.)
        grd = vertex_color[i, 3:6]
        base = vertex_color[i, 0:3]
        v0 = tet_v[0]
        t_en = jnp.maximum(t_enter, 0.0)[..., None]          # ray batch dims
        t_ex = t_exit[..., None]
        p_en = ray_origins[..., None, :] + ray_directions[..., None, :] * t_en[..., None]
        p_ex = ray_origins[..., None, :] + ray_directions[..., None, :] * t_ex[..., None]
        c_en = jnp.maximum(base + jnp.sum(grd * (p_en - v0), -1, keepdims=True), 0.0)
        c_ex = jnp.maximum(base + jnp.sum(grd * (p_ex - v0), -1, keepdims=True), 0.0)
        frac = (t_samples - t_en) / jnp.maximum(t_ex - t_en, 1e-12)
        tet_color = (c_en + (c_ex - c_en) * frac[..., None]).reshape(batch_shape + (3,))

        # Update density and color
        contrib_density = jnp.where(is_inside, tet_density, 0.0)
        contrib_color = jnp.where(
            is_inside[..., None],
            tet_color,
            0.0
        )
        
        return (
            density_acc + contrib_density,
            color_acc + contrib_color * contrib_density[..., None],
            jnp.where(is_inside, jnp.float32(i + 1), id_acc),
        )

    # Process all tetrahedra
    total_density, weighted_colors, tet_ids = jax.lax.fori_loop(
        0, len(tetrahedra),
        process_tetrahedron,
        (total_density, weighted_colors, tet_ids)
    )

    # Normalize colors by density
    avg_colors = jnp.where(
        total_density[..., None] > 1e-6,
        weighted_colors / total_density[..., None],
        0.0
    )

    return total_density, avg_colors, tet_ids

# @jax.jit
def render_tetrahedra_volume(ray_origins, ray_directions, tdist,
                            vertices, tetrahedra, densities, vertex_color,
                            return_extras=False, dist_tmin=0.0):
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
            vertices, tetrahedra, densities, vertex_color
        )
    
    return render_quadrature(tdist, query_fn, return_extras=return_extras,
                             dist_tmin=dist_tmin)

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

@functools.partial(jax.jit, static_argnums=(4, 5))
def render_camera(vertices, indices, vertex_color, tet_density, height, width, viewmat, fx, fy, tmin, linspace):
    """Vectorized camera renderer using JAX."""
    # Extract camera position and convert inputs
    cam_pos = jnp.linalg.inv(viewmat)[:3, 3]
    
    # Generate sample points
    # point_dist = jnp.linalg.norm(cam_pos.reshape(1, 3) - vertices, axis=1)
    point_dist = viewmat @ jnp.concatenate([vertices.T, jnp.ones_like(vertices.T[:1, :])], axis=0)
    # test_pts = point_dist[:3] / point_dist[2:3]
    # jax.debug.print("NDC: {}\ncam_space: {}", test_pts, point_dist)
    
    start = jnp.clip(point_dist[2].min(), tmin, None)
    # tdist is RAY DISTANCE, not depth: slanted rays need up to
    # z / cos(theta) > z * 1.2 at wide FOV, which truncated geometry for
    # off-axis pixels. The farthest point of the convex hull from the camera
    # is a vertex, so the max vertex distance bounds every ray segment.
    vert_dist = jnp.linalg.norm(vertices - cam_pos.reshape(1, 3), axis=1)
    end = jnp.maximum(point_dist[2].max() * 1.2, vert_dist.max() * 1.05)
    # jax.debug.print("start: {} - {}", start, end)
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
                vertices, indices, tet_density.reshape(-1, 1), vertex_color,
                return_extras=True, dist_tmin=tmin
            ),
            in_axes=(0, 0)
        ),
        in_axes=(0, 0)
    )
    
    image, extras = batched_render(ray_origins, ray_directions)
    return image[..., 0, :], extras

@functools.partial(jax.jit)
def render_rays(vertices, indices, vertex_color, tet_density, ray_origins, ray_directions, viewmat, tmin, linspace):
    # ray_origins, ray_directions: (N, 3)
    """Vectorized camera renderer using JAX."""
    # Extract camera position and convert inputs
    cam_pos = jnp.linalg.inv(viewmat)[:3, 3]
    
    # Generate sample points
    # point_dist = jnp.linalg.norm(cam_pos.reshape(1, 3) - vertices, axis=1)
    point_dist = viewmat @ jnp.concatenate([vertices.T, jnp.ones_like(vertices.T[:1, :])], axis=0)
    # test_pts = point_dist[:3] / point_dist[2:3]
    # jax.debug.print("NDC: {}\ncam_space: {}", test_pts, point_dist)
    
    start = tmin
    end = point_dist[2].max()*1.2
    # jax.debug.print("start: {} - {}", start, end)
    tdist = (end - start) * linspace + start
    
    # Vectorize rendering over all pixels
    batched_render = jax.vmap(
        jax.vmap(
            lambda o, d: render_tetrahedra_volume(
                o[None], d[None], tdist,
                vertices, indices, tet_density.reshape(-1, 1), vertex_color,
                return_extras=True, dist_tmin=tmin
            ),
            in_axes=(0, 0)
        ),
        in_axes=(0, 0)
    )
    
    image, extras = batched_render(ray_origins, ray_directions)
    return image[..., 0, :], extras
