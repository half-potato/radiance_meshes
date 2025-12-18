import functools
import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import slangpy
import torch
from jax import random
from pathlib import Path

# fn = slangpy.loadModule(str(Path(__file__).parent / "tri_splinetracer/slang/spline-machine-py.slang"))
fn = slangpy.loadModule(str(Path(__file__).parent / "tri_splinetracer/slang/spline-machine-py.slang"))

np.set_printoptions(edgeitems=30, linewidth=100000)


def log1mexp(x):
    """Accurate computation of log(1 - exp(-x)) for x > 0, thanks watsondaniel."""
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
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


def gaussian_kernel_fn(x, std):
    """A Gaussian kernel (eg, a non-normalized Gaussian PDF."""
    return jnp.exp(-0.5 * (x / std) ** 2)


def triangular_kernel_fn(x, std):
    """A triangular kernel that matches the standard dev. of a Gaussian."""
    # return jnp.maximum(0, 1 - jnp.abs(x) / (jnp.sqrt(6) * std))
    return jnp.maximum(0, 1 - jnp.abs(x) / (std))


gaussian_kernel_fn = triangular_kernel_fn  # UNDO


def query_gaussians(
    tdist,
    means,
    stds,
    heights,
    colors,
    kernel_fn=gaussian_kernel_fn,
    eps=jnp.finfo(jnp.float64).eps,
):
    # Evaluate each gaussian at each query point.
    densities = heights[..., None] * kernel_fn(
        tdist - means[..., None], stds[..., None]
    )

    # The total density of each quadrature bin is the sum of all densities that
    # contribute to it.
    total_density = jnp.sum(densities, axis=-2)

    # The color of each interval is the weighted combination of each Gaussian's
    # color, where the weighting is the *density* of each Gaussian at the center
    # of the interval.
    avg_colors = jnp.sum(
        densities[..., None] * colors[..., None, :], axis=-3
    ) / jnp.maximum(eps, total_density[..., None])

    return total_density, avg_colors


def query_splines(tdist, t_spline, d_spline, rgb_spline):
    interp = functools.partial(jnp.interp, left=0, right=0)
    total_density = jnp.vectorize(interp, signature="(n),(m),(m)->(n)")(
        tdist, t_spline, d_spline
    )

    # Sloppy.
    interp = jnp.interp
    r = jnp.vectorize(interp, signature="(n),(m),(m)->(n)")(
        tdist, t_spline, rgb_spline[..., 0]
    )
    g = jnp.vectorize(interp, signature="(n),(m),(m)->(n)")(
        tdist, t_spline, rgb_spline[..., 1]
    )
    b = jnp.vectorize(interp, signature="(n),(m),(m)->(n)")(
        tdist, t_spline, rgb_spline[..., 2]
    )
    avg_color = jnp.stack([r, g, b], axis=-1)
    return total_density, avg_color


def render_quadrature(tdist, query_fn, return_extras=False):
    """Numerical quadrature rendering of a set of colored Gaussians."""
    t_avg = 0.5 * (tdist[..., 1:] + tdist[..., :-1])
    t_delta = jnp.diff(tdist)
    total_density, avg_colors = query_fn(t_avg)
    weights = compute_alpha_weights_helper(total_density * t_delta)
    rendered_color = jnp.sum(
        weights[..., None] * avg_colors, axis=-2
    )  # Assuming the bg color is 0.

    if return_extras:
        return rendered_color, {
            "tdist": tdist,
            "avg_colors": avg_colors,
            "weights": weights,
            "total_density": jnp.sum(total_density*t_delta, axis=-1),
        }
    else:
        return rendered_color


def render_spline(
    means,
    stds,
    heights,
    colors,
    t_range,
    return_extras=False,
    eps=jnp.finfo(jnp.float64).eps,
):
    """Spline rendering of a set of colored Gaussians."""
    dt = stds
    t_cat = jnp.concatenate([means - dt, means, means + dt], axis=-1)
    # zdt = heights / jnp.maximum(eps, dt)
    # colors1 = jnp.concatenate([jnp.ones_like(colors[..., :1]), colors], axis=-1)
    # drgbs_cat = jnp.concatenate([zdt, -2*zdt, zdt], axis=-1)[..., None] * jnp.concatenate([colors1]*3, axis=-2)

    zdt = 1 / jnp.maximum(eps, dt)
    colors1 = jnp.concatenate(
        [heights[..., None], heights[..., None] * colors], axis=-1
    )
    drgbs_cat = jnp.concatenate([zdt, -2 * zdt, zdt], axis=-1)[
        ..., None
    ] * jnp.concatenate([colors1] * 3, axis=-2)

    # Sort them by `t`.
    idx = jnp.argsort(t_cat, axis=-1)
    t_spline = jnp.take_along_axis(t_cat, idx, axis=-1)
    drgb_spline_diracs = jnp.take_along_axis(drgbs_cat, idx[..., None], axis=-2)

    y = jnp.cumsum(
        jnp.diff(t_spline)[..., None]
        * jnp.cumsum(drgb_spline_diracs[..., :-1, :], axis=-2),
        axis=-2,
    )
    drgb_spline = jnp.concatenate(
        [jnp.zeros_like(y[..., :1, :]), y[..., :-1, :], jnp.zeros_like(y[..., -1:, :])],
        axis=-2,
    )

    # Because color and density are both >= 0, the summed spline should be >= 0.
    drgb_spline = jnp.maximum(0, drgb_spline)

    # Divide the weighted colors by density, while padding the denominator
    # by 2*eps and the numerator by eps (so that 0.5 is the default color),
    # and clipping the output to guard against weird results when the numerical
    # instability of the cumulative sum causes some funkiness.
    d_spline = drgb_spline[..., 0]
    rgb_spline = jnp.clip(
        (eps + drgb_spline[..., 1:]) / (2 * eps + d_spline[..., None]), 0, 1
    )

    # Zero out the first and last density of every spline, as they can only be
    # non-zero due to numerical weirdness.
    z = jnp.zeros_like(d_spline[..., -1:])
    d_spline = jnp.concatenate([z, d_spline[..., 1:-1], z], axis=-1)
    # If a spline knot has a zero density (which happens at the beginning and the
    # end of every spline, and often many times in the middle of the tents
    # don't overlap) then we should fill in the colors of the zero-density knots
    # with their neighboring non-zero density value. This is necessary only
    # for clamping.
    d_spline_pad = jnp.concatenate(
        [
            -jnp.ones_like(d_spline[..., :1]),
            d_spline,
            -jnp.ones_like(d_spline[..., :1]),
        ],
        axis=-1,
    )
    rgb_spline_pad = jnp.concatenate(
        [
            jnp.zeros_like(rgb_spline[..., :1, :]),
            rgb_spline,
            jnp.zeros_like(rgb_spline[..., :1, :]),
        ],
        axis=-2,
    )
    rgb_spline = jnp.where(
        (d_spline > 0)[..., None],
        rgb_spline,
        jnp.where(
            (d_spline_pad[..., 2:] > d_spline)[..., None],
            rgb_spline_pad[..., 2:, :],
            rgb_spline_pad[..., :-2, :],
        ),
    )

    interp = functools.partial(jnp.interp, left=0, right=0)
    d_min = jnp.vectorize(interp, signature="(),(m),(m)->()")(
        t_range[0], t_spline, d_spline
    )
    d_max = jnp.vectorize(interp, signature="(),(m),(m)->()")(
        t_range[1], t_spline, d_spline
    )

    # This is really sloppy, figure out how to properly vectorize.
    interp = functools.partial(jnp.interp)
    r_min = jnp.vectorize(interp, signature="(),(m),(m)->()")(
        t_range[0], t_spline, d_spline * rgb_spline[..., 0]
    )
    r_max = jnp.vectorize(interp, signature="(),(m),(m)->()")(
        t_range[1], t_spline, d_spline * rgb_spline[..., 0]
    )
    g_min = jnp.vectorize(interp, signature="(),(m),(m)->()")(
        t_range[0], t_spline, d_spline * rgb_spline[..., 1]
    )
    g_max = jnp.vectorize(interp, signature="(),(m),(m)->()")(
        t_range[1], t_spline, d_spline * rgb_spline[..., 1]
    )
    b_min = jnp.vectorize(interp, signature="(),(m),(m)->()")(
        t_range[0], t_spline, d_spline * rgb_spline[..., 2]
    )
    b_max = jnp.vectorize(interp, signature="(),(m),(m)->()")(
        t_range[1], t_spline, d_spline * rgb_spline[..., 2]
    )
    rgb_min = jnp.stack([r_min, g_min, b_min], axis=-1) / jnp.maximum(
        eps, d_min[..., None]
    )
    rgb_max = jnp.stack([r_max, g_max, b_max], axis=-1) / jnp.maximum(
        eps, d_max[..., None]
    )

    t_below = t_spline < t_range[0]
    t_spline = jnp.where(t_below, t_range[0], t_spline)
    d_spline = jnp.where(t_below, d_min[..., None], d_spline)
    rgb_spline = jnp.where(t_below[..., None], rgb_min[..., None, :], rgb_spline)

    t_above = t_spline > t_range[1]
    t_spline = jnp.where(t_above, t_range[1], t_spline)
    d_spline = jnp.where(t_above, d_max[..., None], d_spline)
    rgb_spline = jnp.where(t_above[..., None], rgb_max[..., None, :], rgb_spline)

    # https://arxiv.org/abs/2310.20685 says that we can just average the densities
    # for a piecewise linear spline and use that as the density of the bucket.
    d_avg = 0.5 * (d_spline[..., 1:] + d_spline[..., :-1])
    t_delta = jnp.diff(t_spline, axis=-1)

    # THS IS WRONG: Ya gotta derive the average color in the interval as a
    # function of a piecewise linear density and the colors of the knots. Seems
    # hard to do without considering how visible the interval is currently?
    # Probably needs to get rolled into the computation of weights and
    # rendered_color.
    drgb_avg = 0.5 * (
        d_spline[..., 1:, None] * rgb_spline[..., 1:, :]
        + d_spline[..., :-1, None] * rgb_spline[..., :-1, :]
    )
    avg_colors = drgb_avg / jnp.maximum(eps, d_avg[..., None])

    # This isn't right:
    # avg_colors = integrate_colors(d_spline[..., :-1, None], d_spline[..., 1:, None], rgb_spline[..., :-1, :], rgb_spline[..., 1:, :])

    weights = compute_alpha_weights_helper(d_avg * t_delta)
    rendered_color = jnp.sum(weights[..., None] * avg_colors, axis=-2)

    # get the spline of densities and colors and plot it against the quadrature rendering, confirm they're the same. The issue might be in how colors are mixed.

    if return_extras:
        return rendered_color, {
            "avg_colors": avg_colors,
            "t_spline": t_spline,
            "d_spline": d_spline,
            "rgb_spline": rgb_spline,
            "weights": weights,
        }
    else:
        return rendered_color


def slang_render_spline(
    means,
    stds,
    heights,
    colors,
    t_range,
    return_extras=False,
    eps=jnp.finfo(jnp.float64).eps,
):
    t_min, t_max = t_range
    """Spline rendering of a set of colored Gaussians."""
    dt = stds
    t_cat = jnp.concatenate([means - dt, means, means + dt], axis=-1)
    # zdt = heights / jnp.maximum(eps, dt)
    # colors1 = jnp.concatenate([jnp.ones_like(colors[..., :1]), colors], axis=-1)
    # drgbs_cat = jnp.concatenate([zdt, -2*zdt, zdt], axis=-1)[..., None] * jnp.concatenate([colors1]*3, axis=-2)

    zdt = 1 / jnp.maximum(eps, dt)
    colors1 = jnp.concatenate(
        [heights[..., None], heights[..., None] * colors], axis=-1
    )
    drgbs_cat = jnp.concatenate([zdt, -2 * zdt, zdt], axis=-1)[
        ..., None
    ] * jnp.concatenate([colors1] * 3, axis=-2)

    # Sort them by `t`.
    idx = jnp.argsort(t_cat, axis=-1)
    t_spline = jnp.take_along_axis(t_cat, idx, axis=-1)
    drgb_spline_diracs = jnp.take_along_axis(drgbs_cat, idx[..., None], axis=-2)

    spline_color = np.zeros((idx.shape[0], 3))
    weights = np.zeros((idx.shape[0], idx.shape[1]))
    d_spline = np.zeros((idx.shape[0], idx.shape[1]))
    avg_colors = np.zeros((idx.shape[0], idx.shape[1], 3))
    total_density = np.zeros((idx.shape[0]))
    for i in range(idx.shape[0]):
        state = [
            # float t;
            0,
            # float4 drgb;
            [0, 0, 0, 0],
            # float4 d_drgb;
            [0, 0, 0, 0],
            # float logT;
            0,
            0,
            [0, 0, 0],
            0,
            # float3 C;
            [0, 0, 0],
        ]
        t = t_spline[i]
        diracs = drgb_spline_diracs[i]
        for j in range(idx.shape[1]):
            # print(state)
            last_ctrl_pt = [t[j-1], diracs[j-1]]
            if j >= 1 and (t[j] > t_min and t[j - 1] < t_min):
                # insert 0 point
                state = fn.update(state, [t_min, [0, 0, 0, 0]], t_min, t_max)
                # state = fn.update(state, [t_min, [0, 0, 0, 0]], t_min, t_max)
                last_ctrl_pt = [t_min, [0, 0, 0, 0]]
            if j >= 1 and (t[j - 1] < t_max and t[j] > t_max):
                # insert 0 point
                state = fn.update(state, [t_max, [0, 0, 0, 0]], t_min, t_max)
                # state = fn.update(state, [t_max, [0, 0, 0, 0]], t_min, t_max)
                last_ctrl_pt = [t_max, [0, 0, 0, 0]]

            new_state = fn.update(state, [t[j], diracs[j]], t_min, t_max)
            if j > 0:
                new_state_dual = fn.to_dual(new_state, [t[j], diracs[j]])
                state_pred_dual = fn.inverse_update_dual(new_state_dual, last_ctrl_pt, t_min, t_max)
                state_pred = fn.from_dual(state_pred_dual, last_ctrl_pt)
                diff = [np.abs(np.array(v1) - np.array(v2)).mean() > 0
                        for k, (v1, v2) in enumerate(zip(state_pred, state))
                        if k not in [4,5,6]]
                if True in diff:
                    print('pred', state_pred)
                    print('    ', state)
                    # print('inv diff', diff)
                    print(t[j-1], t[j])
            state = new_state
            weights[i, j] = state[-2]
            d_spline[i, j] = state[-4]
            avg_colors[i, j] = np.array(state[-3])
            # print(f"weight: {state[-2]}, col: {state[3]}, acol: {state[-1]}")
        spline_color[i] = state[-1]
        total_density[i] = state[3]
        # print(state[-1], spline_color)
    return spline_color, {
        "avg_colors": avg_colors,
        "t_spline": t_spline,
        "d_spline": d_spline,
        "total_density": total_density,
        # "rgb_spline": rgb_spline,
        "weights": weights,
    }


# t_range = (-10, 20)
no_overlap = True

t_range = (4, 12)
# t_range = (4, 18)

rng = random.PRNGKey(3)
num_pixels = 1
num_gaussians = 16

if no_overlap:
    means = jnp.linspace(-16, 16, num_gaussians + 2)[None, 1:-1].repeat(num_pixels, 0)

    key, rng = random.split(rng)
    max_dt = 8 / ((num_gaussians + 2) * np.sqrt(6))
    stds = (
        random.uniform(key, [num_pixels, num_gaussians], minval=0.2, maxval=1.0)
        * max_dt
    )
    assert np.all(
        (means[..., :-1] + np.sqrt(6) * max_dt) < (means[..., 1:] - np.sqrt(6) * max_dt)
    )

else:
    key, rng = random.split(rng)
    means = random.uniform(key, [num_pixels, num_gaussians]) * 16

    key, rng = random.split(rng)
    stds = jnp.exp(random.normal(key, [num_pixels, num_gaussians]) - 1)
    # stds = jnp.exp(random.normal(key, [num_pixels, num_gaussians])+1)


key, rng = random.split(rng)
heights = jnp.exp(random.normal(key, [num_pixels, num_gaussians]) - 1)

# means = jnp.array([[5,6,  9]])
# stds = jnp.array([[2, 0.3, 1]])
# heights = jnp.array([[.3, 0.8, 1]])

# means = jnp.array([[7]])
# stds = jnp.array([[2]])
# heights = jnp.array([[.7]])


key, rng = random.split(rng)
colors = random.uniform(key, [num_pixels, num_gaussians, 3])
colors = jnp.ones_like(random.uniform(key, [num_pixels, num_gaussians, 3]))

num_quad = 2**14
tdist = jnp.linspace(*t_range, num_quad + 1)

# dtype = jnp.float64
# means = dtype(means)
# stds = dtype(stds)
# heights = dtype(heights)
# colors = dtype(colors)
# tdist = dtype(tdist)

quad_color, quad_extras = render_quadrature(
    tdist,
    lambda t: query_gaussians(
        t, means, stds, heights, colors, kernel_fn=triangular_kernel_fn
    ),
    return_extras=True,
)

spline_color1, spline_extras = render_spline(
    means, stds, heights, colors, t_range=t_range, return_extras=True
)
# print(1, spline_extras["weights"][0])
# print(1, spline_extras["d_spline"][0])
# print(12, spline_extras["avg_colors"][0])
spline_color2, spline_extras = slang_render_spline(
    means, stds, heights, colors, t_range=t_range, return_extras=True
)
print(quad_extras['total_density'], spline_extras['total_density'])
# print(2, spline_extras["weights"][0])
# print(2, spline_extras["d_spline"][0])
# print(22, spline_extras["avg_colors"][0])

# print(quad_color)
# print(spline_color)
print(quad_color - spline_color2)
print(quad_color, spline_color1, spline_color2)
print(f"RMSE = {jnp.sqrt(jnp.mean((quad_color - spline_color2).flatten()**2)):0.3e}")

# key, rng = random.split(rng)
v = np.random.rand(1)

