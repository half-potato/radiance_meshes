import math

import jax
import jax.numpy as jnp
import numpy as np
import torch

import sh_util
from safe_math import safe_div
from util import *

EPS = 1e-20
BASE_VERTICES = jnp.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
    ]
)

INDICES = jnp.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [5, 1, 2],
        [5, 2, 3],
        [5, 3, 4],
        [5, 4, 1],
        [0, 1, 6],
        [0, 2, 6],
        [0, 3, 6],
        [0, 4, 6],
        [5, 1, 6],
        [5, 2, 6],
        [5, 3, 6],
        [5, 4, 6],
        [6, 1, 2],
        [6, 2, 3],
        [6, 3, 4],
        [6, 4, 1],
    ]
)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def ray_intersect_plane(ray_origin, ray_vector, normal, point):
    denom = (normal * ray_vector).sum()
    numer = (normal * ray_origin).sum() - (normal * point).sum()
    dist = jnp.abs(safe_div(numer, denom))
    return dist
    backup = jnp.linalg.norm(point - ray_origin, ord=2)
    return jnp.where((jnp.abs(denom) > 1e-20), dist, backup)


def face2signs(face_id):
    q4 = face_id % 4
    s = jnp.array(
        [
            jnp.where(q4 < 2, 1, -1),
            jnp.where((q4 == 0) | (q4 == 3), 1, -1),
            jnp.where(face_id < 4, 1, -1),
        ]
    )
    return s


def delta_indicator_fn(face_id):
    q4 = face_id % 4
    s = jnp.array(
        [
            jnp.where((face_id < 8) & (q4 % 2 == 0), 1, 0),
            jnp.where((face_id < 8) & (q4 % 2 == 1), 1, 0),
            jnp.where(face_id >= 8, 1, 0),
        ]
    )
    return s


def calc_dirac(Trayd, face_id):
    delta_indicator = delta_indicator_fn(jnp.maximum(face_id - 8, 0))
    inner = -2 * delta_indicator @ jnp.abs(Trayd)

    outer = jnp.abs(face2signs(face_id) @ Trayd)
    dirac_multi = jnp.where(face_id < 8, outer, inner)
    return dirac_multi


def intersect(rayo, rayd, params, face_id):
    scales, mean, quat, feature, density = (
        params["scale"],
        params["mean"],
        params["quat"],
        params["feature"],
        params["density"],
    )
    feature = feature.reshape(-1, 3)
    deg = int(math.sqrt(feature.shape[0]) - 1)
    color = sh_util.eval_sh(deg, feature.T, rayd).reshape(3)
    inds = INDICES[face_id]
    R = jquatToMat3(jl2_normalize(quat))
    S = jnp.diag(scales)
    invS = jnp.diag(safe_div(1, jnp.maximum(scales, 1e-8)))
    T = invS @ R
    Tinv = R.T @ S
    # T = R @ invS
    # Tinv = S @ R.T

    #    """
    tri_a = BASE_VERTICES[inds[0]]
    tri_b = BASE_VERTICES[inds[1]]
    tri_c = BASE_VERTICES[inds[2]]

    # edge1 = Tinv @ (tri_b - tri_a)
    # edge2 = Tinv @ (tri_c - tri_a)
    # cross = jnp.cross(edge1, edge2)
    edge1 = tri_b - tri_a
    edge2 = tri_c - tri_a
    # cross = jnp.linalg.det(Tinv) * T.T @ jnp.cross(edge1, edge2)
    # cross = jnp.linalg.det(Tinv) * (R.T @ (jnp.cross(edge1, edge2) / scales))
    cross = R.T @ (jnp.cross(edge1, edge2) / scales)
    normal = jl2_normalize(cross)

    # t = ray_intersect_plane(rayo, rayd, jnp.array([1.0, 0.0, 0.0]), tri_a)
    t = ray_intersect_plane(rayo, rayd, normal, Tinv @ tri_a + mean)
    """
    tri_a = BASE_VERTICES[inds[0]]
    tri_b = BASE_VERTICES[inds[1]]
    tri_c = BASE_VERTICES[inds[2]]

    edge1 = tri_b - tri_a
    edge2 = tri_c - tri_a
    cross = jnp.cross(edge1, edge2)
    normal = jl2_normalize(Tinv @ cross)
    # normal = jax.lax.stop_gradient(normal)

    point_on_tri = Tinv @ BASE_VERTICES[inds[0]] + mean;
    t = ray_intersect_plane(rayo, rayd, normal, point_on_tri)
    """
    # t = jax.lax.stop_gradient(t)

    Trayd = T @ rayd.reshape(3)
    Trayd = (R @ rayd.reshape(3)) / scales
    # dirac_multi = density * calc_dirac(Trayd, face_id)
    Trayo = (R @ (rayo.reshape(3) - mean)) / scales
    # Trayd = jax.lax.stop_gradient(Trayd)
    # jax.debug.print("norm: {}, trayd: {}", normal, Trayd)
    dirac_multi = jnp.where(
        jnp.linalg.norm(Trayo) <= 1, 0.0, density * calc_dirac(Trayd, face_id)
    )
    # dirac_multi = jnp.where(jnp.linalg.norm(Trayo) <= 1, 0.0, density * calc_dirac(Trayd, face_id))
    out = dict(
        t=t,
        dirac=jnp.array(
            [
                dirac_multi,
                dirac_multi * color[0],
                dirac_multi * color[1],
                dirac_multi * color[2],
            ]
        ).reshape(-1),
    )
    return out


def update(state, ctrl_pt, t_min, t_max):
    t = ctrl_pt["t"]
    dt = t - state["t"]

    new_state = dict()
    new_state["drgb"] = jnp.maximum(
        state["drgb"].reshape(-1) + state["d_drgb"].reshape(-1) * dt, 0.0
    )
    new_state["d_drgb"] = state["d_drgb"].reshape(-1) + ctrl_pt["dirac"].reshape(-1)

    new_state["t"] = t

    # Clamping
    mask = ~((t > t_max) | (state["t"] < t_min))
    drgb = jnp.where(
        mask.reshape(1, 1), state["drgb"].reshape(1, 4), jnp.zeros((1, 4))
    ).reshape(-1)
    new_drgb = jnp.where(
        mask.reshape(1, 1), new_state["drgb"].reshape(1, 4), jnp.zeros((1, 4))
    ).reshape(-1)

    # Integrate information
    avg = jnp.maximum(new_drgb, 0.0) / 2 + jnp.maximum(drgb, 0.0) / 2
    area = avg[0] * dt

    rgb_avg = avg[1:]
    # rgb_norm = safe_div(rgb_avg, jnp.maximum(avg[0], 0))
    rgb_norm = jnp.minimum(safe_div(rgb_avg, jnp.maximum(avg[0], 0)), 1 + 1e-7)
    # rgb_norm = jnp.clip(safe_div(rgb_avg, jnp.maximum(avg[0], 0)), 0, 1)
    # jax.debug.print("area: {}, dt: {}", area, dt)

    new_state["logT"] = area + state["logT"]
    # weight = jnp.exp(log1mexp(area) - state['logT'])
    weight = jnp.minimum((1 - jnp.exp(-area)) * jnp.exp(-state["logT"]), 1)
    new_state["C"] = state["C"] + weight.reshape() * rgb_norm.reshape(3)

    return new_state


def render(params, rayo, rayd, face_ids):
    state = dict(
        t=0.0,
        drgb=jnp.array([0, 0, 0, 0]),
        d_drgb=jnp.array([0, 0, 0, 0]),
        logT=0.0,
        C=jnp.array([0, 0, 0]),
    )
    for face_id in face_ids:
        ctrl_pt = intersect(rayo, rayd, params, face_id)
        # jax.debug.print("face: {}: t: {} dirac: {}, state {}", face_id, ctrl_pt['t'], ctrl_pt['dirac'], state)
        # jax.debug.print("{} t: {}", face_id, ctrl_pt['t'])
        state = update(state, ctrl_pt, 0, 20)
    return state["C"]
