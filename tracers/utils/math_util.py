import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
from plyfile import PlyData, PlyElement


def l2_normalize(x, eps=np.finfo(np.float32).eps):
    """Normalize x to unit length along last axis."""
    return x / np.sqrt(np.clip(np.sum(x**2, axis=-1, keepdims=True), eps, None))


def l2_normalize_th(x, eps=torch.finfo(torch.float32).eps):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(
        torch.clip(torch.sum(x**2, dim=-1, keepdim=True), eps, None)
    )


def jl2_normalize(x, eps=jnp.finfo(jnp.float32).eps):
    """Normalize x to unit length along last axis."""
    # return x / jnp.sqrt(jnp.clip(np.sum(x**2, axis=-1, keepdims=True), eps, None))
    return x / jnp.sqrt(np.sum(x**2, axis=-1, keepdims=True))


def empty_y(R):
    return np.array(
        [
            [R[0, 0], 0.0, R[0, 1]],
            [0.0, 0.0, 0.0],
            [R[1, 0], 0.0, R[1, 1]],
        ]
    )


def rotm(ang):
    return np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])


def quatToMat3(q):
    r = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return (
        np.array(
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - r * z),
                2.0 * (x * z + r * y),
                2.0 * (x * y + r * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - r * x),
                2.0 * (x * z - r * y),
                2.0 * (y * z + r * x),
                1.0 - 2.0 * (x * x + y * y),
            ]
        )
        .reshape(3, 3)
        .T
    )


def jquatToMat3(q):
    r = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return (
        jnp.array(
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - r * z),
                2.0 * (x * z + r * y),
                2.0 * (x * y + r * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - r * x),
                2.0 * (x * z - r * y),
                2.0 * (y * z + r * x),
                1.0 - 2.0 * (x * x + y * y),
            ]
        )
        .reshape(3, 3)
        .T
    )


def jrotm(ang):
    return jnp.array(
        [[jnp.cos(ang), -jnp.sin(ang)], [jnp.sin(ang), jnp.cos(ang)]]
    ).reshape(2, 2)


def jget_rot(ang_or_quat):
    if ang_or_quat.size == 1:
        rot = jrotm(ang_or_quat)
    else:
        rot = jquatToMat3(ang_or_quat)
    return rot


def get_rays(directions, c2w):
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    rays_o = np.tile(c2w[:3, 3], (rays_d.shape[0], 1))  # (H, W, 3)
    return rays_o, rays_d




def build_rotation(r):
    # from 3DGS
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def save_ply(means, scales, quats, densities, features, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    N = means.shape[0]
    xyz = means.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    # features = features.reshape(-1, 1, 3)
    f_dc = (
        features[:, :1]
        .detach()
        .transpose(1, 2)
        .reshape(N, -1)
        .contiguous()
        .cpu()
        .numpy()
    )
    f_rest = (
        features[:, 1:]
        .detach()
        .transpose(1, 2)
        .reshape(N, -1)
        .contiguous()
        .cpu()
        .numpy()
    )

    densities = densities.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    quats = quats.detach().cpu().numpy()

    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(features.shape[2]):
        l.append("f_dc_{}".format(i))
    for i in range((features.shape[1] - 1) * features.shape[2]):
        l.append("f_rest_{}".format(i))
    l.append("density")
    for i in range(scales.shape[1]):
        l.append("scale_{}".format(i))
    for i in range(quats.shape[1]):
        l.append("rot_{}".format(i))
    dtype_full = [(attribute, "f4") for attribute in l]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    print(f_dc.shape, f_rest.shape)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, densities.reshape(-1, 1), scales, quats), axis=1
    )
    print(attributes.shape)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)
