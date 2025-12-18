# This file is full of tools to help debug the gradient in Optix/Slang
import math
from pathlib import Path

import build.splinetracer.extension.splinetracer_cpp_extension as sp
import jax
import jax.numpy as jnp
import numpy as np
import slangpy
import torch

import sh_util
from jax_render import *
from safe_math import safe_div
from splinetracer import trace_rays
from util import *


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


sh_deg = 1
MAX_ITERS = 1000
EPS = 1e-20

np.random.seed(0)
# ============================
# Camera SETUP
# ============================
T = np.eye(4)
# Init rays
res = 800
# fov = 10 * math.pi / 180
fov = 0.5 * math.pi / 180
fx = res / 2 / math.tan(fov / 2)
radius = 2
dx, dy = np.meshgrid(
    np.linspace(-res / 2, res / 2, res) / fx,
    np.linspace(-res / 2, res / 2, res) / fx,
)
crayd = l2_normalize(np.stack([dx, dy, np.ones_like(dx)], axis=-1)).reshape(-1, 3)
rayo, rayd = get_rays(crayd, T)
rayo = rayo.reshape(res, res, 3).reshape(-1, 3)
rayd = rayd.reshape(res, res, 3).reshape(-1, 3)

# ============================
# SCENE SETUP
# ============================
quat = l2_normalize(np.random.rand(4))
rot = quatToMat3(quat)
print(f"Quat: {quat.tolist()}")
mean = np.array([0, 2e-3, 1])
s = np.log(1e-1)
print(s)
scale = np.exp(s) * np.array([1.0, 2.0, 3.0])
# density = -13
n = 2
area = (scale[0] ** n + scale[1] ** n + scale[2] ** n) ** (1 / n)
density = np.exp(np.array([2])) / area
color = np.random.rand(1, (sh_deg + 1) ** 2, 3)

rayo = rayo[0:1]
rayd = rayd[0:1]

# T = np.eye(4)
# # Init rays
# res = 3
# fov = 0.5 * math.pi / 180
# fx = res /2 /math.tan(fov/2)
# radius = 2
# dx, dy = np.meshgrid(
#     np.linspace(-res/2, res/2, res) / fx,
#     np.linspace(-res/2, res/2, res) / fx,
# )
# crayd = l2_normalize(np.stack([ dx, dy, np.ones_like(dx)], axis=-1)).reshape(-1, 3)
# rayo, rayd = get_rays(crayd, T)
# rayo = rayo.reshape(res, res, 3)
# rayd = rayd.reshape(res, res, 3)
#
# # TODO REMOVE
# rayo = rayo[1:2, 1:2].reshape(-1, 3)
# rayd = rayd[1:2, 1:2].reshape(-1, 3)

device = torch.device(0)
rayo_th = torch.as_tensor(rayo.reshape(-1, 3)).float().to(device)
rayd_th = torch.as_tensor(rayd.reshape(-1, 3)).float().to(device)

target_out = np.array([1, 0, 0])
target_out_th = torch.as_tensor(target_out).to(device)


def l2_loss2(color):
    return ((color - target_out_th) ** 2).mean()


def l2loss_n_extract(state):
    return ((state["C"] - target_out) ** 2).mean()


params = dict(scale=scale, mean=mean, quat=quat, feature=color, density=density)

device = torch.device(0)
ctx = sp.OptixContext(device)
prims = sp.Primitives(device)
# prims = sp.Primitives("/usr/local/google/home/alexmai/optix-examples/models/lego.ply", device)
print("Scales", scale)
means_th = torch.as_tensor(mean).reshape(1, 3).float().to(device)
scale_th = torch.as_tensor(scale).reshape(1, 3).float().to(device)
quat_th = torch.as_tensor(quat).reshape(1, 4).float().to(device)
density_th = torch.tensor([density]).reshape(1).float().to(device)
color_th = torch.as_tensor(color).reshape(1, (sh_deg + 1) ** 2, 3).float().to(device)
prims.add_primitives(means_th, scale_th, quat_th, density_th, color_th)
gas = sp.GAS(ctx, device, prims, True, True, True)

forward = sp.Forward(ctx, device, prims, True)
out = forward.trace_rays(gas, rayo_th, rayd_th, MAX_ITERS)
# collect_ids = sp.CollectIds(ctx, device, prims)
# tri_collection = collect_ids.trace_rays(gas, rayo_th, rayd_th, out["saved"])
tri_collection = out["tri_collection"]
iters = out["saved"].iters.reshape(-1).item()
print(out["saved"].iters, out["color"])
tri_ids = tri_collection.cpu().numpy()[:iters]
face_ids = [i % 20 for i in tri_ids]
print(tri_ids)

# Forward
states = [
    dict(
        t=0.0,
        drgb=jnp.array([0, 0, 0, 0]),
        d_drgb=jnp.array([0, 0, 0, 0]),
        logT=0.0,
        C=jnp.array([0, 0, 0]),
    )
]
ctrl_pts = []
for face_id in face_ids:
    ctrl_pts.append(intersect(rayo, rayd, params, face_id))
    states.append(update(states[-1], ctrl_pts[-1], 0, 20))
print("state check", states[-1]["C"])

backward = sp.Backward(ctx, device, prims)

loss_grad = torch.tensor(
    np.array(jax.grad(l2loss_n_extract)(states[-1])["C"]), device=device
)
loss_grad = torch.ones((1, 3), device=device)
print(loss_grad)
# print("Backward")
# bwout = backward.trace_rays(gas, rayo_th, rayd_th, out['saved'], loss_grad)
jloss_grad = np.array(loss_grad.cpu().reshape(-1))

print(f"\n{bcolors.HEADER}END RANDOM DEBUG{bcolors.ENDC}\nGT vs Pred")
print(params, rayo, rayd, tri_ids)
_, bwd_render = jax.vjp(render, params, rayo, rayd, tri_ids)
jax_grad, drayo, drayd, _ = bwd_render(jloss_grad)
jax_grad["rayo"] = drayo
jax_grad["rayd"] = drayd
print(jax_grad, jloss_grad)

print(f"\n{bcolors.HEADER}Derivative Kernel{bcolors.ENDC}")
kernels = slangpy.loadModule(
    str(Path(__file__).parent / "tri_splinetracer/slang/backwards_kernel.slang")
)

start_ids = torch.cumsum(out["saved"].iters, dim=0).int()

num_prims = means_th.shape[0]
dL_dCs = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
dL_dmeans = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
dL_dscales = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)
dL_dquats = torch.zeros((num_prims, 4), dtype=torch.float32, device=device)
dL_ddensities = torch.zeros((num_prims), dtype=torch.float32, device=device)
dL_dfeatures = torch.zeros(
    (num_prims, (sh_deg + 1) ** 2, 3), dtype=torch.float32, device=device
)
num_rays = rayo_th.shape[0]
dL_drayo = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)
dL_drayd = torch.zeros((num_rays, 3), dtype=torch.float32, device=device)
dL_dmeans2D = torch.zeros((num_prims, 3), dtype=torch.float32, device=device)

kernels.backwards_kernel(
    last_state=out["saved"].states,
    last_dirac=out["saved"].diracs,
    start_ids=start_ids,
    iters=out["saved"].iters,
    tri_collection=tri_collection,
    ray_origins=rayo_th,
    ray_directions=rayd_th,
    model=(
        means_th,
        scale_th,
        quat_th,
        density_th,
        color_th,
        dL_dmeans,
        dL_dscales,
        dL_dquats,
        dL_ddensities,
        dL_dfeatures,
        dL_drayo,
        dL_drayd,
        dL_dmeans2D,
    ),
    dL_dCs=loss_grad.reshape(-1, 3),
    tmin=0,
    tmax=20,
    max_iters=MAX_ITERS,
).launchRaw(blockSize=(32, 1, 1), gridSize=(64, 1, 1))

# color, vjp = torch.autograd.functional.vjp(
#     trace_rays,
#     (
#         means_th,
#         scale_th,
#         quat_th,
#         density_th,
#         color_th,
#         rayo_th,
#         rayd_th,
#     ),
#     torch.ones((1, 3), device=device),
# )
bwout = dict(
    mean=dL_dmeans,
    scale=dL_dscales,
    quat=dL_dquats,
    density=dL_ddensities,
    feature=dL_dfeatures,
    rayo=dL_drayo,
    rayd=dL_drayd,
)
print(dL_dmeans)

print(f"\n{bcolors.HEADER}Full Grad Errs:{bcolors.ENDC}")
for key in jax_grad.keys():
    err = ((bwout[key].cpu().numpy() - jax_grad[key]) ** 2).mean()
    print(f"{key}: {err}")

print(f"{bcolors.HEADER}Parameters{bcolors.ENDC}")
print(params)
color1 = out["color"][:, :3].cpu().numpy()
color2 = render(params, rayo, rayd, tri_ids)
print(f"{bcolors.HEADER}compare{bcolors.ENDC}", color1, color2)
# assert np.allclose(color1, color2), f"{color1} vs {color2}"


print(f"\n{bcolors.HEADER}Testing Intersect Derivative{bcolors.ENDC}")
fn = slangpy.loadModule(
    str(Path(__file__).parent / "tri_splinetracer/slang/tri-intersect.slang")
)

for face_id in face_ids:
    print(f"{bcolors.BOLD}Face id: {face_id}{bcolors.ENDC}")
    derivative = dict(t=0.0, dirac=np.array([0.0, 1.0, 1.0, 1.0]))
    props = [
        "rayo",
        "rayd",
        "scale",
        "mean",
        "feature",
        "quat",
        "density",
    ]
    feature = params["feature"]
    deg = int(math.sqrt(feature.shape[1]) - 1)
    color = sh_util.eval_sh(deg, feature.T, rayd)
    deriv_out = fn.intersect_derivative(
        rayo.reshape(-1).tolist(),
        rayd.reshape(-1).tolist(),
        params["scale"].reshape(-1).tolist(),
        params["mean"].reshape(-1).tolist(),
        params["quat"].tolist(),
        color.tolist(),
        params["density"].tolist()[0],
        face_id,
        derivative["t"],
        derivative["dirac"].tolist(),
    )
    deriv_out = {k: np.array(v) for k, v in zip(props, deriv_out)}
    # print(deriv_out)

    jprimals, bwd_diff_intersect = jax.vjp(intersect, rayo, rayd, params, face_id)
    print(f"Dirac multi: {jprimals['dirac'][0]}")
    """
    primals = fn.intersect(
        rayo.reshape(-1).tolist(), rayd.reshape(-1).tolist(),
        params['scale'].reshape(-1).tolist(), params['mean'].reshape(-1).tolist(), params['quat'].tolist(),
        params['feature'].tolist(), params['density'].tolist()[0], face_id)
    print('jax', jprimals)
    print('slang', primals)
    """
    drayo, drayd, jderiv_out, _ = bwd_diff_intersect(derivative)
    for k in jderiv_out.keys():
        err = ((jderiv_out[k] - deriv_out[k]) ** 2).mean()
        print(f"{k} err: {err}")
        # print(f"{k}: {jderiv_out[k]}")

print(f"\n{bcolors.HEADER}Testing Update Derivative{bcolors.ENDC}")
smachine = slangpy.loadModule(
    str(Path(__file__).parent / "tri_splinetracer/slang/spline-machine.slang")
)

spline_state_keys = ["t", "drgb", "d_drgb", "logT", "C"]


def pack_dict_to_state(d):
    return [(0.0, 0.0, 0.0)] + [d[k] for k in spline_state_keys]


def pack_state_to_dict(s):
    state = {k: np.array(v) for k, v in zip(["padding"] + spline_state_keys, s)}
    del state["padding"]
    return state


spline_ctrlpt_keys = ["t", "dirac"]


def pack_dict_to_ctrlpt(d):
    return [d[k] for k in spline_ctrlpt_keys]


def pack_ctrlpt_to_dict(s):
    return {k: np.array(v) for k, v in zip(spline_ctrlpt_keys, s)}


def l2loss_n_extract(state):
    return ((state["C"] - target_out) ** 2).mean()


dLdStates = [jax.grad(l2loss_n_extract)(states[-1])]
jdLdStates = [jax.grad(l2loss_n_extract)(states[-1])]
jdParam = None
dParam = None
for i in range(len(face_ids) - 1, -1, -1):
    jinput_deriv = jdLdStates[-1]
    input_deriv = dLdStates[-1]
    new_state, bwd_diff_update = jax.vjp(update, states[i], ctrl_pts[i], 0, 20)
    jdLdState, jdLdCtrl, _, _ = bwd_diff_update(jinput_deriv)
    jdLdStates.append(jdLdState)

    jprimals, bwd_diff_intersect = jax.vjp(intersect, rayo, rayd, params, face_ids[i])
    drayo, drayd, jderiv_out, _ = bwd_diff_intersect(jdLdCtrl)
    if jdParam is None:
        jdParam = jderiv_out
    else:
        for k in jdParam.keys():
            jdParam[k] += jderiv_out[k]

    print(f"{bcolors.BOLD}Face id[{i}]: {face_ids[i]}{bcolors.ENDC}")

    dLdState, dLdCtrl = smachine.update_derivative(
        pack_dict_to_state(states[i]),
        pack_dict_to_ctrlpt(ctrl_pts[i]),
        0,
        20,
        pack_dict_to_state(jinput_deriv),
    )
    dLdState = pack_state_to_dict(dLdState)
    dLdCtrl = pack_ctrlpt_to_dict(dLdCtrl)
    dLdStates.append(dLdState)

    feature = params["feature"]
    deg = int(math.sqrt(feature.shape[1]) - 1)
    color = sh_util.eval_sh(deg, feature.T, rayd)
    deriv_out = fn.intersect_derivative(
        rayo.reshape(-1).tolist(),
        rayd.reshape(-1).tolist(),
        params["scale"].reshape(-1).tolist(),
        params["mean"].reshape(-1).tolist(),
        params["quat"].tolist(),
        color.tolist(),
        params["density"].tolist()[0],
        face_ids[i],
        jdLdCtrl["t"],
        jdLdCtrl["dirac"].tolist(),
    )
    deriv_out = {k: np.array(v) for k, v in zip(props, deriv_out)}
    if dParam is None:
        dParam = deriv_out
    else:
        for k in dParam.keys():
            dParam[k] += deriv_out[k]

    # print(f'{bcolors.BOLD}inputs{bcolors.ENDC}')
    # print('input_state:', states[i])
    # print('input_ctrl_pt:', ctrl_pts[i])
    # print('input_deriv:', input_deriv)
    # print('jinput_deriv:', jinput_deriv)
    # print(f'{bcolors.BOLD}gt update derivative{bcolors.ENDC}', jdLdState, jdLdCtrl)
    # print(f'{bcolors.BOLD}pred update derivative{bcolors.ENDC}', dLdState, dLdCtrl)
    # print(f'{bcolors.BOLD}intersection derivative{bcolors.ENDC}', jderiv_out)
    for k in jderiv_out.keys():
        err = ((jderiv_out[k] - deriv_out[k]) ** 2).mean()
        print(f"{k} err: {err}")

    state = new_state
#
# for dp in dParams[1:]:
#     for k in dParam.keys():
#         dParam[k] += dp[k]
print(f"Manual unroll: {dParam}")
print(f"Jax grad vs {jax_grad}")

print("mean_grad:", bwout["mean"])
print("mean_grad_gt:", jax_grad["mean"])
print("quat_grad:", bwout["quat"])
print("quat_grad_gt:", jax_grad["quat"])
print("quat_grad_unroll:", dParam["quat"])
print("scale_grad:", bwout["scale"])
print("scale_grad_gt:", jax_grad["scale"])
print("color_grad_optix:", bwout["feature"])
print("color_grad_unroll, slang:", dParam["feature"])
print("color_grad_unroll, jax:", jdParam["feature"])
print("color_grad_gt:", jax_grad["feature"])

print("Collected triangles: ", tri_ids)
