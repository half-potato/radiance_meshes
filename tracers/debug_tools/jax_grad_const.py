# This file just checks the gradient of the splinetracer in Jax vs Optix/Slang
import math
from pathlib import Path

# import build.splinetracer.extension.const_splinetracer_cpp_extension as sp
import build.splinetracer.extension.tetra_splinetracer_cpp_extension as sp
import jax
import jax.numpy as jnp
import numpy as np
import slangpy
import torch
from icecream import ic

from jax_render_const import *
from safe_math import safe_div
from util import *

np.random.seed(0)

MAX_ITERS = 100
sh_deg = 0
quat = l2_normalize(
    np.array(
        [
            0.9811850346541486,
            0.13331543321474051,
            0.07252929475418773,
            0.1193416291155902,
        ]
    )
)
quat = l2_normalize(
    np.array(
        [
            0.7161766493797235,
            0.654162074589675,
            0.10972513770463387,
            0.21707920491715532,
        ]
    )
)
# quat = l2_normalize(np.array([1, 0.0, 0, 0]))
quat = l2_normalize(np.random.rand(4))
quat = np.array(
    [0.7002152964925366, 0.2520461093253735, 0.6677710020092772, 0.015911826021448486]
)

# tri_ids = [5, 14, 18]
# tri_ids = [5, 17, 1]
print(f"Quat: {quat.tolist()}")
mean = np.array([1e-1, 1e-1, 1])
s = 1e-0
scale = s * np.random.rand(3) + 1e-3
# mean test case:
scale = np.array([0.00113759, 0.00115165, 0.00035421])
scale = np.array([0.00792179, 0.00743873, 0.00103447])
scale = 0.1 * np.array([1, 1, 1])

density = np.array([10.0])
color = np.random.rand((sh_deg + 1) ** 2, 3)

# ============================
# SCENE SETUP
# ============================
# quat = l2_normalize(np.random.rand(4))
print(f"Quat: {quat.tolist()}")
mean = np.array([-1e-1, -1e-1, 1.2])
s = 8e-1
scale = s * np.array([1, 2, 3])
density = np.array([100.0])

params = dict(scale=scale, mean=mean, quat=quat, feature=color, density=density)

T = np.eye(4)
# Init rays
res = 1
fov = 0.5 * math.pi / 180
fx = res / 2 / math.tan(fov / 2)
radius = 2
dx, dy = np.meshgrid(
    np.linspace(-res / 2, res / 2, res) / fx,
    np.linspace(-res / 2, res / 2, res) / fx,
)
crayd = l2_normalize(np.stack([dx, dy, np.ones_like(dx)], axis=-1)).reshape(-1, 3)
rayo, rayd = get_rays(crayd, T)
rayo = rayo.reshape(res, res, 3)
rayd = rayd.reshape(res, res, 3)

# TODO REMOVE
# rayo = rayo[1:2, 1:2].reshape(-1, 3)
# rayd = rayd[1:2, 1:2].reshape(-1, 3)

rayo = rayo.reshape(-1, 3)
rayd = rayd.reshape(-1, 3)

device = torch.device(0)
rayo_th = torch.as_tensor(rayo.reshape(-1, 3)).float().to(device)
rayd_th = torch.as_tensor(rayd.reshape(-1, 3)).float().to(device)

target_out = np.array([1, 0, 0])
target_out_th = torch.as_tensor(target_out).to(device)


def l2_loss(params, rayo, rayd, tri_ids):
    color = render(params, rayo, rayd, tri_ids)
    return ((color - target_out) ** 2).mean()


def l2_loss2(color):
    return ((color - target_out_th) ** 2).mean()


def l2loss_n_extract(state):
    return ((state - target_out) ** 2).mean()


device = torch.device(0)
ctx = sp.OptixContext(device)
prims = sp.Primitives(device)
# prims = sp.Primitives("/usr/local/google/home/alexmai/optix-examples/models/lego.ply", device)
print("Scales", scale)
means_th = torch.as_tensor(mean).reshape(1, 3).float().to(device)
scale_th = torch.as_tensor(scale).reshape(1, 3).float().to(device)
quat_th = torch.as_tensor(quat).reshape(1, 4).float().to(device)
density_th = torch.tensor([density]).reshape(1).float().to(device)
color_th = torch.as_tensor(color).reshape(1, -1, 3).float().to(device)
ic(color_th)
prims.add_primitives(means_th, scale_th, quat_th, density_th, color_th)
gas = sp.GAS(ctx, device, prims, True, True, True)

forward = sp.Forward(ctx, device, prims, True)
out = forward.trace_rays(gas, rayo_th, rayd_th, 0, 1000, MAX_ITERS, 100)
# collect_ids = sp.CollectIds(ctx, device, prims)
# tri_collection = collect_ids.trace_rays(gas, rayo_th, rayd_th, out["saved"])
tri_collection = out["tri_collection"]
iters = out["saved"].iters.reshape(-1).item()
tri_ids = tri_collection.cpu().numpy()[:iters]

print(tri_ids)
loss_grad = torch.tensor(
    np.array(jax.grad(l2loss_n_extract)(out["color"][:, :3].cpu().numpy())),
    device=device,
)

print(loss_grad)
# print("Backward")
# bwout = backward.trace_rays(gas, rayo_th, rayd_th, out['saved'], loss_grad)

print(f"\n{bcolors.HEADER}END RANDOM DEBUG{bcolors.ENDC}\nGT vs Pred")

jax_grad = jax.grad(l2_loss)(params, rayo, rayd, tri_ids)

print(f"\n{bcolors.HEADER}Derivative Kernel{bcolors.ENDC}")
kernels = slangpy.loadModule(
    str(Path(__file__).parent / "tetra_splinetracer/slang/backwards_kernel.slang"), verbose=True
)
print(str(Path(__file__).parent / "tetra_splinetracer/slang/backwards_kernel.slang"))

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
touch_count = torch.zeros((num_prims), dtype=torch.int32, device=device)

print("BW")
kernels.backwards_kernel(
    last_state=out["saved"].states,
    last_dirac=out["saved"].diracs,
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
    touch_count=touch_count,
    # means=means_th,
    # scales=scale_th,
    # quats=quat_th,
    # densities=density_th,
    # features=color_th,
    #
    # dL_dCs=loss_grad.reshape(-1, 3),
    # dL_dmeans=dL_dmeans,
    # dL_dscales=dL_dscales,
    # dL_dquats=dL_dquats,
    # dL_ddensities=dL_ddensities,
    # dL_dfeatures=dL_dfeatures,
    # dL_drayo=dL_drayo,
    # dL_drayd=dL_drayd,
    dL_dCs=loss_grad.reshape(-1, 3),
    wcts=torch.ones((1, 4, 4), device=device, dtype=torch.float32),
    tmin=0,
    tmax=20,
    max_iters=MAX_ITERS,
    max_prim_size=100,
).launchRaw(blockSize=(32, 1, 1), gridSize=(64, 1, 1))
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

# print(f"{bcolors.HEADER}Parameters{bcolors.ENDC}")
# print(params)
color1 = out["color"][:, :3].cpu().numpy()
color2 = render(params, rayo, rayd, tri_ids)
print(tri_ids)
print(f"compare slang: {color1}, jax: {color2}")
# assert np.allclose(color1, color2), f"{color1} vs {color2}"

