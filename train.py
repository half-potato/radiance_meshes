# %%
import os
from pathlib import Path
import sys
# sys.path.append(str(Path(os.path.abspath('')).parent))
# print(str(Path(os.path.abspath('')).parent))
import math
import torch
from torch import Tensor
from delaunay_rasterization.internal.alphablend_tiled_slang import render_alpha_blend_tiles_slang_raw
from scipy.spatial import Voronoi, Delaunay
from torch import nn
import matplotlib.pyplot as plt
import mediapy
from icecream import ic
from data import loader
import random
import time
import tinycudann as tcnn
from utils.contraction import contract_mean_std
from utils import topo_utils
from gDel3D.build.gdel3d import Del
from tqdm import tqdm

tile_size = 8
K = 1

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def get_slang_projection_matrix(znear, zfar, fy, fx, height, width, device):
    tanHalfFovX = width/(2*fx)
    tanHalfFovY = height/(2*fy)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    z_sign = 1.0

    P = torch.tensor([
       [2.0 * znear / (right - left),     0.0,                          (right + left) / (right - left), 0.0 ],
       [0.0,                              2.0 * znear / (top - bottom), (top + bottom) / (top - bottom), 0.0 ],
       [0.0,                              0.0,                          z_sign * zfar / (zfar - znear),  -(zfar * znear) / (zfar - znear) ],
       [0.0,                              0.0,                          z_sign,                          0.0 ]
    ], device=device)

    return P

def common_camera_properties_from_gsplat(viewmats, Ks, height, width):
  """ Fetches all the Camera properties from the inria defined object"""
  zfar = 100.0
  znear = 0.01
  
  world_view_transform = viewmats
  fx = Ks[0,0]
  fy = Ks[1,1]
  projection_matrix = get_slang_projection_matrix(znear, zfar, fy, fx, height, width, Ks.device)
  fovx = focal2fov(fx, width)
  fovy = focal2fov(fy, height)

  cam_pos = viewmats.inverse()[:, 3]

  return world_view_transform, projection_matrix, cam_pos, fovy, fovx




# %%
train_cameras, test_cameras, scene_info = loader.load_dataset("/optane/nerf_datasets/360/bicycle", "images_8", data_device="cuda", eval=True)

# %%

torch.manual_seed(2)
N = scene_info.point_cloud.points.shape[0]
# vertices = (torch.rand((N, 3)) * 2 - 1)*10
vertices = torch.as_tensor(scene_info.point_cloud.points)[:N]
minv = vertices.min(dim=0, keepdim=True).values
maxv = vertices.max(dim=0, keepdim=True).values
N = 700000
repeats = 10
vertices = vertices.reshape(-1, 1, 3).expand(-1, repeats, 3)
vertices = vertices + torch.randn(*vertices.shape) * 1e-1
N = 50000
vertices = torch.cat([
  vertices.reshape(-1, 3),
  torch.rand((N, 3)) * (maxv - minv) + minv
], dim=0)
v = Delaunay(vertices.numpy())

indices = torch.as_tensor(v.simplices).cuda()
vertices = nn.Parameter(vertices.cuda())
print(vertices.shape)

# num_freq = 5
# net = nn.Sequential(
#   nn.Linear(3*num_freq, 256),
#   nn.ReLU(inplace=True),
#   nn.Linear(256, 256),
#   nn.ReLU(inplace=True),
#   nn.Linear(256, 256),
#   nn.ReLU(inplace=True),
#   nn.Linear(256, 4),
# ).cuda()
device = torch.device('cuda')
encoding = tcnn.Encoding(3, dict(
    otype="HashGrid",
    n_levels=16,
    n_features_per_level=2,
    log2_hashmap_size=14,
    base_resolution=1,
    per_level_scale=1.5
))
network = tcnn.Network(encoding.n_output_dims, 4, dict(
    # otype="CutlassMLP",
    otype="FullyFusedMLP",
    activation="ReLU",
    output_activation="None",
    n_neurons=64,
    n_hidden_layers=2,
))
net = torch.nn.Sequential(
    encoding, network
).to(device)


# freq = 2**torch.arange(num_freq).cuda()

def safe_exp(x):
  return x.clip(max=5).exp()

def safe_trig_helper(x, fn, t=100 * torch.pi):
  """Helper function used by safe_cos/safe_sin: mods x before sin()/cos()."""
  return fn(torch.nan_to_num(torch.where(torch.abs(x) < t, x, x % t)))


def safe_cos(x):
  """jnp.cos() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, torch.cos)


def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, torch.sin)

def rgbs_fn(xyz):
  # exyz = xyz.reshape(-1, 3, 1) * freq.reshape(1, -1).expand(1, 3, -1)
  # exyz = contraction(exyz)
  # exy = safe_cos(exyz.reshape(-1, 3 * num_freq))
  # rgbs_raw = net(exy)
  # if xyz.abs().max() > 1e2:
  #   print(f'real big: {(xyz.abs() > 1e2).sum()}')
  xyz = torch.where(xyz < -1e2, -xyz, xyz)
  cxyz, _ = contract_mean_std(xyz, torch.ones_like(xyz[..., 0]))
  rgbs_raw = net((cxyz/2 + 1)/2).float()
  rgbs = torch.cat([torch.sigmoid(rgbs_raw[:, :3]), safe_exp(rgbs_raw[:, 3:]-3)], dim=1)
  return rgbs


# %%
camera = train_cameras[0]
print(camera.projection_matrix)

# %%
def render(camera, indices, vertices):
    # camera.projection_matrix = get_slang_projection_matrix(
    #     0.01, 100, fov2focal(camera.fovx, camera.image_width), fov2focal(camera.fovx, camera.image_width),
    #     camera.image_height, camera.image_width, "cuda")

    fy = fov2focal(camera.fovy, camera.image_height)
    fx = fov2focal(camera.fovx, camera.image_width)
    K = torch.tensor([
    [fx, 0, camera.image_width/2],
    [0, fy, camera.image_height/2],
    [0, 0, 1],
    ]).cuda()

    world_view_transform, projection_matrix, cam_pos, fovy, fovx = common_camera_properties_from_gsplat(
        camera.world_view_transform.T, K, camera.image_height, camera.image_width)

    render_pkg = render_alpha_blend_tiles_slang_raw(indices, vertices, rgbs_fn,
                                                    camera.world_view_transform.T, K, cam_pos,
                                                    camera.fovy, camera.fovx, camera.image_height, camera.image_width, tile_size=tile_size)
    return render_pkg

optim = torch.optim.Adam([
    {"params": net.parameters(), "lr": 5e-3},
    {"params": [vertices], "lr": 1e-4},
])
images = []
# v = Delaunay(vertices.detach().cpu().numpy())
# indices_np = v.simplices
v = Del(vertices.shape[0])
indices_np = v.compute(vertices.detach().cpu()).numpy()
indices = torch.as_tensor(indices_np).cuda()
old_vertices = vertices.detach().cpu().numpy()
progress_bar = tqdm(range(2001))
for i in progress_bar:
    optim.zero_grad()


    ind = random.randint(0, len(train_cameras)-1)
    camera = train_cameras[ind]
    target = camera.original_image.cuda()

    render_pkg = render(camera, indices, vertices)
    # print(1/(time.time()-st))
    image = render_pkg['render']
    loss = ((target - image)**2).mean()

    ind = 0
    camera = train_cameras[ind]
    render_pkg = render(camera, indices, vertices)
    image = render_pkg['render']
    image = image.permute(1, 2, 0)
    image = image.detach().cpu().numpy()
    images.append(image)
    loss.backward()
    optim.step()

    psnr = 20 * math.log10(1.0 / math.sqrt(loss.detach().cpu().item()))
    progress_bar.set_postfix({"PSNR": f"{psnr:.02f}"})

    st = time.time()
    if i % K == 0:
        v = Del(vertices.shape[0])
        indices_np = v.compute(vertices.detach().cpu()).numpy()
        # v = Delaunay(vertices.detach().cpu().numpy())
        # indices_np = v.simplices
        indices = torch.as_tensor(indices_np).cuda()
        old_vertices = vertices.detach().cpu().numpy()

    # new_vertices = vertices.detach().cpu().numpy()
    # indices_np, _ = topo_utils.update_tetrahedralization(old_vertices, new_vertices, indices_np)
    # indices = torch.as_tensor(indices_np).cuda()
    # print((time.time()-st))

    # if i % 50 == 0:
    #     plt.imshow(image)
    #     plt.show()
    #     mediapy.show_video(images)
# mediapy.show_video(images)
mediapy.write_video(f"training_{K}.mp4", images)

# %%
# torch.save(dict(
#     net=net.state_dict(),
#     vertices = vertices
# ), "edge_cases/1.pth")

# %%



