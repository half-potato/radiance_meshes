# %%

import os
VERSION = 9
if VERSION is not None:
    os.environ["CC"] = f"/usr/bin/gcc-{VERSION}"
    os.environ["CXX"] = f"/usr/bin/g++-{VERSION}"
from pathlib import Path
import sys
sys.path.append(str(Path(os.path.abspath('')).parent))
print(str(Path(os.path.abspath('')).parent))
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
from tqdm import tqdm
import numpy as np
# from dtet import DelaunayTriangulation
# from dtet.build.dtet import DelaunayTriangulation
from gDel3D.build.gdel3d import Del
from scipy.spatial import KDTree
from utils import cam_util
from utils import train_util
from utils import viz_util
from plyfile import PlyData, PlyElement
from utils.graphics_utils import l2_normalize_th

K = 20

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
train_cameras, test_cameras, scene_info = loader.load_dataset(
    "/optane/nerf_datasets/360/bicycle", "images_8", data_device="cuda", eval=True)

# %%

device = torch.device('cuda')
torch.manual_seed(2)
N = scene_info.point_cloud.points.shape[0]
# N = 1000
vertices = torch.as_tensor(scene_info.point_cloud.points)[:N]
minv = vertices.min(dim=0, keepdim=True).values
maxv = vertices.max(dim=0, keepdim=True).values
repeats = 10
vertices = vertices.reshape(-1, 1, 3).expand(-1, repeats, 3)
vertices = vertices + torch.randn(*vertices.shape) * 1e-1
vertices = vertices.reshape(-1, 3)
# N = 10000
# vertices = torch.cat([
#   vertices.reshape(-1, 3),
#   (torch.rand((N, 3)) * (maxv - minv) + minv) * 2
# ], dim=0)
vertices = nn.Parameter(vertices.cuda())
scaling = torch.tensor([[1, 1, 1, 1]], device=device)
offset = torch.tensor([[0, 0, 0, math.log(0.01)]], device=device)
vertex_rgbs_param = nn.Parameter(scaling * (2*torch.rand((vertices.shape[0], 4), device=device)-1) + offset)
dim = 8
offset = torch.zeros((1, dim), device=device)
offset[0, 0] = math.log(0.01)
vertex_rgbs_param = nn.Parameter((2*torch.rand((vertices.shape[0], dim), device=device)-1) + offset)

encoding = tcnn.Encoding(3, dict(
    otype="HashGrid",
    n_levels=16,
    n_features_per_level=2,
    log2_hashmap_size=14,
    base_resolution=16,
    per_level_scale=2
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

def rgbs_activation(rgbs_raw):
    rgbs = torch.cat([torch.sigmoid(rgbs_raw[:, :3]), safe_exp(rgbs_raw[:, 3:])], dim=1)
    return rgbs

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
    cxyz, _ = contract_mean_std(xyz, torch.ones_like(xyz[..., 0]))
    rgbs_raw = net((cxyz/2 + 1)/2).float()
    return rgbs_activation(rgbs_raw)

encoding = tcnn.Encoding(3+dim-4, dict(
    otype = "Composite",
    nested = [
        {
            "n_dims_to_encode": 3,
            "otype": "SphericalHarmonics",
            "degree": 4
        },
        {
            "otype": "Identity"
        }
    ]
))
network = tcnn.Network(encoding.n_output_dims, 3, dict(
    # otype="CutlassMLP",
    otype="FullyFusedMLP",
    activation="ReLU",
    output_activation="None",
    n_neurons=64,
    n_hidden_layers=2,
))
view_dep_net = torch.nn.Sequential(
    encoding, network
).to(device)

# %%
path = "/home/dronelab/gaussian-splatting-merge/eval/bicycle/point_cloud/iteration_7000/point_cloud.ply"
plydata = PlyData.read(path)

xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),  axis=1)
opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

features_dc = np.zeros((xyz.shape[0], 3, 1))
features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

# extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
# extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
# assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
# features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
# for idx, attr_name in enumerate(extra_f_names):
#     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
# # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
# features_extra = features_extra.reshape((features_extra.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3))

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def inverse_sigmoid(y):
    return torch.log(y / (1 - y))

vertices = nn.Parameter(torch.as_tensor(xyz).cuda())
dim = 4
offset = torch.zeros((1, dim), device=device)
offset[0, 0] = math.log(0.01)
vertex_rgbs_param = (2*torch.rand((vertices.shape[0], dim), device=device)-1) + offset
vertex_rgbs_param[:, 1:4] = inverse_sigmoid(torch.as_tensor(SH2RGB(features_dc)).cuda()).reshape(-1, 3)
vertex_rgbs_param = nn.Parameter(vertex_rgbs_param)

# %%
camera = train_cameras[0]
# def render(camera, indices, vertices):
#     fy = fov2focal(camera.fovy, camera.image_height)
#     fx = fov2focal(camera.fovx, camera.image_width)
#     K = torch.tensor([
#     [fx, 0, camera.image_width/2],
#     [0, fy, camera.image_height/2],
#     [0, 0, 1],
#     ]).to(camera.world_view_transform.device)

#     # world_view_transform, projection_matrix, cam_pos, fovy, fovx = common_camera_properties_from_gsplat(
#     #     camera.world_view_transform.T, K, camera.image_height, camera.image_width)
#     cam_pos = camera.world_view_transform.T.inverse()[:, 3]
#     cell_values = rgbs_activation(vertex_rgbs_param[indices.reshape(-1)].reshape(-1, 4, 4).sum(dim=1) / 4)
#     # ic(vertex_rgbs_param[indices.reshape(-1)], vertex_rgbs_param)
#     # cell_values = torch.gather(vertex_rgbs_param, 0, indices.long()).sum(dim=1) / 4
#     # ic(torch.gather(vertex_rgbs_param, 0, indices.long()).shape)
#     # cell_values = rgbs_activation(cell_values)


#     render_pkg = render_alpha_blend_tiles_slang_raw(indices, vertices, rgbs_fn,
#                                                     camera.world_view_transform.T, K, cam_pos,
#                                                     camera.fovy, camera.fovx, camera.image_height,
#                                                     camera.image_width, cell_values=cell_values,
#                                                     tile_size=tile_size)
#     return render_pkg

# optim = torch.optim.Adam([
#     # {"params": net.parameters(), "lr": 1e-3},
#     {"params": [vertices], "lr": 1e-4},
#     {"params": [vertex_rgbs_param], "lr": 1e-1},
# ])
def render(camera, indices, vertices):
    fy = fov2focal(camera.fovy, camera.image_height)
    fx = fov2focal(camera.fovx, camera.image_width)
    K = torch.tensor([
    [fx, 0, camera.image_width/2],
    [0, fy, camera.image_height/2],
    [0, 0, 1],
    ]).to(camera.world_view_transform.device)

    cam_pos = camera.world_view_transform.T.inverse()[:, 3]
    features = vertex_rgbs_param[indices.reshape(-1)].reshape(-1, 4, 4).sum(dim=1) / 4
    # directions = l2_normalize_th(vertices - camera.camera_center.reshape(1, 3))
    # feline_features = torch.cat([directions, vertex_rgbs_param[..., 4:]], dim=1)
    # rgb_features = view_dep_net(feline_features)
    # # reattach the density
    # features = torch.cat([vertex_rgbs_param[..., 1:4] + 1e-2*rgb_features, vertex_rgbs_param[..., :1]], dim=1)
    # features = features[indices.reshape(-1)].reshape(-1, 4, 4).sum(dim=1) / 4
    cell_values = rgbs_activation(features)

    render_pkg = render_alpha_blend_tiles_slang_raw(indices, vertices, rgbs_fn,
                                                    camera.world_view_transform.T, K, cam_pos,
                                                    camera.fovy, camera.fovx, camera.image_height,
                                                    camera.image_width, cell_values=cell_values,
                                                    tile_size=tile_size)
    return render_pkg

optim = torch.optim.Adam([
    # {"params": net.parameters(), "lr": 1e-3},
    {"params": [vertices], "lr": 1e-4},
    {"params": view_dep_net.parameters(), "lr": 1e-3},
    {"params": [vertex_rgbs_param], "lr": 5e-2},
])
tile_size = 8
images = []

# v = DelaunayTriangulation()
# v.init_from_points(vertices.detach().cpu().numpy())
# indices_np = v.get_cells()
v = Del(vertices.shape[0])
indices_np, prev = v.compute(vertices.detach().cpu())
indices_np = indices_np.numpy()

# v = Delaunay(vertices.detach().cpu().numpy())
# indices_np = v.simplices

# indices_np = indices_np[np.lexsort(indices_np.T)].astype(np.int32)
indices_np = indices_np[(indices_np < vertices.shape[0]).all(axis=1)]
indices = torch.as_tensor(indices_np).cuda()

print(indices.shape)

old_vertices = vertices.detach().cpu().numpy()
render_pkg = render(camera, indices, vertices)
last_circumcenters = render_pkg['circumcenters']
num_violations = []
num_tets = []
psnrs = [[]]
inds = []
training_images = []
do_delaunay = False


# progress_bar = tqdm(range(4*len(train_cameras)+2))
progress_bar = tqdm(range(5000))
for i in progress_bar:
    do_delaunay = i % 10 == 0
    optim.zero_grad()

    # ind = random.randint(0, len(train_cameras)-1)
    if len(inds) == 0:
        inds = list(range(len(train_cameras)))
        random.shuffle(inds)
        psnrs.append([])
    ind = inds.pop()
    # ind = 1
    camera = train_cameras[ind]
    target = camera.original_image.cuda()

    # if do_delaunay:
    #     st = time.time()
    #     with torch.no_grad():
    #         sensitivity = topo_utils.compute_vertex_sensitivity(indices, vertices)
    #         scaling = 1/(sensitivity.reshape(-1, 1)+1e-5)
    # else:
    #     scaling = torch.tensor(0.0, device=device)
    # scale_vertices = train_util.ScaleGradients.apply(vertices, scaling)
    # print(f'sensitivity: {1/(time.time()-st)}')

    st = time.time()
    render_pkg = render(camera, indices, vertices)
    # torch.cuda.synchronize()
    # print(f'render: {1/(time.time()-st)}')
    image = render_pkg['render']
    loss = ((target - image)**2).mean()
 
    st = time.time()
    loss.backward()
    optim.step()
    # torch.cuda.synchronize()
    # print(f'bw: {1/(time.time()-st)}')

    psnr = 20 * math.log10(1.0 / math.sqrt(loss.detach().cpu().item()))
    psnrs[-1].append(psnr)

    disp_im = torch.cat([target, image], dim=2)
    disp_im = disp_im.permute(1, 2, 0)
    disp_im = disp_im.detach().cpu().numpy()
    training_images.append(disp_im)
    
    disp_ind = max(len(psnrs)-2, 0)
    avg_psnr = sum(psnrs[disp_ind]) / max(len(psnrs[disp_ind]), 1)
    progress_bar.set_postfix({"PSNR": f"{psnr:.02f} Mean: {avg_psnr}"})

    with torch.no_grad():
        if i % 1 == 0:
            ind = 1
            camera = train_cameras[ind]
            render_pkg = render(camera, indices, vertices)
            image = render_pkg['render']
            image = image.permute(1, 2, 0)
            image = image.detach().cpu().numpy()
            images.append(image)

    # st = time.time()
    if do_delaunay:
        st = time.time()
        v = Del(vertices.shape[0])
        indices_np, prev = v.compute(vertices.detach().cpu())
        indices_np = indices_np.numpy()
        # indices_np = v.compute_from_prev(vertices.detach().cpu(), indices.cpu()).numpy()
        indices_np = indices_np[(indices_np < vertices.shape[0]).all(axis=1)]
        # print(f'del: {1/(time.time()-st)}')
        indices = torch.as_tensor(indices_np.astype(np.int32)).cuda()
        old_vertices = vertices.detach().cpu().numpy()

    if i % 500 == 0:
        plt.imshow(image)
        plt.show()
        avged_psnrs = [sum(v)/len(v) for v in psnrs if len(v) == len(train_cameras)]
        plt.plot(range(len(avged_psnrs)), avged_psnrs)
        plt.show()
        mediapy.show_video(images)
avged_psnrs = [sum(v)/len(v) for v in psnrs if len(v) == len(train_cameras)]
plt.plot(range(len(avged_psnrs)), avged_psnrs)
plt.show()
mediapy.show_video(images)
plt.plot(range(len(num_violations)), num_violations)
plt.plot(range(len(num_tets)), num_tets)
plt.show()

# %%
viz_util.create_image_viewer(training_images, skip_value=20)

# %%
epath = cam_util.generate_cam_path(train_cameras, 400)
eimages = []
for camera in tqdm(epath):
    render_pkg = render(camera, indices, vertices)
    image = render_pkg['render']
    image = image.permute(1, 2, 0)
    image = image.detach().cpu().numpy()
    eimages.append(image)

mediapy.show_video(eimages)


