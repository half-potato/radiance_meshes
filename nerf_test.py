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
from torch import nn
import matplotlib.pyplot as plt
import mediapy
from icecream import ic
from data import loader
import random
import time
import tinycudann as tcnn
from tqdm import tqdm
import numpy as np
# from dtet import DelaunayTriangulation
# from dtet.build.dtet import DelaunayTriangulation
from gDel3D.build.gdel3d import Del
from utils import cam_util
from utils.train_util import *
from models.vertex_color import Model, TetOptimizer
from utils import viz_util
from plyfile import PlyData, PlyElement
from utils.graphics_utils import l2_normalize_th
from utils import topo_utils
import imageio
from fused_ssim import fused_ssim
from utils.loss_utils import ssim

train_cameras, test_cameras, scene_info = loader.load_dataset(
    "/optane/nerf_datasets/360/bicycle", "images_4", data_device="cuda", eval=True)


sh_deg = 3
# dim = 8
# vertex_rgbs_param[:, 4:] = 0

# path = "/home/dronelab/gaussian-splatting-merge/eval/bicycle/point_cloud/iteration_7000/point_cloud.ply"
# plydata = PlyData.read(path)

# xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
#                 np.asarray(plydata.elements[0]["y"]),
#                 np.asarray(plydata.elements[0]["z"])),  axis=1)
# opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

# features_dc = np.zeros((xyz.shape[0], 3, 1))
# features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
# features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
# features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

# # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
# # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
# # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
# # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
# # for idx, attr_name in enumerate(extra_f_names):
# #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
# # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
# # features_extra = features_extra.reshape((features_extra.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3))


# vertices = torch.as_tensor(xyz)
# N = 10000
# vertices = torch.cat([
#   vertices.reshape(-1, 3),
#   (torch.rand((N, 3)) * (maxv - minv) + minv) * 2
# ], dim=0)

# vertices = nn.Parameter(vertices.cuda())
# offset = torch.zeros((1, dim), device=device)
# offset[0, 3] = math.log(0.1)
# vertex_rgbs_param = (2*torch.rand((vertices.shape[0], dim), device=device)-1) + offset
# # vertex_rgbs_param[:, 1:4] = inverse_sigmoid(torch.as_tensor(SH2RGB(features_dc)).cuda()).reshape(-1, 3)
# vertex_rgbs_param = nn.Parameter(vertex_rgbs_param)

camera = train_cameras[0]

tile_size = 4

device = torch.device('cuda')
model = Model.init_from_pcd(scene_info.point_cloud, 2, device)
tet_optim = TetOptimizer(model)

images = []
psnrs = [[]]
inds = []

args = lambda x: x
args.start_tracking = 000
args.cloning_interval = 500
args.budget = 1_000_000
args.num_densification_samples = len(train_cameras)
args.densify_start = 1500
args.num_densify_iter = 16000 - args.densify_start
args.num_iter = 30000
args.sh_degree_interval = 2000

# args = lambda x: x
# args.start_tracking = 000
# args.cloning_interval = 500
# args.budget = 500_000
# args.num_densification_samples = 50
# args.num_densify_iter = 2500
# args.densify_start = 1000
# args.num_iter = args.densify_start + args.num_densify_iter + 1500
# args.sh_degree_interval = 500

def target_num(x):
    S = model.vertices.shape[0]
    N = args.num_densify_iter // args.cloning_interval
    k = (args.budget - S) // N
    return (args.budget - S - k * N) // N**2 * x**2 + k * x + S

print([target_num(i+1) for i in range(args.num_densify_iter // args.cloning_interval)])

progress_bar = tqdm(range(args.num_iter))
for train_ind in progress_bar:
    delaunay_interval = 10#1 if train_ind < args.densify_start else 10
    do_delaunay = train_ind % delaunay_interval == 0 and train_ind > 0
    do_cloning = max(train_ind - args.densify_start, 0) % args.cloning_interval == 0 and (args.num_densify_iter + args.densify_start) > train_ind > args.densify_start
    do_tracking = False
    do_sh = train_ind % args.sh_degree_interval == 0 and train_ind > 0
    do_sh_step = train_ind % 16 == 0


    # ind = random.randint(0, len(train_cameras)-1)
    if len(inds) == 0:
        inds = list(range(len(train_cameras)))
        random.shuffle(inds)
        psnrs.append([])
    ind = inds.pop()
    # ind = 1
    camera = train_cameras[ind]
    target = camera.original_image.cuda()

    st = time.time()
    render_pkg = render(camera, model)
    # torch.cuda.synchronize()
    # print(f'render: {(time.time()-st)}')
    image = render_pkg['render']
    l2_loss = ((target - image)**2).mean()
    # ssim_loss = 1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))
    # lambda_ssim = 0.2
    # loss = (1-lambda_ssim)*l2_loss + lambda_ssim*ssim_loss
    loss = l2_loss
 
    st = time.time()
    loss.backward()
    tet_optim.optim.step()
    tet_optim.optim.zero_grad()

    if do_tracking:
        tet_optim.track_gradients()

    if do_sh_step:
        tet_optim.sh_optim.step()
        tet_optim.sh_optim.zero_grad()

    if do_delaunay:
        tet_optim.vertex_optim.step()
        tet_optim.vertex_optim.zero_grad()
    # print(f'bw: {(time.time()-st)}')

    if do_sh:
        model.sh_up()

    if do_cloning:
        # collect data
        tet_optim.optim.zero_grad()
        tet_optim.vertex_optim.zero_grad()
        tet_optim.sh_optim.zero_grad()

        full_inds = list(range(len(train_cameras)))
        random.shuffle(full_inds)

        tet_optim.reset_tracker()
        sampled_cameras = [train_cameras[i] for i in full_inds[:args.num_densification_samples]]
        tet_rgbs_grad = None
        for camera in sampled_cameras:
            render_pkg = render(camera, model, register_tet_hook=True)
            # torch.cuda.synchronize()
            # print(f'render: {(time.time()-st)}')
            image = render_pkg['render']
            l2_loss = ((target - image)**2).mean()
            # ssim_loss = 1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))
            # lambda_ssim = 0.2
            # loss = (1-lambda_ssim)*l2_loss + lambda_ssim*ssim_loss
            loss = l2_loss
        
            loss.backward()
            tet_grad = render_pkg['tet_grad'][0]
            scores = tet_grad.abs().sum(dim=-1)
            if tet_rgbs_grad is None:
                tet_rgbs_grad = scores
            else:
                tet_rgbs_grad = torch.maximum(scores, tet_rgbs_grad)

        # tet_optim.track_gradients()

        tet_optim.optim.step()
        tet_optim.vertex_optim.step()
        tet_optim.sh_optim.step()
        tet_optim.sh_optim.zero_grad()
        tet_optim.optim.zero_grad()
        tet_optim.vertex_optim.zero_grad()

        target_addition = target_num((train_ind - args.densify_start) // args.cloning_interval + 1) - model.vertices.shape[0]
        ic(target_addition, (train_ind - args.densify_start) // args.cloning_interval + 1)
        rgbs_threshold = torch.sort(tet_rgbs_grad).values[-min(int(target_addition), tet_rgbs_grad.shape[0])]

        clone_mask = tet_rgbs_grad > rgbs_threshold

        # rgbs_grad, vertex_grad = tet_optim.get_tracker_predicates() 
        # reduce_type = "sum"
        # # tet_std = torch.std(model.vertex_rgbs_param[model.indices], dim=1).max(dim=1).values
        # tet_rgbs_grad = rgbs_grad[model.indices].sum(dim=1)
        # tet_vertex_grad = vertex_grad[model.indices].sum(dim=1)
        # clone_mask = (tet_rgbs_grad > rgbs_threshold) | (tet_vertex_grad > vertex_threshold)
        clone_indices = model.indices[clone_mask]

        barycentric = torch.rand((clone_indices.shape[0], clone_indices.shape[1], 1), device=device)
        # barycentric = torch.ones((clone_indices.shape[0], clone_indices.shape[1], 1), device=device)
        barycentric_weights = barycentric / (1e-3+barycentric.sum(dim=1, keepdim=True))
        tet_optim.split(clone_indices, barycentric_weights)

        # new_vertex_location, radius = topo_utils.calculate_circumcenters_torch(model.vertices[clone_indices])
        # new_feat = model.vertex_rgbs_param[clone_indices].mean(dim=1)
        # perturb = sample_uniform_in_sphere(new_vertex_location.shape[0], 3).to(new_vertex_location.device)
        # tet_optim.add_points(new_vertex_location + radius.reshape(-1, 1) * perturb, new_feat)

        out = f"#RGBS Clone: {(tet_rgbs_grad > rgbs_threshold).sum()} "
        # out += f"#Vertex Clone: {(tet_vertex_grad > vertex_threshold).sum()} "
        out += f"∇RGBS: {tet_rgbs_grad.mean()} "
        out += f"target_addition: {target_addition} "
        # out += f"∇Vertex: {tet_vertex_grad.mean()} "
        # out += f"σ: {tet_std.mean()}"
        print(out)
        tet_optim.reset_tracker()

    psnr = 20 * math.log10(1.0 / math.sqrt(l2_loss.detach().cpu().item()))
    psnrs[-1].append(psnr)

    disp_ind = max(len(psnrs)-2, 0)
    avg_psnr = sum(psnrs[disp_ind]) / max(len(psnrs[disp_ind]), 1)
    # progress_bar.set_postfix({"PSNR": f"{psnr:>8.2f} Mean: {avg_psnr:>8.2f} #V: {len(model)} #T: {model.indices.shape[0]}"})
    progress_bar.set_postfix({
        "PSNR": repr(f"{psnr:>5.2f}"),
        "Mean": repr(f"{avg_psnr:>5.2f}"),
        "#V": len(model),
        "#T": model.indices.shape[0]
    })

    st = time.time()
    with torch.no_grad():
        if train_ind % 1 == 0:
            train_ind = 1
            camera = train_cameras[train_ind]
            render_pkg = render(camera, model)
            image = render_pkg['render']
            image = image.permute(1, 2, 0)
            image = image.detach().cpu().numpy()
            images.append(image)
    # print(f'second: {(time.time()-st)}')

    if do_delaunay:
        st = time.time()
        model.update_triangulation()
        # print(f'update: {(time.time()-st)}')

avged_psnrs = [sum(v)/len(v) for v in psnrs if len(v) == len(train_cameras)]
plt.plot(range(len(avged_psnrs)), avged_psnrs)
plt.show()
mediapy.write_video("training.mp4", images)

torch.cuda.synchronize()
torch.cuda.empty_cache()

epath = cam_util.generate_cam_path(train_cameras, 400)
eimages = []
for camera in tqdm(epath):
    render_pkg = render(camera, model)
    image = render_pkg['render']
    image = image.permute(1, 2, 0)
    image = image.detach().cpu().numpy()
    eimages.append(image)

mediapy.write_video("rotating.mp4", eimages)


