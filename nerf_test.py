import os
# VERSION = 9
# if VERSION is not None:
#     os.environ["CC"] = f"/usr/bin/gcc-{VERSION}"
#     os.environ["CXX"] = f"/usr/bin/g++-{VERSION}"
from pathlib import Path
import sys
sys.path.append(str(Path(os.path.abspath('')).parent))
print(str(Path(os.path.abspath('')).parent))
import math
import torch
import matplotlib.pyplot as plt
import mediapy
from icecream import ic
from data import loader
import random
import time
from tqdm import tqdm
import numpy as np
from utils import cam_util
from utils.train_util import *
# from models.vertex_color import Model, TetOptimizer
from models.ingp_color import Model, TetOptimizer
from fused_ssim import fused_ssim
from pathlib import Path, PosixPath
from utils.args import Args
import pickle
import json
from utils import safe_math
from delaunay_rasterization.internal.render_err import render_err
import imageio
from torch.profiler import profile, ProfilerActivity, record_function
from utils import test_util
import cv2
from utils.graphics_utils import tetra_volume


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PosixPath):
            return str(o)
        return super().default(o)

class SimpleSampler:
    def __init__(self, total_num_samples, batch_size):
        self.total_num_samples = total_num_samples
        self.batch_size = batch_size
        self.curr = total_num_samples
        self.ids = None

    def nextids(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        self.curr += batch_size
        if self.curr + batch_size > self.total_num_samples:
            # self.ids = torch.LongTensor(np.random.permutation(self.total_num_samples))
            self.ids = torch.randperm(self.total_num_samples, dtype=torch.long, device=device)
            self.curr = 0
        ids = self.ids[self.curr : self.curr + batch_size]
        return ids

args = Args()
args.tile_size = 16
args.sh_deg = 3
args.output_path = Path("output")
args.densify_interval = 500
args.budget = 1_000_000
args.num_samples = 200
args.densify_start = 2000
args.densify_end = 5000
args.freeze_start = 100000
args.iterations = 7000
args.sh_interval = 500
args.image_folder = "images_4"
args.eval = True
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/test/")
args.s_param_lr = 0.025
args.rgb_param_lr = 0.025
args.sh_param_lr = 0.0025
args.delaunay_start = 0

args.log2_hashmap_size = 22
args.per_level_scale = 2
args.L = 10
args.density_offset = -1
args.light_offset = -3

args.hidden_dim = 64
args.scale_multi = 1.0

args.vertices_lr = 1e-4
args.lr_delay = 50
args.final_vertices_lr = 1e-6
args.vertices_lr_max_steps = args.iterations

args.vertices_lr_delay_multi = 1e-8
# args.network_lr = 0.00125
# args.final_network_lr = 0.000125
args.encoding_lr = 0.00125
args.final_encoding_lr = 0.000125
# args.lambda_dist = 1e-2

# args.vertices_lr_delay_mult = 0.01
# args.network_lr = 1e-3
# args.final_network_lr = 1e-5
args.network_lr = 0.00125
args.final_network_lr = 0.000125
# args.encoding_lr = 1e-2
# args.final_encoding_lr = 1e-3

args.lambda_ssim = 0.1
args.clone_lambda_ssim = 0.1

args.clip_multi = 1e-4
args.weight_decay = 0.01
args.render_size_min = 0
args.vertices_beta = [0.9, 0.99]
args.net_weight_decay = 0.0
args.contract_vertices = False
args.vertices_warmup = 0
args.hashmap_dim = 4
args.min_clone_size = 1e-3

# args.lambda_dist = 1e-3
# args.ladder_p = -0.5
# args.pre_multi = 4000.0
args.lambda_dist = 1e-2
args.ladder_p = -0.25
args.pre_multi = 10000

args.split_std = 0.1
args.split_mode = "barycentric"
args.clone_schedule = "linear"

args = Args.from_namespace(args.get_parser().parse_args())

args.output_path.mkdir(exist_ok=True, parents=True)

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cuda", eval=args.eval)

args.num_samples = min(len(train_cameras), args.num_samples)

camera = train_cameras[0]
with open('camera.pkl', 'wb') as f:
    pickle.dump(camera, f)

device = torch.device('cuda')
model = Model.init_from_pcd(scene_info.point_cloud, train_cameras, device, **args.as_dict())
tet_optim = TetOptimizer(model, **args.as_dict())
sample_camera = test_cameras[4]

images = []
psnrs = [[]]
inds = []
# def target_num(x):

num_densify_iter = args.densify_end - args.densify_start
N = num_densify_iter // args.densify_interval + 1

def target_num(x):
    if args.clone_schedule == "linear":
        S = model.vertices.shape[0]
        k = (args.budget - S) // N
        return k * x + S
    elif args.clone_schedule == "quadratic":
        S = model.vertices.shape[0]
        k = 2 * (args.budget - S) // N
        a = (args.budget - S - k * N) // N**2
        return a * x**2 + k * x + S
    else:
        raise Exception(f"Clone Schedule: {args.clone_schedule} is not supported")

print([target_num(i+1) for i in range(N)])

# epath = cam_util.generate_cam_path(train_cameras, 400)
# sample_camera = epath[175]
# sample_camera = train_cameras[60]

densification_sampler = SimpleSampler(len(train_cameras), args.num_samples)

video_writer = cv2.VideoWriter(str(args.output_path / "training.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30,
                               (sample_camera.image_width, sample_camera.image_height))

progress_bar = tqdm(range(args.iterations))
torch.cuda.empty_cache()
last_densify = 1000
for iteration in progress_bar:
    delaunay_interval = 1 if iteration < args.delaunay_start else 10
    do_delaunay = iteration % delaunay_interval == 0 and iteration > args.vertices_warmup
    do_cloning = max(iteration - args.densify_start, 0) % args.densify_interval == 0 and args.densify_end > iteration >= args.densify_start
    do_tracking = False
    do_sh = iteration % args.sh_interval == 0 and iteration > 0
    do_sh_step = iteration % 16 == 0
    do_vertices_warmup = iteration < args.vertices_warmup# or iteration > args.freeze_start

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
    L = 2000
    x = np.clip(iteration-last_densify, 0, L)
    bg_scale = np.clip(np.exp(x-L) - np.exp(-L), 0, 1)
    # bg_scale = min(max((iteration-1000)/2000, 0), 1)
    bg = bg_scale if iteration < args.densify_end + L//2 else 0
    # bg = 0
    render_pkg = render(camera, model, bg=bg, min_t=model.scene_scaling * 0.3, **args.as_dict())
    # render_pkg = render(camera, model, min_t=model.scene_scaling * 0.1, **args.as_dict())
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #          profile_memory=True, with_stack=True) as prof:
    #     with record_function("model_inference"):
    #         render_pkg = render(camera, model, min_t=model.scene_scaling * 0.1, **args.as_dict())

    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_memory_usage", row_limit=10))
    # prof.export_chrome_trace("trace.json")

    # torch.cuda.synchronize()
    # print(f'render: {(time.time()-st)}')
    image = render_pkg['render'].clip(min=0, max=1)
    l2_loss = ((target - image)**2).mean()
    reg = tet_optim.regularizer()
    ssim_loss = 1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))
    loss = (1-args.lambda_ssim)*l2_loss + args.lambda_ssim*ssim_loss + reg + args.lambda_dist * render_pkg['distortion_loss']

    st = time.time()
    loss.backward()

    tet_optim.main_step()
    tet_optim.main_zero_grad()

    if do_sh_step:
        tet_optim.sh_optim.step()
        tet_optim.sh_optim.zero_grad()

    if do_vertices_warmup:
        tet_optim.vertex_optim.zero_grad()

    if do_delaunay:
        tet_optim.vertex_optim.step()
        tet_optim.vertex_optim.zero_grad()

    tet_optim.update_learning_rate(iteration)

    if do_sh:
        model.sh_up()

    # st = time.time()
    with torch.no_grad():
        if iteration % 1 == 0:
            render_pkg = render(sample_camera, model, tile_size=args.tile_size, bg=bg_scale)
            sample_image = render_pkg['render']
            sample_image = sample_image.permute(1, 2, 0)
            sample_image = (sample_image.detach().cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
            video_writer.write(sample_image)
            # images.append(image)
    # print(f'second: {(time.time()-st)}')

    if do_cloning:
        # collect data
        tet_optim.optim.zero_grad()
        tet_optim.vertex_optim.zero_grad()
        tet_optim.sh_optim.zero_grad()


        cmap = plt.get_cmap("jet")
        sampled_cameras = [train_cameras[i] for i in densification_sampler.nextids()]
        tet_rgbs_grad = torch.zeros((model.indices.shape[0]), device=device)
        tet_count = torch.zeros((model.indices.shape[0]), device=device)
        for camera in sampled_cameras:
            with torch.no_grad():
                target = camera.original_image.cuda()
                tet_grad, extras = render_err(target, camera, model, tile_size=args.tile_size, lambda_ssim=args.clone_lambda_ssim)
                # tet_rgbs_grad = torch.maximum(tet_grad, tet_rgbs_grad)
                tet_grad = tet_grad * extras['tet_area'].clip(min=1, max=50)
                # tet_rgbs_grad = tet_grad + tet_rgbs_grad
                # tet_count += extras['mask']
                tet_rgbs_grad = torch.maximum(tet_grad, tet_rgbs_grad)
        torch.cuda.empty_cache()
        # tet_rgbs_grad = tet_rgbs_grad / tet_count.clip(min=1)
        #         tet_rgbs_grad = tet_grad + tet_rgbs_grad
        #         tet_count += extras['mask']
        # torch.cuda.empty_cache()
        # tet_rgbs_grad = tet_rgbs_grad / tet_count.clip(min=1)


        with torch.no_grad():
            render_tensor = tet_rgbs_grad
            tensor_min, tensor_max = render_tensor.min(), render_tensor.max()
            normalized_tensor = (render_tensor - tensor_min) / (tensor_max - tensor_min)

            # Convert to RGB (NxMx3) using the colormap
            tet_grad_color = torch.as_tensor(cmap(normalized_tensor.cpu().numpy())).float().cuda()
            tet_grad_color[:, 3] = model.get_cell_values(camera)[:, 3]
            render_pkg = render(sample_camera, model, cell_values=tet_grad_color, tile_size=args.tile_size)

            image = render_pkg['render']
            image = image.permute(1, 2, 0)
            image = (image.detach().cpu().numpy() * 255).clip(min=0, max=255).astype(np.uint8)
            imageio.imwrite(args.output_path / f'grad{iteration}.png', image)
            imageio.imwrite(args.output_path / f'im{iteration}.png', cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))

            del render_pkg, image, render_tensor

        with torch.no_grad():
            # tet_volume = tetra_volume(model.vertices, model.indices)
            # large_enough = tet_volume > args.min_clone_size
            # tet_rgbs_grad = torch.where(large_enough, tet_rgbs_grad, 0)
            target_addition = target_num((iteration - args.densify_start) // args.densify_interval + 1) - model.vertices.shape[0]
            ic(target_addition, (iteration - args.densify_start) // args.densify_interval + 1)
            if target_addition > 0:
                rgbs_threshold = torch.sort(tet_rgbs_grad).values[-min(int(target_addition), tet_rgbs_grad.shape[0])]
                clone_mask = (tet_rgbs_grad > rgbs_threshold)

                # binary_color = clone_mask.float().reshape(-1, 1).expand(-1, 3).contiguous()
                binary_color = torch.zeros_like(tet_grad_color)
                binary_color[clone_mask, 0] = normalized_tensor[clone_mask]
                binary_color[~clone_mask, 1] = normalized_tensor[~clone_mask]
                binary_color[:, 3] = tet_grad_color[:, 3]
                render_pkg = render(sample_camera, model, cell_values=binary_color, tile_size=args.tile_size)
                image = render_pkg['render']
                binary_im = (image.permute(1, 2, 0)*255).clip(min=0, max=255).cpu().numpy().astype(np.uint8)

                imageio.imwrite(args.output_path / f'densify{iteration}.png', binary_im)

                # rgbs_grad, vertex_grad = tet_optim.get_tracker_predicates() 
                # reduce_type = "sum"
                # # tet_std = torch.std(model.vertex_rgbs_param[model.indices], dim=1).max(dim=1).values
                # tet_rgbs_grad = rgbs_grad[model.indices].sum(dim=1)
                # tet_vertex_grad = vertex_grad[model.indices].sum(dim=1)
                # clone_mask = (tet_rgbs_grad > rgbs_threshold) | (tet_vertex_grad > vertex_threshold)
                clone_indices = model.indices[clone_mask]

                tet_optim.split(clone_indices, args.split_mode)

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
                torch.cuda.empty_cache()
                last_densify = iteration

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

    if do_delaunay:
        st = time.time()
        model.update_triangulation()
        # print(f'update: {(time.time()-st)}')
        # torch.cuda.empty_cache()

avged_psnrs = [sum(v)/len(v) for v in psnrs if len(v) == len(train_cameras)]
plt.plot(range(len(avged_psnrs)), avged_psnrs)
plt.show()
# mediapy.write_video(args.output_path / "training.mp4", images)
video_writer.release()

with (args.output_path / "alldata.json").open("w") as f:
    all_data = dict(**args.as_dict(), 
        psnr = avged_psnrs[-1] if len(avged_psnrs) > 0 else 0,
    )
    json.dump(all_data, f, cls=CustomEncoder)

torch.cuda.synchronize()
torch.cuda.empty_cache()

with torch.no_grad():
    epath = cam_util.generate_cam_path(train_cameras, 400)
    eimages = []
    for camera in tqdm(epath):
        render_pkg = render(camera, model, tile_size=args.tile_size)
        image = render_pkg['render']
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        eimages.append(image)

mediapy.write_video(args.output_path / "rotating.mp4", eimages)
model.save2ply(args.output_path / "ckpt.ply", sample_camera)
torch.save(model.state_dict(), args.output_path / "ckpt.pth")

test_util.evaluate_and_save(model, test_cameras, args.output_path, args.tile_size)