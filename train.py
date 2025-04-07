import cv2
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
from utils.graphics_utils import tetra_volume
import termplotlib as tpl
from delaunay_rasterization.internal.alphablend_tiled_slang import render_constant_color
from utils.lib_bilagrid import BilateralGrid, total_variation_loss, slice
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, ChainedScheduler
import gc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyvista as pv


torch.set_num_threads(1)


cmap = plt.get_cmap("jet")

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

eps = torch.finfo(torch.float).eps
args = Args()
args.tile_size = 16
args.image_folder = "images_4"
args.eval = False
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/test/")
args.iterations = 10000
args.ckpt = ""

# Light Settings
args.light_offset = -3
args.lights_lr = 1e-4
args.final_lights_lr = 1e-4
args.color_lr = 1e-2
args.final_color_lr = 1e-2
args.num_lights = 2
args.sh_interval = 1000

# iNGP Settings
args.encoding_lr = 3e-3
args.final_encoding_lr = 3e-4
args.network_lr = 3e-4
args.final_network_lr = 3e-4
args.hidden_dim = 64
args.scale_multi = 1.0
args.log2_hashmap_size = 22
args.per_level_scale = 2
args.L = 10
args.density_offset = -4
args.weight_decay = 0.01
args.hashmap_dim = 4
args.grad_clip = 1e-2

# Vertex Settings
args.lr_delay = 50
args.vert_lr_delay = 50
args.vertices_lr = 1e-3
args.final_vertices_lr = 1e-5
args.max_steps = 30000
args.vertices_lr_delay_multi = 1e-8
args.vertices_beta = [0.9, 0.99]
args.contract_vertices = False
args.clip_multi = 50.0
args.delaunay_start = 17000
args.freeze_start = 25000

# Distortion Settings
args.lambda_dist = 1e-5
args.ladder_p = -0.25
args.pre_multi = 10000

# Clone Settings
args.p_norm = 100
args.clone_lambda_ssim = 0.2
args.split_std = 0.1
args.split_mode = "barycentric"
args.clone_schedule = "quadratic"
args.min_tet_count = 4
args.prune_alpha_thres = 1/255
args.densify_start = 2000
args.densify_end = 15000
args.num_samples = 200
args.densify_interval = 500
args.budget = 2_000_000
args.lambda_noise = 1e-2

args.lambda_ssim = 0.2
args.base_min_t = 0.01
args.sample_cam = 4
args.data_device = 'cuda'

# Bilateral grid arguments
# Bilateral grid parameters
args.use_bilateral_grid = False
args.bilateral_grid_shape = [16, 16, 8]
args.bilateral_grid_lr = 0.003  # Match gsplat's default
args.lambda_tv: float = 10.0

args = Args.from_namespace(args.get_parser().parse_args())

args.output_path.mkdir(exist_ok=True, parents=True)

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device=args.data_device, eval=args.eval)

args.num_samples = min(len(train_cameras), args.num_samples)

device = torch.device('cuda')
if len(args.ckpt) > 0: 
    model = Model.load_ckpt(Path(args.ckpt), device)
else:
    model = Model.init_from_pcd(scene_info.point_cloud, train_cameras, device,
                                max_lights = args.num_lights if args.sh_interval <= 0 else 0,
                                **args.as_dict())
min_t = model.scene_scaling * args.base_min_t

tet_optim = TetOptimizer(model, **args.as_dict())
if args.eval:
    sample_camera = test_cameras[args.sample_cam]
else:
    sample_camera = train_cameras[args.sample_cam]

camera_inds = {}
camera_inds_back = {}
for i, camera in enumerate(train_cameras):
    camera_inds[camera.uid] = i
    camera_inds_back[i] = camera.uid

images = []
psnrs = [[]]
inds = []

num_densify_iter = args.densify_end - args.densify_start
N = num_densify_iter // args.densify_interval + 1
S = model.vertices.shape[0]

def target_num(x):
    if args.clone_schedule == "linear":
        k = (args.budget - S) // N
        return k * x + S
    elif args.clone_schedule == "quadratic":
        k = 2 * (args.budget - S) // N
        a = (args.budget - S - k * N) // N**2
        return a * x**2 + k * x + S
    else:
        raise Exception(f"Clone Schedule: {args.clone_schedule} is not supported")

xs = list(range(N))
ys = [target_num(i+1) for i in xs]
fig = tpl.figure()
fig.plot(xs, ys, width=100, height=20)
fig.show()

densification_sampler = SimpleSampler(len(train_cameras), args.num_samples)

# ----- Initialize bilateral grid if enabled -----
bil_grids = None
bil_optimizer = None
if args.use_bilateral_grid:
    print("\nInitializing Bilateral Grid:")
    print(f"- Grid shape: {args.bilateral_grid_shape}")
    print(f"- Learning rate: {args.bilateral_grid_lr}")
    print(f"- TV loss weight: {args.lambda_tv}")
    bil_grids = BilateralGrid(
        len(train_cameras),
        grid_X=args.bilateral_grid_shape[0],
        grid_Y=args.bilateral_grid_shape[1],
        grid_W=args.bilateral_grid_shape[2],
    ).to("cuda")
    bil_optimizer = torch.optim.Adam([bil_grids.grids], lr=args.bilateral_grid_lr, eps=1e-15)
    
    # Create a chained scheduler with warmup like in gsplat
    # First 1000 iterations: linear warmup from 1% to 100% of learning rate
    # Then exponential decay to 1% of initial learning rate by the end of training
    bil_warmup = LinearLR(bil_optimizer, start_factor=0.01, total_iters=1000)
    bil_decay = ExponentialLR(bil_optimizer, gamma=0.01**(1.0/args.iterations))
    bil_scheduler = ChainedScheduler([bil_warmup, bil_decay])
    
    print(f"- Number of grids: {len(train_cameras)}")
    print("- Using LinearLR warmup + ExponentialLR decay scheduler")
    print("Bilateral Grid initialized successfully!\n")
# ------------------------------------------------

video_writer = cv2.VideoWriter(str(args.output_path / "training.mp4"), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'avc1'), 30,
                               pad_hw2even(sample_camera.image_width, sample_camera.image_height))

cc_locations = []
progress_bar = tqdm(range(args.iterations))
torch.cuda.empty_cache()
for iteration in progress_bar:
    delaunay_interval = 10 if iteration < args.delaunay_start else 100
    do_delaunay = iteration % delaunay_interval == 0 and iteration < args.freeze_start
    do_cloning = max(iteration - args.densify_start, 0) % args.densify_interval == 0 and args.densify_end > iteration >= args.densify_start
    do_sh_up = not args.sh_interval == 0 and iteration % args.sh_interval == 0 and iteration > 0
    do_sh_step = iteration % 1 == 0

    if len(inds) == 0:
        inds = list(range(len(train_cameras)))
        random.shuffle(inds)
        psnrs.append([])
    ind = inds.pop()
    camera = train_cameras[ind]
    target = camera.original_image.cuda()

    st = time.time()
    bg = 0
    render_pkg = render(camera, model, bg=bg, min_t=min_t, **args.as_dict())
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #          profile_memory=True, with_stack=True) as prof:
    #     with record_function("model_inference"):
    #         render_pkg = render(camera, model, min_t=model.scene_scaling * 0.1, **args.as_dict())

    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_memory_usage", row_limit=10))
    # prof.export_chrome_trace("trace.json")

    # torch.cuda.synchronize()
    # print(f'render: {(time.time()-st)}')
    image = render_pkg['render'].clip(min=0, max=1)

    # ----- Apply bilateral grid transformation if enabled -----
    if args.use_bilateral_grid:
        # Get camera ID for this viewpoint
        camera_id = camera_inds[camera.uid]
        
        # Create normalized pixel coordinates [0,1]
        h, w = image.shape[1], image.shape[2]
        y_coords, x_coords = torch.meshgrid(
            (torch.arange(h, device="cuda") + 0.5) / h,
            (torch.arange(w, device="cuda") + 0.5) / w,
            indexing="ij"
        )
        coords = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
        
        # Reshape image for bilateral grid transformation
        img_for_bil = image.permute(1, 2, 0).reshape(-1, 3)
        
        # Create image IDs tensor (all pixels have same image ID)
        img_ids = torch.full((img_for_bil.shape[0],), camera_id, 
                              device="cuda", dtype=torch.long)
        
        # Apply bilateral transformation
        transformed = slice(bil_grids, coords, img_for_bil, img_ids)
        
        # Reshape back to original format
        image = transformed["rgb"].reshape(h, w, 3).permute(2, 0, 1)
    # --------------------------------------------------------

    l2_loss = ((target - image)**2).mean()
    reg = tet_optim.regularizer()
    ssim_loss = 1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))
    dl_loss = render_pkg['distortion_loss']
    loss = (1-args.lambda_ssim)*l2_loss + args.lambda_ssim*ssim_loss + reg + args.lambda_dist * dl_loss

    mask = render_pkg['mask']
    cc = render_pkg['normed_cc']
    reg_perturb = compute_perturbation(model.indices, model.vertices, cc, render_pkg['density'],
                                   mask, render_pkg['cc_sensitivity'],
                                   tet_optim.vertex_lr, k=100, t=(1-0.005))
    reg = args.lambda_noise * reg_perturb
    loss = loss + reg

    # ----- Add total variation loss for bilateral grid if enabled -----
    tvloss = None
    if args.use_bilateral_grid:
        # Use the configurable lambda_tv parameter (default is 10.0)
        tvloss = args.lambda_tv * total_variation_loss(bil_grids.grids)
        loss += tvloss
    # --------------------------------------------------------------

    st = time.time()
    loss.backward()
    # tet_optim.clip_gradient(args.grad_clip)

    tet_optim.main_step()
    tet_optim.main_zero_grad()

    if do_sh_step:
        tet_optim.sh_optim.step()
        tet_optim.sh_optim.zero_grad()

    if do_delaunay:
        circumcenters = model.get_circumcenters()
        cc_locations.append(
            model.contract(circumcenters.detach()).cpu().numpy()
        )
        tet_optim.vertex_optim.step()
        tet_optim.vertex_optim.zero_grad()

    # ----- Update bilateral grid if enabled -----
    if args.use_bilateral_grid:
        bil_optimizer.step()
        bil_optimizer.zero_grad(set_to_none=True)
        bil_scheduler.step()
    # ------------------------------------------

    tet_optim.update_learning_rate(iteration)

    if do_sh_up:
        model.sh_up()

    with torch.no_grad():
        if iteration % 10 == 0:
            render_pkg = render(sample_camera, model, min_t=min_t, tile_size=args.tile_size)
            sample_image = render_pkg['render']
            sample_image = sample_image.permute(1, 2, 0)
            sample_image = (sample_image.detach().cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
            video_writer.write(pad_image2even(sample_image))

            # di = render_pkg['distortion_img'].detach().cpu().numpy()
            # di = (cmap(di / di.max())*255).clip(min=0, max=255).astype(np.uint8)
            # imageio.imwrite('di.png', di)
            # images.append(image)
    # print(f'second: {(time.time()-st)}')

    if do_cloning:
        # collect data
        tet_optim.optim.zero_grad()
        tet_optim.vertex_optim.zero_grad()
        tet_optim.sh_optim.zero_grad()


        sampled_cameras = [train_cameras[i] for i in densification_sampler.nextids()]
        tet_rgbs_grad = torch.zeros((model.indices.shape[0]), device=device)
        tet_count = torch.zeros((model.indices.shape[0]), device=device)
        for camera in sampled_cameras:
            with torch.no_grad():
                target = camera.original_image.cuda()
                tet_grad, extras = render_err(target, camera, model, tile_size=args.tile_size, lambda_ssim=args.clone_lambda_ssim)
                visible = extras['tet_count'] > args.min_tet_count
                if args.p_norm > 10:
                    tet_rgbs_grad[visible] = torch.maximum(tet_grad, tet_rgbs_grad)[visible]
                else:
                    tet_count += visible
                    tet_rgbs_grad[visible] = (tet_rgbs_grad + tet_grad.abs().clip(min=eps).pow(args.p_norm))[visible]
                del tet_grad, extras
        torch.cuda.empty_cache()
        if args.p_norm < 10:
            tet_rgbs_grad = tet_rgbs_grad / tet_count.clip(min=1)

        with torch.no_grad():
            render_tensor = tet_rgbs_grad
            tensor_min, tensor_max = render_tensor.min(), render_tensor.max()
            normalized_tensor = (render_tensor - tensor_min) / (tensor_max - tensor_min)

            # Convert to RGB (NxMx3) using the colormap
            tet_grad_color = torch.as_tensor(cmap(normalized_tensor.cpu().numpy())).float().cuda()
            _, features = model.get_cell_values(camera)
            tet_grad_color[:, 3] = features[:, 0]
            render_pkg = render_constant_color(model.indices, model.vertices, None, sample_camera, cell_values=tet_grad_color)

            image = render_pkg['render']
            image = image.permute(1, 2, 0)
            image = (image.detach().cpu().numpy() * 255).clip(min=0, max=255).astype(np.uint8)
            imageio.imwrite(args.output_path / f'grad{iteration}.png', image)
            imageio.imwrite(args.output_path / f'im{iteration}.png', cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))

            del render_pkg, image, render_tensor

        with torch.no_grad():
            target = target_num((iteration - args.densify_start) // args.densify_interval + 1)
            target_addition = target - model.vertices.shape[0]
            if target_addition > 0:
                rgbs_threshold = torch.sort(tet_rgbs_grad).values[-min(int(target_addition), tet_rgbs_grad.shape[0])]
                clone_mask = (tet_rgbs_grad > rgbs_threshold)

                binary_color = torch.zeros_like(tet_grad_color)
                binary_color[clone_mask, 0] = normalized_tensor[clone_mask]
                binary_color[~clone_mask, 1] = normalized_tensor[~clone_mask]
                binary_color[:, 3] = tet_grad_color[:, 3]
                render_pkg = render_constant_color(model.indices, model.vertices, None, sample_camera, min_t=min_t, cell_values=binary_color)
                image = render_pkg['render']
                binary_im = (image.permute(1, 2, 0)*255).clip(min=0, max=255).cpu().numpy().astype(np.uint8)
                imageio.imwrite(args.output_path / f'densify{iteration}.png', binary_im)

                clone_indices = model.indices[clone_mask]

                tet_optim.split(clone_indices, args.split_mode, args.prune_alpha_thres)


                out = f"#RGBS Clone: {(tet_rgbs_grad > rgbs_threshold).sum()} "
                out += f"∇RGBS: {tet_rgbs_grad.mean()} "
                out += f"target_addition: {target_addition} "
                print(out)
            gc.collect()
            torch.cuda.empty_cache()

    psnr = 20 * math.log10(1.0 / math.sqrt(l2_loss.detach().cpu().item()))
    psnrs[-1].append(psnr)

    disp_ind = max(len(psnrs)-2, 0)
    avg_psnr = sum(psnrs[disp_ind]) / max(len(psnrs[disp_ind]), 1)
    progress_bar.set_postfix({
        "PSNR": repr(f"{psnr:>5.2f}"),
        "Mean": repr(f"{avg_psnr:>5.2f}"),
        "#V": len(model),
        "#T": model.indices.shape[0],
        "DL": repr(f"{dl_loss:>5.2f}"),
    })

    if do_delaunay:
        st = time.time()
        model.update_triangulation()

avged_psnrs = [sum(v)/len(v) for v in psnrs if len(v) == len(train_cameras)]
video_writer.release()



def pad_point_clouds(cc_locations):
    """
    Pads all frames in cc_locations to have the same number of points
    as the frame with the maximum number of points, padding with zeros.

    Parameters:
        cc_locations (list or array): List/array of frames (num_frames, num_points, 3)

    Returns:
        np.ndarray: Padded cc_locations of shape (num_frames, max_num_points, 3)
    """
    max_num_points = max(frame.shape[0] for frame in cc_locations)

    padded_frames = []
    for frame in cc_locations:
        num_points = frame.shape[0]
        if num_points < max_num_points:
            padding = np.zeros((max_num_points - num_points, 3))
            padded_frame = np.vstack([frame, padding])
        else:
            padded_frame = frame

        padded_frames.append(padded_frame)

    return np.array(padded_frames)

# cc_locations: (num_frames, num_points, 3) numpy array
plotter = pv.Plotter()
cc_locations = pad_point_clouds(cc_locations)
points = cc_locations[0]

# Initialize the scatter plot
point_cloud = pv.PolyData(points)
scatter = plotter.add_mesh(point_cloud, render_points_as_spheres=False, point_size=2.0)

# Set consistent camera bounds
plotter.set_background('white')
plotter.show_bounds(bounds=(-2, 2, -2, 2, -2, 2))

# Create and save the animation
plotter.open_movie(str(args.output_path / 'circumcenters.mp4'), framerate=24)

for i, frame in tqdm(enumerate(cc_locations)):
    # Efficiently update points in-place
    point_cloud.points = frame

    # Update the text efficiently
    plotter.add_text(f"Frame {i+1}/{len(cc_locations)}", position='upper_edge', font_size=12, name='frame_label')

    # Write current frame
    plotter.write_frame()

    # Remove only the text (avoid clearing all actors)
    plotter.remove_actor('frame_label')

plotter.close()

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
        render_pkg = render(camera, model, min_t=min_t, tile_size=args.tile_size)
        image = render_pkg['render']
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        eimages.append(image)

mediapy.write_video(args.output_path / "rotating.mp4", eimages)
model.save2ply(args.output_path / "ckpt.ply")
torch.save(model.state_dict(), args.output_path / "ckpt.pth")

test_util.evaluate_and_save(model, test_cameras, args.output_path, args.tile_size, min_t)