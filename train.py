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
from utils.lib_bilagrid import BilateralGrid, total_variation_loss, slice
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, ChainedScheduler
import gc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyvista as pv


torch.set_num_threads(1)



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
args.dataset_path = Path("/data/nerf_datasets/360/garden")
args.output_path = Path("output/test/")
args.iterations = 10000
args.max_steps = 20000
args.ckpt = ""
args.render_train = False

# Light Settings
args.max_sh_deg = 3
args.sh_interval = 0
args.sh_step = 1

# iNGP Settings
args.encoding_lr = 6e-3
args.final_encoding_lr = 6e-4
args.network_lr = 1e-3
args.final_network_lr = 1e-3
args.hidden_dim = 64
args.scale_multi = 1.0
args.log2_hashmap_size = 22
args.per_level_scale = 2
args.L = 10
args.density_offset = -4
args.weight_decay = 0.01
args.hashmap_dim = 4
args.grad_clip = 1e-2
args.spike_duration = 150
args.k_samples = 1
args.trunc_sigma = 0.3

args.density_lr = 5e-5
args.color_lr = 5e-5
args.gradient_lr = 5e-5
args.sh_lr = 5e-5

# Vertex Settings
args.lr_delay = 50
args.vert_lr_delay = 50
args.vertices_lr = 1e-4
args.final_vertices_lr = 5e-8
args.vertices_lr_delay_multi = 1e-8
args.vertices_beta = [0.9, 0.99]
args.contract_vertices = True
args.start_clip_multi = 1e-3
args.end_clip_multi = 1e-3
args.delaunay_start = 17000
args.freeze_start = 17000
args.ext_convex_hull = False

# Distortion Settings
args.lambda_dist = 1e-5

# Clone Settings
args.num_samples = 200
args.p_norm = 100
args.clone_lambda_ssim = 0.2
args.split_std = 0.1
args.split_mode = "split_point"
args.clone_schedule = "quadratic"
args.min_tet_count = 4
args.prune_density_threshold = 0.0
args.densify_start = 3000
args.densify_end = 15000
args.densify_interval = 500
args.budget = 2_000_000
args.lambda_noise = 0.0
args.perturb_t = 1-0.005
args.noise_start = 2000
args.clone_velocity = 0.1
args.speed_mul = 100
args.clone_min_alpha = 1/255
args.clone_min_density = 1e-3
args.normalize_err = False
args.lambda_tetvar = 0.1
args.percent_split = 0.9
args.density_t = 1.0

args.lambda_ssim = 0.2
args.base_min_t = 0.2
args.sample_cam = 4
args.data_device = 'cpu'
args.lambda_alpha = 0.0
args.lambda_density = 0.0
args.lambda_color = 0.0
args.lambda_tv = 0.0
args.density_threshold = 0.0
args.voxel_size = 0.05
args.init_repeat = 10

# Bilateral grid arguments
# Bilateral grid parameters
args.use_bilateral_grid = False
args.bilateral_grid_shape = [16, 16, 8]
args.bilateral_grid_lr = 0.003  # Match gsplat's default
args.lambda_tv_grid = 0.0

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
                                current_sh_deg = args.max_sh_deg if args.sh_interval <= 0 else 0,
                                **args.as_dict())
min_t = args.min_t = args.base_min_t * model.scene_scaling.item()
ic(args.min_t)

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

print("Encoding LR")
xs = list(range(args.iterations))
ys = [tet_optim.encoder_scheduler_args(x) for x in xs]
fig = tpl.figure()
fig.plot(xs, ys, width=150, height=20)
fig.show()

densification_sampler = SimpleSampler(len(train_cameras), args.num_samples)

# ----- Initialize bilateral grid if enabled -----
bil_grids = None
bil_optimizer = None
if args.use_bilateral_grid:
    print("\nInitializing Bilateral Grid:")
    print(f"- Grid shape: {args.bilateral_grid_shape}")
    print(f"- Learning rate: {args.bilateral_grid_lr}")
    print(f"- TV loss weight: {args.lambda_tv_grid}")
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

# cc_locations = []
vert_alive = torch.ones((model.contracted_vertices.shape[0]), dtype=bool, device=device)

tet_optim.build_tv()
progress_bar = tqdm(range(args.iterations))
torch.cuda.empty_cache()
for iteration in progress_bar:
    delaunay_interval = 10 if iteration < args.delaunay_start else 100
    do_delaunay = iteration % delaunay_interval == 0 and iteration < args.freeze_start
    do_freeze = iteration == args.freeze_start
    do_cloning = max(iteration - args.densify_start, 0) % args.densify_interval == 0 and args.densify_end > iteration >= args.densify_start
    do_sh_up = not args.sh_interval == 0 and iteration % args.sh_interval == 0 and iteration > 0
    do_sh_step = iteration % args.sh_step == 0

    if len(inds) == 0:
        inds = list(range(len(train_cameras)))
        random.shuffle(inds)
        psnrs.append([])
    ind = inds.pop()
    camera = train_cameras[ind]
    target = camera.original_image.cuda()

    st = time.time()
    bg = 0
    render_pkg = render(camera, model, bg=bg, scene_scaling=model.scene_scaling, clip_multi=tet_optim.clip_multi, **args.as_dict())
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #          profile_memory=True, with_stack=True) as prof:
    #     with record_function("model_inference"):
    #         render_pkg = render(camera, model, min_t=model.scene_scaling * 0.1, **args.as_dict())

    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_memory_usage", row_limit=10))
    # prof.export_chrome_trace("trace.json")

    # torch.cuda.synchronize()
    # print(f'render: {(time.time()-st)}')
    image = render_pkg['render']#.clip(min=0, max=1)
    vert_alive |= render_pkg['vert_alive'][:vert_alive.shape[0]]

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

    l1_loss = (target - image).abs().mean()
    l2_loss = ((target - image)**2).mean()
    reg = tet_optim.regularizer(render_pkg)
    ssim_loss = (1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))).clip(min=0, max=1)
    dl_loss = render_pkg['distortion_loss']
    loss = (1-args.lambda_ssim)*l1_loss + \
           args.lambda_ssim*ssim_loss + \
           reg + \
           tet_optim.lambda_dist(iteration) * dl_loss

    # ----- Add total variation loss for bilateral grid if enabled -----
    tvloss = None
    if args.use_bilateral_grid:
        # Use the configurable lambda_tv parameter (default is 10.0)
        tvloss = args.lambda_tv_grid * total_variation_loss(bil_grids.grids)
        loss += tvloss
    # --------------------------------------------------------------

    mask = render_pkg['mask']
    cc = render_pkg['normed_cc']
    st = time.time()
    # # loss += args.lambda_alpha * (-4 * alphas * (alphas-1)).mean()
    # loss += args.lambda_alpha * - ((alphas * safe_math.safe_log(alphas) + (1-alphas) * safe_math.safe_log(1-alphas))).mean()
    # x = render_pkg['density'][mask]
    # loss += args.lambda_density * (x * (-x**2).exp()).mean()
    # tet_optim.clip_gradient(args.grad_clip)

    loss.backward()

    tet_optim.main_step()
    tet_optim.main_zero_grad()

    if do_sh_step and tet_optim.sh_optim is not None:
        tet_optim.sh_optim.step()
        tet_optim.sh_optim.zero_grad()

    if do_delaunay:
        if iteration > args.noise_start:
            alphas = compute_alpha(model.indices, model.vertices, render_pkg['density'], mask)
            v_perturb = compute_v_perturbation(
                model.indices, model.vertices, cc, render_pkg['density'],
                mask, render_pkg['cc_sensitivity'],
                tet_optim.vertex_lr, k=100, t=args.perturb_t)
            model.perturb_vertices(args.lambda_noise * v_perturb)
        # circumcenters = model.get_circumcenters()
        # cc_locations.append(
        #     model.contract(circumcenters.detach()).cpu().numpy()
        # )
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

    if do_cloning:
        with torch.no_grad():
            # collect data
            # print(f"Culling {(~vert_alive).sum()} dead vertices")
            # tet_optim.remove_points(vert_alive)

            sampled_cameras = [train_cameras[i] for i in densification_sampler.nextids()]
            tet_moments = torch.zeros((model.indices.shape[0], 4), device=device)
            tet_votes = torch.zeros((model.indices.shape[0], 4), device=device)
            tet_count = torch.zeros((model.indices.shape[0]), device=device)
            tet_size = torch.zeros((model.indices.shape[0]), device=device)

            total_split_votes = torch.zeros((model.indices.shape[0], 2), device=device)
            total_grow_moments = torch.zeros((model.indices.shape[0], 3), device=device)
            total_grow_votes = torch.zeros((model.indices.shape[0], 2), device=device)
            split_rays = torch.zeros((model.indices.shape[0], 2, 6), device=device)
            for camera in sampled_cameras:
                target = camera.original_image.cuda()
                image_votes, extras = render_err(target, camera, model,
                                             scene_scaling=model.scene_scaling,
                                             tile_size=args.tile_size,
                                             density_t=args.density_t,
                                             lambda_ssim=args.clone_lambda_ssim)
                density = extras['cell_values'][:, 0]
                tc = extras['tet_count']
                # num_votes = extras['pixel_err'].sum()

                # -----------------------------------------------------------------------
                # Split
                # -----------------------------------------------------------------------
                split_mask = (tc > 2000) | (tc < 4)
                s0 = image_votes[:, 0] 
                s1 = image_votes[:, 1]
                s2 = image_votes[:, 2]
                split_mu = safe_math.safe_div(s1, s0)
                split_std = safe_math.safe_div(s2, s0) - split_mu**2
                split_std[s0 < 1e-4] = 0
                split_std[split_mask] = 0

                split_votes = s0 * split_std

                w = image_votes[:, 12:13]
                seg_exit = safe_math.safe_div(image_votes[:, 9:12], w)
                seg_enter = safe_math.safe_div(image_votes[:, 6:9], w)
                rays = torch.cat([ seg_enter, seg_exit ], dim=1)

                # ---------- keep the *three* candidates then drop the worst -----------
                # votes: shape (N, 3)   rays: shape (N, 3, 3)
                votes_3 = torch.cat([total_split_votes, split_votes.unsqueeze(1)], dim=1)
                rays_3  = torch.cat([split_rays,    rays.unsqueeze(1)],    dim=1)

                votes_sorted, idx_sorted = votes_3.sort(dim=1, descending=True)
                total_split_votes = votes_sorted[:, :2]
                split_rays    = torch.gather(
                    rays_3,
                    dim=1,
                    index=idx_sorted[:, :2].unsqueeze(-1).expand(-1, -1, 6)
                )

                # -----------------------------------------------------------------------
                # Grow
                # -----------------------------------------------------------------------
                grow_mask = (tc < 2000) & (tc > 4)
                total_grow_moments[grow_mask] += image_votes[grow_mask, 3:6]
                tet_moments[grow_mask, :3] += image_votes[grow_mask, 13:16]
                tet_moments[grow_mask, 3] += image_votes[grow_mask, 3]
                # s0 = image_votes[:, 3] 
                # s1 = image_votes[:, 4]
                # s2 = image_votes[:, 5]
                # grow_mu = safe_math.safe_div(s1, s0)
                # grow_std = safe_math.safe_div(s2, s0) - grow_mu**2
                # grow_std[s0 < 1e-4] = 0
                # grow_votes = grow_std
                # grow_votes[grow_mask] = 0
                # total_grow_votes = torch.sort(torch.cat([
                #     total_grow_votes, grow_votes.reshape(-1, 1)
                # ], dim=1), dim=1).values[:, 1:]
                tet_count += tc > 0
                tet_size += tc
            torch.cuda.empty_cache()
            tet_size = tet_size / tet_count.clip(min=1)

            split_score = total_split_votes.sum(dim=1)

            s0 = total_grow_moments[:, 0] 
            s1 = total_grow_moments[:, 1]
            s2 = total_grow_moments[:, 2]
            grow_mu = safe_math.safe_div(s1, s0)
            grow_std = safe_math.safe_div(s2, s0) - grow_mu**2
            grow_std[s0 < 1] = 0
            grow_score = s0 * grow_std
            alphas = compute_alpha(model.indices, model.vertices, model.calc_tet_density()).reshape(-1)
            grow_score[alphas < args.clone_min_alpha] = 0
            split_score[alphas < args.clone_min_alpha] = 0
            # grow_score[tet_size < 20] = 0
            # grow_score[tet_size > 8000] = 0


            target = target_num((iteration - args.densify_start) // args.densify_interval + 1)
            target_addition = target - model.vertices.shape[0]
            if target_addition > 0:
                target_split = args.percent_split * target_addition
                target_grow = (1-args.percent_split) * target_addition
                split_mask = torch.zeros((split_score.shape[0]), device=device, dtype=bool)
                grow_mask = torch.zeros((split_score.shape[0]), device=device, dtype=bool)
                grow_mask[torch.topk(grow_score, int(target_grow))[1]] = True
                grow_mask &= grow_score > 0
                split_score[grow_mask] = 0

                split_mask[torch.topk(split_score, int(target_split))[1]] = True
                split_mask &= split_score > 0
                # grow_score[split_mask] = 0
                clone_mask = split_mask | grow_mask

                f = clone_mask.float().reshape(-1, 1).expand(-1, 4).clone()
                f[:, :3] = torch.rand_like(f[:, :3])
                f[:, 3] *= 1.0
                binary_im = render_debug(f, model, sample_camera, 10)
                imageio.imwrite(args.output_path / f'densify{iteration}.png', binary_im)
                clone_indices = model.indices[clone_mask]

                split_ratio_img = render_debug(split_score.reshape(-1, 1), model, sample_camera)
                grow_ratio_img = render_debug(grow_score.reshape(-1, 1), model, sample_camera)
                imageio.imwrite(args.output_path / f'split{iteration}.png', split_ratio_img)
                imageio.imwrite(args.output_path / f'grow{iteration}.png', grow_ratio_img)
                imageio.imwrite(args.output_path / f'im{iteration}.png', cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
                # di = render_pkg['distortion_img'].detach().cpu().numpy().reshape
                # di = (cmap(di / di.max())*255).clip(min=0, max=255).astype(np.uint8)
                # imageio.imwrite(args.output_path / f'di{iteration}.png', di)

                split_point = torch.zeros((model.indices.shape[0], 3), device=device)
                split_point[grow_mask] = safe_math.safe_div(tet_moments[:, :3], tet_moments[:, 3:4])[grow_mask]
                split_point[split_mask] = get_approx_ray_intersections(split_rays)[split_mask]
                tet_optim.split(clone_indices, split_point[clone_mask], args.split_mode, args.prune_density_threshold)


                out = f"#Split: {split_mask.sum()} "
                out += f"#Grow: {grow_mask.sum()} "
                out += f"#T_Split: {target_split} "
                out += f"#T_Grow: {target_grow} "
                out += f"Grow Ratio: {total_grow_votes.mean()} "
                out += f"Split Ratio: {total_split_votes.mean()} "
                out += f"target_addition: {target_addition} "
                print(out)

                # clone vertices
                raw_verts = model.contracted_vertices
                stored_state = tet_optim.vertex_optim.get_state_by_name('contracted_vertices')
                velocity = stored_state['exp_avg'] * args.speed_mul
                J_d = topo_utils.contraction_jacobian_d_in_chunks(
                    model.vertices[:model.contracted_vertices.shape[0]]).reshape(-1, 1)
                speed = torch.linalg.norm(velocity * J_d, dim=1)
                mask = speed > args.clone_velocity
                new_verts = (raw_verts + velocity)[mask]
                print(f"Adding {new_verts.shape[0]} new verts using velocity. Mean velocity: {speed.mean()}")
                tet_optim.add_points(new_verts, raw_verts=True)

                gc.collect()
                torch.cuda.empty_cache()
                vert_alive = torch.zeros((model.contracted_vertices.shape[0]), dtype=bool, device=device)

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
        tet_optim.update_triangulation(density_threshold=args.density_threshold, high_precision=do_freeze)

avged_psnrs = [sum(v)/len(v) for v in psnrs if len(v) == len(train_cameras)]
video_writer.release()

torch.cuda.synchronize()
torch.cuda.empty_cache()

model.save2ply(args.output_path / "ckpt.ply")
torch.save(model.state_dict(), args.output_path / "ckpt.pth")

if args.render_train:
    splits = zip(['train', 'test'], [train_cameras, test_cameras])
else:
    splits = zip(['test'], [test_cameras])
results = test_util.evaluate_and_save(model, splits, args.output_path, args.tile_size, min_t)

with (args.output_path / "results.json").open("w") as f:
    all_data = dict(
        psnr = avged_psnrs[-1] if len(avged_psnrs) > 0 else 0,
        **results
    )
    json.dump(all_data, f, cls=CustomEncoder)

with (args.output_path / "alldata.json").open("w") as f:
    all_data = dict(**args.as_dict(), 
        psnr = avged_psnrs[-1] if len(avged_psnrs) > 0 else 0,
        **results
    )
    json.dump(all_data, f, cls=CustomEncoder)

with torch.no_grad():
    epath = cam_util.generate_cam_path(train_cameras, 400)
    eimages = []
    for camera in tqdm(epath):
        render_pkg = render(camera, model, min_t=min_t, tile_size=args.tile_size)
        image = render_pkg['render']
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        eimages.append(pad_image2even(image))
mediapy.write_video(args.output_path / "rotating.mp4", eimages)
