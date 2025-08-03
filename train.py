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
from data import loader
import random
import time
from tqdm import tqdm
import numpy as np
from utils import cam_util
from utils.train_util import *
from delaunay_rasterization import render
# from models.vertex_color import Model, TetOptimizer
from models.ingp_color import Model, TetOptimizer
# from models.ingp_linear import Model, TetOptimizer
from fused_ssim import fused_ssim
from pathlib import Path, PosixPath
from utils.args import Args
from utils import test_util
from utils.lib_bilagrid import BilateralGrid, total_variation_loss, slice
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, ChainedScheduler
import gc
from utils.densification import collect_render_stats, apply_densification, determine_cull_mask
import mediapy
from torch import nn


torch.set_num_threads(1)

eps = torch.finfo(torch.float).eps
args = Args()
args.tile_size = 16
args.image_folder = "images_4"
args.eval = False
args.dataset_path = Path("/optane/nerf_datasets/360/garden")
args.output_path = Path("output/test/")
args.iterations = 30000
args.ckpt = ""
args.render_train = False
args.delaunay_interval = 10
args.orient_scene = True
args.freeze_features = True

# Light Settings
args.max_sh_deg = 3
args.sh_interval = 2000
args.sh_step = 1
args.bake_model = True

args.glo_dim = 0
args.glo_lr = 1e-3
args.glo_network_lr = 5e-5
args.glo_weight_decay = 1e-1
args.glo_net_decay = 1e-6

# iNGP Settings
args.base_resolution = 64
args.encoding_lr = 3e-3
args.final_encoding_lr = 3e-4
args.network_lr = 1e-3
args.final_network_lr = 1e-4
args.scale_multi = 0.35 # chosen such that 96% of the distribution is within the sphere 
args.log2_hashmap_size = 23
args.per_level_scale = 2
args.L = 6
args.density_offset = -4
args.weight_decay = 0.1
args.hashmap_dim = 16
args.percent_alpha = 0.04 # preconditioning
args.spike_duration = 500
args.hidden_dim = 64
args.sh_hidden_dim = 256
args.sh_weight_decay = 1e-5
args.sh_lr_div = 20

args.dg_init=0.1
args.g_init=0.1
args.s_init=0.1
args.d_init=0.1
args.c_init=0.1

# Vertex Settings
args.lr_delay = 0
args.vert_lr_delay = 0
args.vertices_lr = 1e-4
args.final_vertices_lr = 1e-6
args.vertices_lr_delay_multi = 1e-8
args.clip_multi = 0
args.delaunay_start = 30000

args.freeze_start = 16000
args.freeze_lr = 5e-3
args.final_freeze_lr = 1e-4
args.feature_lr = 1e-3
args.final_feature_lr = 1e-4
args.fnetwork_lr = 1e-3
args.final_fnetwork_lr = 1e-4

# Distortion Settings
args.lambda_dist = 1e-6
args.lambda_density = 0.0
args.lambda_aniso = 0.0
args.lambda_cost = 1e-3

# Clone Settings
args.num_samples = 200
args.split_std = 1e-1
args.min_tet_count = 9
args.densify_start = 2000
args.densify_end = 16000
args.densify_interval = 500
args.budget = 2_000_000
args.clone_min_alpha = 0.025
args.clone_min_density = 0.025

args.lambda_ssim = 0.2
args.base_min_t = 0.2
args.sample_cam = 1
args.data_device = 'cpu'
args.lambda_tv = 0.0
args.contrib_threshold = 0.025
args.density_threshold = 0.1
args.alpha_threshold = 0.1
args.total_thresh = 0.025
args.within_thresh = 0.4
args.density_intercept = 0.2
args.voxel_size = 0.01
args.start_threshold = 5500
args.ext_convex_hull = True

args.use_bilateral_grid = False
args.bilateral_grid_shape = [16, 16, 8]
args.bilateral_grid_lr = 0.003
args.lambda_tv_grid = 0.0
args.record_training = False
args.checkpoint_iterations = []

parser = args.get_parser()
args = Args.from_namespace(parser.parse_args())

# if a ckpt is loaded, load config, then override config with user specified flags
if len(args.ckpt) > 0: 
    config_path = Path(args.ckpt) / "config.json"
    config = Args.load_from_json(str(config_path))
    parser.set_defaults(**config.as_dict())
args = Args.from_namespace(parser.parse_args())

args.output_path.mkdir(exist_ok=True, parents=True)
# args.checkpoint_iterations.append(args.freeze_start-1)
args.checkpoint_iterations = [int(i) for i in args.checkpoint_iterations]
print(args.checkpoint_iterations)

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device=args.data_device, eval=args.eval)

np.savetxt(str(args.output_path / "transform.txt"), scene_info.transform)

args.num_samples = min(len(train_cameras), args.num_samples)

with (args.output_path / "config.json").open("w") as f:
    json.dump(args.as_dict(), f, cls=CustomEncoder)

final_iter = args.freeze_start if args.bake_model else args.iterations
device = torch.device('cuda')
if len(args.ckpt) > 0: 
    try:
        model = Model.load_ckpt(Path(args.ckpt), device, args)
        tet_optim = TetOptimizer(model, final_iter=final_iter, **args.as_dict())
    except:
        if args.freeze_features:
            from models.frozen_features import FrozenTetModel, FrozenTetOptimizer
        else:
            from models.frozen import FrozenTetModel, FrozenTetOptimizer
        model = FrozenTetModel.load_ckpt(Path(args.ckpt), device)
        tet_optim = FrozenTetOptimizer(model, final_iter=final_iter, **args.as_dict())
else:
    model = Model.init_from_pcd(scene_info.point_cloud, train_cameras, device,
                                current_sh_deg = args.max_sh_deg if args.sh_interval <= 0 else 0,
                                **args.as_dict())
    tet_optim = TetOptimizer(model, final_iter=final_iter, **args.as_dict())

min_t = args.min_t = args.base_min_t# * model.scene_scaling.item()
if args.eval:
    sample_camera = test_cameras[args.sample_cam]
    # sample_camera = train_cameras[args.sample_cam]
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

dschedule = list(range(args.densify_start, args.densify_end, args.densify_interval))

# print("Encoding LR")
# xs = list(range(args.iterations))
# ys = [tet_optim.encoder_scheduler_args(x) for x in xs]
# fig = tpl.figure()
# fig.plot(xs, ys, width=150, height=20)
# fig.show()

densification_sampler = SimpleSampler(len(train_cameras), args.num_samples)

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
    
    bil_warmup = LinearLR(bil_optimizer, start_factor=0.01, total_iters=1000)
    bil_decay = ExponentialLR(bil_optimizer, gamma=0.01**(1.0/args.iterations))
    bil_scheduler = ChainedScheduler([bil_warmup, bil_decay])
    
    print(f"- Number of grids: {len(train_cameras)}")
    print("- Using LinearLR warmup + ExponentialLR decay scheduler")
    print("Bilateral Grid initialized successfully!\n")

glo_list = lambda x: None
glo_optim = None
if args.glo_dim > 0:
    glo_list = nn.Embedding(len(train_cameras), args.glo_dim)
    with torch.no_grad():
        glo_list.weight *= 0
    glo_list = glo_list.cuda()
    glo_optim = torch.optim.Adam(glo_list.parameters(), lr=args.glo_lr)

if args.record_training:
    video_writer = cv2.VideoWriter(str(args.output_path / "training.mp4"), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'h264'), 30,
                               pad_hw2even(sample_camera.image_width, sample_camera.image_height))

progress_bar = tqdm(range(args.iterations))
torch.cuda.empty_cache()

densification_cam_buffer = []

for iteration in progress_bar:
    delaunay_interval = args.delaunay_interval if iteration < args.delaunay_start else 100
    do_delaunay = iteration % delaunay_interval == 0 and iteration < args.freeze_start
    do_freeze = iteration == args.freeze_start
    do_cloning = iteration in dschedule and iteration < args.freeze_start
    do_sh_up = not args.sh_interval == 0 and iteration % args.sh_interval == 0 and iteration > 0
    do_sh_step = iteration % args.sh_step == 0

    if do_delaunay or do_freeze:
        st = time.time()
        dt = args.density_threshold if iteration > args.start_threshold else 0
        at = args.alpha_threshold if iteration > args.start_threshold else 0
        tet_optim.update_triangulation(density_threshold=dt, alpha_threshold=at, high_precision=do_freeze)
        if do_freeze and args.bake_model and not model.frozen:
            # model.save2ply(args.output_path / "ckpt_prefreeze.ply")
            if args.freeze_features:
                from models.frozen_features import freeze_model
            else:
                from models.frozen import freeze_model
            del tet_optim
            model.eval()
            mask = determine_cull_mask(train_cameras, model, glo_list, args, device)
            model.train()
            model, tet_optim = freeze_model(model, mask, args)
            gc.collect()
            torch.cuda.empty_cache()

    if len(inds) == 0:
        inds = list(range(len(train_cameras)))
        random.shuffle(inds)
        psnrs.append([])
    ind = inds.pop()
    densification_cam_buffer.append(ind)
    camera = train_cameras[ind]
    target = camera.original_image.cuda()

    st = time.time()
    ray_jitter = torch.rand((camera.image_height, camera.image_width, 2), device=device)
    tid = torch.LongTensor([camera.uid]).cuda()
    render_pkg = render(camera, model, scene_scaling=model.scene_scaling,
                        ray_jitter=ray_jitter, glo=glo_list(tid), **args.as_dict())
    image = render_pkg['render']

    if args.use_bilateral_grid:
        camera_id = camera_inds[camera.uid]
        h, w = image.shape[1], image.shape[2]
        y_coords, x_coords = torch.meshgrid(
            (torch.arange(h, device="cuda") + 0.5) / h,
            (torch.arange(w, device="cuda") + 0.5) / w,
            indexing="ij"
        )
        coords = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
        img_for_bil = image.permute(1, 2, 0).reshape(-1, 3)
        img_ids = torch.full((img_for_bil.shape[0],), camera_id, 
                              device="cuda", dtype=torch.long)
        transformed = slice(bil_grids, coords, img_for_bil, img_ids)
        image = transformed["rgb"].reshape(h, w, 3).permute(2, 0, 1)

    l1_loss = (target - image).abs().mean()
    l2_loss = ((target - image)**2).mean()
    reg = tet_optim.regularizer(render_pkg, args.weight_decay, args.lambda_tv)
    ssim_loss = (1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))).clip(min=0, max=1)
    dl_loss = render_pkg['distortion_loss']
    a = args.density_intercept
    mask = render_pkg['mask']
    area = topo_utils.tet_surface_areas(model.vertices[model.indices])
    density = render_pkg['density'].reshape(-1)

    # density_loss = ((-(density - a)**2 / a**2 + 1).clip(min=0) * area)[mask].mean()
    density_loss = density[mask].mean()
    lambda_dist = args.lambda_dist if iteration > 1000 else 0
    lambda_density = lambda_dist * args.lambda_density if iteration > 1000 else 0
    lambda_aniso = args.lambda_aniso if iteration > 1000 else 0
    aniso_loss = model.calc_aniso_loss(render_pkg['density'])[mask].mean()
    # area = topo_utils.tet_volumes(model.vertices[model.indices])
    render_cost = (density.clip(max=2*args.density_threshold) * area.reshape(-1))[mask].mean()
    loss = (1-args.lambda_ssim)*l1_loss + \
           args.lambda_ssim*ssim_loss + \
           reg + \
           lambda_dist * dl_loss + \
           lambda_density * density_loss + \
           lambda_aniso * aniso_loss + \
           args.lambda_cost * render_cost

    if args.use_bilateral_grid:
        tvloss = args.lambda_tv_grid * total_variation_loss(bil_grids.grids)
        loss += tvloss

    if args.glo_dim > 0:
        loss += args.glo_weight_decay * (glo_list.weight**2).mean()

    mask = render_pkg['mask']
    st = time.time()

    loss.backward()

    tet_optim.main_step()
    tet_optim.main_zero_grad()

    if do_sh_step and tet_optim.sh_optim is not None:
        tet_optim.sh_optim.step()
        tet_optim.sh_optim.zero_grad()

    if do_delaunay:
        tet_optim.vertex_optim.step()
        tet_optim.vertex_optim.zero_grad()

    if glo_optim:
        glo_optim.step()
        glo_optim.zero_grad()

    if args.use_bilateral_grid:
        bil_optimizer.step()
        bil_optimizer.zero_grad(set_to_none=True)
        bil_scheduler.step()

    tet_optim.update_learning_rate(iteration)

    with torch.no_grad():
        if iteration % 10 == 0 and args.record_training:
            render_pkg = render(sample_camera, model, min_t=min_t, tile_size=args.tile_size)
            sample_image = render_pkg['render']
            sample_image = sample_image.permute(1, 2, 0)
            sample_image = (sample_image.detach().cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
            video_writer.write(pad_image2even(sample_image))

    if do_cloning and not model.frozen:
        with torch.no_grad():
            # sampled_cams = [train_cameras[i] for i in densification_sampler.nextids()]
            sampled_cams = [train_cameras[i] for i in np.unique(densification_cam_buffer)]

            model.eval()
            stats = collect_render_stats(sampled_cams, model, glo_list, args, device)
            model.train()
            render_pkg = render(sample_camera, model, min_t=min_t, tile_size=args.tile_size)
            sample_image = render_pkg['render']
            sample_image = sample_image.permute(1, 2, 0)
            sample_image = (sample_image.detach().cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)

            apply_densification(
                stats,
                model       = model,
                tet_optim   = tet_optim,
                args        = args,
                iteration   = iteration,
                device      = device,
                sample_cam  = sample_camera,
                sample_image= sample_image,     # whatever RGB debug frame you use
                budget      = max(args.budget - model.vertices.shape[0], 0)
            )
            gc.collect()
            torch.cuda.empty_cache()
            densification_cam_buffer = []

    # Save checkpoints at specified iterations
    if iteration in args.checkpoint_iterations:
        model.save2ply(args.output_path / f"ckpt_{iteration}.ply")
        print(f"Saved checkpoint at iteration {iteration}")

    if do_sh_up:
        model.sh_up()

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
        "Density": repr(f"{density_loss.item():.3f}")
    })

avged_psnrs = [sum(v)/len(v) for v in psnrs if len(v) == len(train_cameras)]
if args.record_training:
    video_writer.release()

model.save2ply(args.output_path / "ckpt.ply")

torch.cuda.synchronize()
torch.cuda.empty_cache()

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

with torch.no_grad():
    if args.glo_dim > 0:
        mean_glo = glo_list.weight.data.mean(dim=0)
    else:
        mean_glo = None
    epath = cam_util.generate_cam_path(train_cameras, 400)
    eimages = []
    for camera in tqdm(epath):
        render_pkg = render(camera, model, glo=mean_glo, min_t=min_t, tile_size=args.tile_size)
        image = render_pkg['render']
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        eimages.append(pad_image2even(image))
mediapy.write_video(args.output_path / "rotating.mp4", eimages)

sd = model.state_dict()
sd['indices'] = model.indices
torch.save(sd, args.output_path / "ckpt.pth")
if args.glo_dim > 0:
    torch.save(glo_list.state_dict(), args.output_path / "glo.pth")
