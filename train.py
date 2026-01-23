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
from utils.train_util import render, pad_image2even, pad_hw2even, SimpleSampler
# from models.vertex_color import Model, TetOptimizer
from models.ingp_color import Model, TetOptimizer
# from models.frozen import FrozenTetModel as Model
# from models.frozen import FrozenTetOptimizer as TetOptimizer
from models.frozen import freeze_model
from fused_ssim import fused_ssim
from pathlib import Path, PosixPath
from utils.args import Args
import json
import imageio
from utils import test_util
import termplotlib as tpl
import gc
from utils.densification import collect_render_stats, apply_densification
import mediapy
from utils.graphics_utils import calculate_norm_loss, depth_to_normals
from icecream import ic

torch.set_num_threads(1)

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PosixPath):
            return str(o)
        return super().default(o)

eps = torch.finfo(torch.float).eps
args = Args()
args.tile_size = 4
args.image_folder = "images_2"
args.eval = False
args.dataset_path = Path("/optane/nerf_datasets/360/bonsai")
args.output_path = Path("output/test/")
args.iterations = 30000
args.ckpt = ""
args.resolution = 1
args.render_train = False

# Light Settings
args.max_sh_deg = 3
args.sh_interval = 0
args.sh_step = 1

# iNGP Settings
args.use_tccn = False
args.encoding_lr = 3e-3
args.final_encoding_lr = 3e-4
args.network_lr = 1e-3
args.final_network_lr = 1e-4
args.hidden_dim = 64
args.scale_multi = 0.35 # chosen such that 96% of the distribution is within the sphere 
args.log2_hashmap_size = 23
args.per_level_scale = 2
args.L = 8
args.hashmap_dim = 8
args.base_resolution = 64
args.density_offset = -4
args.lambda_weight_decay = 1
args.percent_alpha = 0.0 # preconditioning
args.spike_duration = 500
args.additional_attr = 0

args.g_init=1.0
args.s_init=1e-4
args.d_init=0.1
args.c_init=0.8

# Vertex Settings
args.lr_delay = 0
args.vert_lr_delay = 0
args.vertices_lr = 1e-4
args.final_vertices_lr = 1e-6
args.vertices_lr_delay_multi = 1e-8
args.delaunay_interval = 10

args.freeze_start = 18000
args.freeze_lr = 1e-3
args.final_freeze_lr = 1e-4

# Distortion Settings
args.lambda_dist = 0.0
args.lambda_norm = 0.0
args.lambda_sh = 0.0

# Clone Settings
args.num_samples = 200
args.k_samples = 1
args.trunc_sigma = 0.35
args.min_tet_count = 9
args.densify_start = 2000
args.densify_end = 16000
args.densify_interval = 500
args.budget = 2_000_000
args.within_thresh = 0.5
args.total_thresh = 2.0
args.clone_min_contrib = 0.003
args.split_min_contrib = 0.01

args.lambda_ssim = 0.2
args.min_t = 0.4
args.sample_cam = 8
args.data_device = 'cpu'
args.density_threshold = 0.1
args.alpha_threshold = 0.1
args.contrib_threshold = 0.0
args.threshold_start = 4500
args.voxel_size = 0.01

args.ablate_gradient = False
args.ablate_circumsphere = True
args.ablate_downweighing = False


args = Args.from_namespace(args.get_parser().parse_args())

args.output_path.mkdir(exist_ok=True, parents=True)

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device=args.data_device, eval=args.eval, resolution=args.resolution)

np.savetxt(str(args.output_path / "transform.txt"), scene_info.transform)

args.num_samples = min(len(train_cameras), args.num_samples)

with (args.output_path / "config.json").open("w") as f:
    json.dump(args.as_dict(), f, cls=CustomEncoder)

device = torch.device('cuda')
if len(args.ckpt) > 0: 
    model = Model.load_ckpt(Path(args.ckpt), device)
else:
    model = Model.init_from_pcd(scene_info.point_cloud, train_cameras, device,
                                current_sh_deg = args.max_sh_deg if args.sh_interval <= 0 else 0,
                                **args.as_dict())
min_t = args.min_t

tet_optim = TetOptimizer(model, **args.as_dict())
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

densification_sampler = SimpleSampler(len(train_cameras), args.num_samples, device)

video_writer = cv2.VideoWriter(str(args.output_path / "training.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30,
                               pad_hw2even(sample_camera.image_width, sample_camera.image_height))

progress_bar = tqdm(range(args.iterations))
torch.cuda.empty_cache()
for iteration in progress_bar:
    do_delaunay = iteration % args.delaunay_interval == 0 and iteration < args.freeze_start
    do_freeze = iteration == args.freeze_start
    do_cloning = iteration in dschedule
    do_sh_up = not args.sh_interval == 0 and iteration % args.sh_interval == 0 and iteration > 0
    do_sh_step = iteration % args.sh_step == 0

    if do_delaunay or do_freeze:
        st = time.time()
        tet_optim.update_triangulation(
            density_threshold=args.density_threshold if iteration > args.threshold_start else 0,
            alpha_threshold=args.alpha_threshold if iteration > args.threshold_start else 0, high_precision=do_freeze)
        if do_freeze:
            del tet_optim
            # model.eval()
            # mask = determine_cull_mask(train_cameras, model, args, device)
            n_tets = model.indices.shape[0]
            mask = torch.ones((n_tets), device=device, dtype=bool)
            # model.train()
            print(f"Kept {mask.sum()} tets")
            model, tet_optim = freeze_model(model, mask, args)
            # model, tet_optim = freeze_model(model, **args.as_dict())
            gc.collect()
            torch.cuda.empty_cache()

    if len(inds) == 0:
        inds = list(range(len(train_cameras)))
        random.shuffle(inds)
        psnrs.append([])
    ind = inds.pop()
    camera = train_cameras[ind]
    target = camera.original_image.cuda()
    gt_mask = camera.gt_alpha_mask.cuda()

    st = time.time()
    ray_jitter = torch.rand((camera.image_height, camera.image_width, 2), device=device)
    render_pkg = render(camera, model, scene_scaling=model.scene_scaling, ray_jitter=ray_jitter, **args.as_dict())
    image = render_pkg['render']

    l1_loss = ((target - image).abs() * gt_mask).mean()
    l2_loss = ((target - image)**2 * gt_mask).mean()
    reg = tet_optim.regularizer(render_pkg, **args.as_dict())
    ssim_loss = (1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))).clip(min=0, max=1)
    dl_loss = render_pkg['distortion_loss']
    norm_loss = calculate_norm_loss(render_pkg['xyzd'], camera.fx, camera.fy)
    loss = (1-args.lambda_ssim)*l1_loss + \
           args.lambda_ssim*ssim_loss + \
           reg + \
           args.lambda_dist * dl_loss + \
           args.lambda_sh * render_pkg['sh_reg']
        #    args.lambda_norm * norm_loss + \

    loss.backward()

    tet_optim.main_step()
    tet_optim.main_zero_grad()

    if do_sh_step and tet_optim.sh_optim is not None:
        tet_optim.sh_optim.step()
        tet_optim.sh_optim.zero_grad()

    if do_delaunay:
        tet_optim.vertex_optim.step()
        tet_optim.vertex_optim.zero_grad()

    tet_optim.update_learning_rate(iteration)

    if do_sh_up:
        model.sh_up()

    # if iteration % 10 == 0:
    #     with torch.no_grad():
    #         render_pkg = render(sample_camera, model, min_t=min_t, tile_size=args.tile_size)
    #         sample_image = render_pkg['render']
    #         sample_image = sample_image.permute(1, 2, 0)
    #         sample_image = (sample_image.detach().cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
    #         sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
    #         video_writer.write(pad_image2even(sample_image))

    if do_cloning and not model.frozen:
        with torch.no_grad():
            sampled_cams = [train_cameras[i] for i in densification_sampler.nextids()]

            render_pkg = render(sample_camera, model, min_t=min_t, tile_size=args.tile_size)
            sample_image = render_pkg['render']
            sample_image = sample_image.permute(1, 2, 0)
            sample_image = (sample_image.detach().cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)

            pred_normal = depth_to_normals(render_pkg['xyzd'][..., 3:], camera.fx, camera.fy)
            # vis_depth, _ = visualize_depth_numpy(render_pkg['xyzd'][..., 3:].cpu().numpy())
            vis_normal = (render_pkg['xyzd'][..., :3] * 127 + 128).clamp(0, 255).byte().cpu().numpy()
            vis_pred_normal = (pred_normal * 127 + 128).clamp(0, 255).byte().cpu().numpy()
            imageio.imwrite(args.output_path / f"normal{iteration}.png",
                            vis_normal)
            imageio.imwrite(args.output_path / f"pred_normal{iteration}.png",
                            vis_pred_normal)
            # imageio.imwrite(args.output_path / f"depth{iteration}.png",
            #                 vis_depth)
            # video_writer.write(pad_image2even(sample_image))

            gc.collect()
            torch.cuda.empty_cache()
            model.eval()
            stats = collect_render_stats(sampled_cams, model, args, device)
            model.train()
            # target_addition = targets[dschedule.index(iteration)] - model.vertices.shape[0]
            target_addition = args.budget - model.vertices.shape[0]

            apply_densification(
                stats,
                model       = model,
                tet_optim   = tet_optim,
                args        = args,
                iteration   = iteration,
                device      = device,
                sample_cam  = sample_camera,
                sample_image= sample_image,     # whatever RGB debug frame you use
                target_addition= target_addition

            )
            # tet_optim.prune(**args.as_dict())
            del stats
            gc.collect()
            torch.cuda.empty_cache()

    psnr = -20 * math.log10(math.sqrt(l2_loss.detach().cpu().clip(min=1e-6).item()))
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

avged_psnrs = [sum(v)/len(v) for v in psnrs if len(v) == len(train_cameras)]
video_writer.release()

torch.cuda.synchronize()
torch.cuda.empty_cache()

model.save2ply(args.output_path / "ckpt.ply")
sd = model.state_dict()
sd['indices'] = model.indices
sd['empty_indices'] = model.empty_indices
torch.save(sd, args.output_path / "ckpt.pth")

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
    epath = cam_util.generate_cam_path(train_cameras, 400)
    eimages = []
    for camera in tqdm(epath):
        render_pkg = render(camera, model, min_t=min_t, tile_size=args.tile_size)
        image = render_pkg['render']
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        eimages.append(pad_image2even(image))
mediapy.write_video(args.output_path / "rotating.mp4", eimages)