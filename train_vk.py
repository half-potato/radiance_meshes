"""Training loop using rmeshvk (wgpu) renderer instead of Slang.

Key differences from train.py:
  - Main render call uses render_vk (wgpu-based)
  - No distortion loss (rmeshvk doesn't provide it)
  - No ray jitter support
  - Densification still uses Slang render() for per-tet stats
  - Evaluation still uses Slang render() for cross-check
  - Renderer is recreated after topology changes
"""

import cv2
import os
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
# NOTE: Do NOT import from utils.train_util at top level -- it triggers Slang
# shader compilation which can hang if the cache is stale. Slang-dependent
# imports are deferred to the code blocks that actually need them.
from models.ingp_color import Model, TetOptimizer
from models.frozen import freeze_model
from fused_ssim import fused_ssim
from pathlib import Path, PosixPath
from utils.args import Args
import json
import imageio
import termplotlib as tpl
import gc
from utils.decimation import apply_decimation
import mediapy
from utils.graphics_utils import calculate_norm_loss, depth_to_normals
from icecream import ic

from utils.render_vk import render_vk, create_vk_renderer


# Inline simple utilities to avoid importing utils.train_util at top level
def pad_hw2even(h, w):
    return int(math.ceil(h / 2)) * 2, int(math.ceil(w / 2)) * 2

def pad_image2even(im, fnp=np):
    h, w = im.shape[:2]
    nh, nw = pad_hw2even(h, w)
    im_full = fnp.zeros((nh, nw, 3), dtype=im.dtype)
    im_full[:h, :w] = im
    return im_full

class SimpleSampler:
    def __init__(self, total_num_samples, batch_size, device):
        self.total_num_samples = total_num_samples
        self.batch_size = batch_size
        self.curr = total_num_samples
        self.ids = None
        self.device = device

    def nextids(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        self.curr += batch_size
        if self.curr + batch_size > self.total_num_samples:
            self.ids = torch.randperm(self.total_num_samples, dtype=torch.long, device=self.device)
            self.curr = 0
        ids = self.ids[self.curr : self.curr + batch_size]
        return ids

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
args.max_norm = 1

# iNGP Settings
args.use_tcnn = False
args.encoding_lr = 3e-3
args.final_encoding_lr = 3e-4
args.network_lr = 3e-3
args.final_network_lr = 3e-4
args.hidden_dim = 64
args.scale_multi = 0.35 # chosen such that 96% of the distribution is within the sphere
args.log2_hashmap_size = 23
args.per_level_scale = 2
args.L = 8
args.hashmap_dim = 8
args.base_resolution = 64
args.density_offset = -3
args.lambda_weight_decay = 1
args.percent_alpha = 0.0 # preconditioning
args.spike_duration = 500
args.additional_attr = 0

args.g_init=1e-5
args.s_init=1e-1
args.d_init=0.1
args.c_init=1e-1

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
args.lambda_opacity = 0.0

# Clone Settings
args.num_samples = 200
args.k_samples = 1
args.trunc_sigma = 0.35
args.min_tet_count = 9
args.densify_start = 2000
args.densify_end = 16000
args.densify_interval = 500
args.budget = 2_000_000
# args.within_thresh = 0.5
# args.total_thresh = 2.0
# args.clone_min_contrib = 0.003
# args.split_min_contrib = 0.01

args.within_thresh = 0.3 / 2.7
args.total_thresh = 2.0
args.clone_min_contrib = 5/255
args.split_min_contrib = 10/255

args.lambda_ssim = 0.2
args.lambda_ssim_bw = 0.0
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

# Decimation Settings
args.decimate_start = 4000
args.decimate_end = 17000
args.decimate_interval = 2000
args.decimate_count = 5000
args.decimate_threshold = 0.0

# Edge Length Regularization
args.lambda_edge_length = 0.0

# MCMC-style vertex noise (SGLD)
args.noise_lr = 0.0


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
dschedule_decimate = list(range(args.decimate_start, args.decimate_end, args.decimate_interval))

densification_sampler = SimpleSampler(len(train_cameras), args.num_samples, device)

# --- rmeshvk renderer state ---
vk_renderer = None
vk_sh_deg = None
vk_width = None
vk_height = None


def maybe_recreate_renderer(model, camera, vk_renderer, vk_sh_deg, vk_width, vk_height):
    """Recreate RMeshRenderer if None or if resolution/SH degree changed."""
    w, h = camera.image_width, camera.image_height
    deg = model.current_sh_deg
    if vk_renderer is None or deg != vk_sh_deg or w != vk_width or h != vk_height:
        vk_renderer = create_vk_renderer(model, camera, w, h)
        vk_sh_deg = deg
        vk_width = w
        vk_height = h
    return vk_renderer, vk_sh_deg, vk_width, vk_height


progress_bar = tqdm(range(args.iterations))
torch.cuda.empty_cache()
for iteration in progress_bar:
    do_delaunay = iteration % args.delaunay_interval == 0 and iteration < args.freeze_start
    do_freeze = iteration == args.freeze_start
    do_cloning = iteration in dschedule
    do_sh_up = not args.sh_interval == 0 and iteration % args.sh_interval == 0 and iteration > 0
    do_sh_step = iteration % args.sh_step == 0
    do_decimation = iteration in dschedule_decimate and not model.frozen

    if do_delaunay or do_freeze:
        st = time.time()
        tet_optim.update_triangulation(
            density_threshold=args.density_threshold if iteration > args.threshold_start else 0,
            alpha_threshold=args.alpha_threshold if iteration > args.threshold_start else 0, high_precision=do_freeze)
        vk_renderer = None  # topology changed
        if do_freeze:
            del tet_optim
            n_tets = model.indices.shape[0]
            mask = torch.ones((n_tets), device=device, dtype=bool)
            print(f"Kept {mask.sum()} tets")
            model, tet_optim = freeze_model(model, mask, args)
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

    # Ensure renderer is up-to-date
    vk_renderer, vk_sh_deg, vk_width, vk_height = maybe_recreate_renderer(
        model, camera, vk_renderer, vk_sh_deg, vk_width, vk_height)

    render_pkg = render_vk(camera, model, vk_renderer)
    image = render_pkg['render']

    l1_loss = ((target - image).abs() * gt_mask).mean()
    l2_loss = ((target - image)**2 * gt_mask).mean()
    reg = tet_optim.regularizer(render_pkg, **args.as_dict())
    ssim_loss = (1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))).clip(min=0, max=1)
    # No distortion loss from rmeshvk
    loss = (1-args.lambda_ssim)*l1_loss + \
           args.lambda_ssim*ssim_loss + \
           reg + \
           args.lambda_opacity * (1-render_pkg['alpha']).mean()
    if args.lambda_ssim_bw > 0:
        bw_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device).view(3, 1, 1)
        image_bw = (image * bw_weights).sum(dim=0, keepdim=True)
        target_bw = (target * bw_weights).sum(dim=0, keepdim=True)
        ssim_bw_loss = (1-fused_ssim(image_bw.unsqueeze(0), target_bw.unsqueeze(0))).clip(min=0, max=1)
        loss = loss + args.lambda_ssim_bw * ssim_bw_loss

    if args.lambda_edge_length > 0 and not model.frozen:
        idx = model.indices.long()
        v = model.vertices
        edge_len_sq = (
            (v[idx[:, 0]] - v[idx[:, 1]]).pow(2).sum(-1) +
            (v[idx[:, 0]] - v[idx[:, 2]]).pow(2).sum(-1) +
            (v[idx[:, 0]] - v[idx[:, 3]]).pow(2).sum(-1) +
            (v[idx[:, 1]] - v[idx[:, 2]]).pow(2).sum(-1) +
            (v[idx[:, 1]] - v[idx[:, 3]]).pow(2).sum(-1) +
            (v[idx[:, 2]] - v[idx[:, 3]]).pow(2).sum(-1)
        ) / 6
        loss = loss + args.lambda_edge_length * edge_len_sq.mean()

    loss.backward()

    # --- Diagnostics: save images and check gradients ---
    if iteration % 50 == 0:
        with torch.no_grad():
            img_np = image.detach().permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            imageio.imwrite(args.output_path / f"vk_render_{iteration:05d}.png", img_np)
            gt_np = target.detach().permute(1, 2, 0).cpu().numpy()
            gt_np = (gt_np * 255).clip(0, 255).astype(np.uint8)
            imageio.imwrite(args.output_path / f"gt_{iteration:05d}.png", gt_np)

            # Print gradient stats for the backbone
            for name, p in model.backbone.named_parameters():
                if p.grad is not None:
                    g = p.grad
                    print(f"  grad {name}: mean={g.mean():.6f} std={g.std():.6f} max={g.abs().max():.6f}")
                    break  # just print the first one for brevity

    if iteration == 0:
        # Also render with Slang for comparison (deferred import)
        with torch.no_grad():
            try:
                from utils.train_util import render
                render_pkg_slang = render(camera, model, min_t=min_t, tile_size=args.tile_size)
                slang_img = render_pkg_slang['render'].detach().permute(1, 2, 0).cpu().numpy()
                slang_img = (slang_img * 255).clip(0, 255).astype(np.uint8)
                imageio.imwrite(args.output_path / f"slang_render_000.png", slang_img)
                print(f"Slang vs VK L1 diff: {(render_pkg_slang['render'] - image).abs().mean():.6f}")
            except Exception as e:
                print(f"Slang comparison render failed: {e}")

    tet_optim.main_step()
    tet_optim.main_zero_grad()

    if do_sh_step and tet_optim.sh_optim is not None:
        tet_optim.sh_optim.step()
        tet_optim.sh_optim.zero_grad()

    if do_delaunay:
        tet_optim.vertex_optim.step()
        tet_optim.vertex_optim.zero_grad()

        # MCMC-style SGLD noise on vertex positions
        if args.noise_lr > 0 and not model.frozen:
            with torch.no_grad():
                vlr = tet_optim.vertices_lr
                noise = torch.randn_like(model.interior_vertices) * args.noise_lr * vlr
                model.interior_vertices.data.add_(noise)

    tet_optim.update_learning_rate(iteration)

    if do_sh_up:
        model.sh_up()
        vk_renderer = None  # SH degree changed

    if do_cloning and not model.frozen:
        with torch.no_grad():
            from utils.train_util import render
            from utils.densification import collect_render_stats, apply_densification
            sampled_cams = [train_cameras[i] for i in densification_sampler.nextids()]

            # Use Slang renderer for densification stats (needs per-tet info)
            render_pkg_slang = render(sample_camera, model, min_t=min_t, tile_size=args.tile_size)
            sample_image = render_pkg_slang['render']
            sample_image = sample_image.permute(1, 2, 0)
            sample_image = (sample_image.detach().cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)

            if render_pkg_slang['xyzd'] is not None and render_pkg_slang['xyzd'].abs().sum() > 0:
                pred_normal = depth_to_normals(render_pkg_slang['xyzd'][..., 3:], camera.fx, camera.fy)
                vis_pred_normal = (pred_normal * 127 + 128).clamp(0, 255).byte().cpu().numpy()
                imageio.imwrite(args.output_path / f"pred_normal{iteration}.png",
                                vis_pred_normal)

            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)

            gc.collect()
            torch.cuda.empty_cache()
            model.eval()
            stats = collect_render_stats(sampled_cams, model, args, device)
            model.train()
            target_addition = args.budget - model.vertices.shape[0]

            apply_densification(
                stats,
                model       = model,
                tet_optim   = tet_optim,
                args        = args,
                iteration   = iteration,
                device      = device,
                sample_cam  = sample_camera,
                sample_image= sample_image,
                target_addition= target_addition
            )
            del stats
            vk_renderer = None  # topology changed
            gc.collect()
            torch.cuda.empty_cache()

    if do_decimation:
        with torch.no_grad():
            n_removed = apply_decimation(model, tet_optim, args, device)
            vk_renderer = None  # topology changed
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
    })

avged_psnrs = [sum(v)/len(v) for v in psnrs if len(v) == len(train_cameras)]

torch.cuda.synchronize()
torch.cuda.empty_cache()

model.save2ply(args.output_path / "ckpt.ply")
sd = model.state_dict()
sd['indices'] = model.indices
sd['empty_indices'] = model.empty_indices
torch.save(sd, args.output_path / "ckpt.pth")

# Evaluation uses Slang renderer for cross-check
from utils import test_util
if args.render_train:
    splits = zip(['train', 'test'], [train_cameras, test_cameras])
else:
    splits = zip(['test'], [test_cameras])
results = test_util.evaluate_and_save(model, splits, args.output_path, args.tile_size, min_t)

with (args.output_path / "results.json").open("w") as f:
    all_data = dict(
        psnr = avged_psnrs[-1] if len(avged_psnrs) > 0 else 0,
        n_vertices = model.vertices.shape[0],
        n_interior_vertices = model.num_int_verts,
        n_tets = model.indices.shape[0],
        **results
    )
    json.dump(all_data, f, cls=CustomEncoder)

# Rotating video uses Slang renderer
with torch.no_grad():
    from utils.train_util import render as slang_render
    epath = cam_util.generate_cam_path(train_cameras, 400)
    eimages = []
    for camera in tqdm(epath):
        render_pkg = slang_render(camera, model, min_t=min_t, tile_size=args.tile_size)
        image = render_pkg['render']
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        eimages.append(pad_image2even(image))
mediapy.write_video(args.output_path / "rotating.mp4", eimages)
