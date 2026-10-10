import cv2
import os
# Skip slangtorch's in-lock ninja recheck on warm caches (same wrapper as
# full_inverse.py). The recheck forks a compiler from a threaded process and
# can deadlock while holding the compile lock; loadModule's slang-source
# mtime check still catches real .slang edits.
import slangtorch
if not getattr(slangtorch.loadModule, "_skipninja_wrapped", False):
    _orig_loadModule = slangtorch.loadModule
    def _loadModule_skipninja(*a, **kw):
        kw.setdefault("skipNinjaCheck", True)
        return _orig_loadModule(*a, **kw)
    _loadModule_skipninja._skipninja_wrapped = True
    slangtorch.loadModule = _loadModule_skipninja
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
from models.ingp_color import Model, TetOptimizer
from models.frozen import freeze_model
from fused_ssim import fused_ssim
from pathlib import Path, PosixPath
from utils.args import Args
import json
import imageio
from utils import test_util
import termplotlib as tpl
import gc
from utils.densification import collect_render_stats, apply_densification, cull_occluded_tets
from utils.densify_criteria import (
    build_criteria, compare_criteria, apply_scored_densification)
from utils.decimation import apply_decimation
import mediapy
from utils.graphics_utils import depth_to_normal
from icecream import ic

# from tracers.splinetracers.tetra_splinetracer import render_rt
# render = render_rt

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
# Bitwise-equal Triton hash grid (utils/triton_hashgrid.py): ~5x encoding
# forward, checkpoint-friendly backward. Opt-in until burned in.
args.use_triton_hashgrid = False
# Fuse the 4 MLP heads into one Triton kernel (tf32, matches eager-tf32 within
# noise; ~5x head forward, ~1.18x compiled+checkpoint producer). Opt-in.
args.use_triton_heads = False
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

args.kernel_sigma = 1.5
args.project_cc = False
args.values_lr = 5e-3
args.final_values_lr = 5e-4

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

# Stochastic size-proportional tet sampling (unbiased LOD). When on, sub-pixel
# tets are Russian-roulette culled with prob 1-min(s,1) and survivors' density
# reweighted by 1/min(s,1); >=1px tets are bit-exact. Off = baseline.
args.stochastic_subpixel = False

# Distortion Settings
args.lambda_dist = 0.0
# Power-ladder ramp for the distortion loss: when > 0, the effective weight
# climbs exponentially over 3 decades from lambda_dist/1000 to lambda_dist,
# reaching full strength at this iteration (constant afterwards). Constant
# full-strength distortion from iter 0 starves coarse geometry growth.
args.lambda_dist_ramp = 0
args.lambda_norm = 0.0
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
args.within_thresh = 0.5
args.total_thresh = 2.0
args.clone_min_contrib = 0.003
args.split_min_contrib = 0.01
# Densification criterion (utils/densify_criteria.py):
#   error_votes  -- legacy render-error stats (threshold + random cap)
#   dwr          -- accumulated |dL/d(cell_values)| over the densify window
#   cell_jump    -- inter-tet cell-value jumps (view-independent)
#   dwr_jump     -- rank-product of the two
# densify_shadow: csv of extra criteria scored + logged each round for
# side-by-side comparison (jaccard/spearman vs the active one); no effect on
# which tets are split.
args.densify_criterion = "error_votes"
# Scale-free densification thresholds: 'median' replaces the absolute
# within_thresh/total_thresh with within_rel/total_rel x median(nonzero
# score), so one setting transfers between synthetic and photo error scales.
args.densify_rel_mode = "none"
args.within_rel = 8.0
args.total_rel = 8.0
args.densify_shadow = ""
# Cap insertions per round (0 = legacy unlimited burst-to-budget). Scored
# criteria are only meaningful with rationing -- an unrationed first round
# spends the whole budget against a barely-trained model.
args.densify_max_per_round = 0
# Save a rolling ckpt.pth every N iters (0 = end-of-run only).
args.ckpt_interval = 5000
# Optional dense init: path to an .npz with points/colors/normals in the
# loader's normalized frame (e.g. a MoGe fused cloud built by full_inverse's
# init path). Replaces the COLMAP point cloud for Delaunay initialization.
# The normalization transform is deterministic per dataset, so clouds are
# portable between pipelines (verified identical transform.txt).
args.init_pcd = ""
# Initialization-free training: ignore the COLMAP/random point cloud and
# start from N uniform random points instead. Points are sampled in the
# normalized (post-PCA) frame inside a ball sized from the camera positions
# (radius = random_init_radius x max camera distance from origin). 0 = off.
args.random_init = 0
args.random_init_radius = 1.5
# Fraction of random-init points placed in a heavy-tailed background shell
# beyond the camera ball (radius up to shell_extent x r_max, density ~1/r^2,
# like a real COLMAP cloud's far field). 0 = all points in the near ball;
# stump's COLMAP cloud has 36% of points beyond 1.5x camera radius.
args.random_init_shell_frac = 0.0
args.random_init_shell_extent = 20.0
# Section profiling (utils/section_prof.py): collect CUDA-event timings for
# iterations [profile_start, profile_start + profile_iters), print the table,
# and exit. 0 disables. profile_start leaves room for compile/warmup.
args.profile_iters = 0
args.profile_start = 60
# Skip autograd through the geometric vertex-normal density pass (full-table
# iNGP fwd+bwd per iteration, ~60% of the step). Safe when lambda_norm and
# lambda_dist are 0 (no loss consumes normals); learned normal offsets keep
# training either way.
args.detach_geo_normals = False

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

# Occlusion-based tet culling (RadSplat peak_contrib = max over rays of
# alpha*transmittance-in-front). Removes interior vertices whose incident tets
# are buried behind opaque geometry in every sampled view. cull_interval=0 ->
# OFF (legacy path byte-identical). See densification.cull_occluded_tets.
args.cull_interval = 0        # 0 = OFF. >0 runs a cull pass every this many iters.
args.cull_start = 8000        # after geometry settles (> threshold_start=4500)
args.cull_end = 16000         # stop before freeze so budget can still refill
args.cull_thresh = 0.01       # cull verts whose max(alpha*T) never exceeds this
args.cull_min_obs = 2         # grace: require >= this many observed passes first

# Edge Length Regularization
args.lambda_edge_length = 0.0

# MCMC-style vertex noise (SGLD)
args.noise_lr = 0.0


# White-background blender protocol (composite GT + render onto white via
# DSPLAT_WHITE_BG; default keeps legacy black/alpha-masked behavior).
args.white_bg = False
# Deprecated no-op, kept so existing launch commands still parse: the frozen
# vertex LR schedule continuation is now always on (see
# FrozenTetOptimizer.update_learning_rate).
args.frozen_vlr_sched = False

args = Args.from_namespace(args.get_parser().parse_args())
if args.white_bg:
    os.environ["DSPLAT_WHITE_BG"] = "1"

args.output_path.mkdir(exist_ok=True, parents=True)

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device=args.data_device, eval=args.eval, resolution=args.resolution)

np.savetxt(str(args.output_path / "transform.txt"), scene_info.transform)

args.num_samples = min(len(train_cameras), args.num_samples)

with (args.output_path / "config.json").open("w") as f:
    json.dump(args.as_dict(), f, cls=CustomEncoder)

device = torch.device('cuda')
init_pcd = scene_info.point_cloud
if args.init_pcd:
    from data.types import BasicPointCloud
    _npz = np.load(args.init_pcd)
    init_pcd = BasicPointCloud(points=_npz['points'],
                               colors=_npz['colors'],
                               normals=_npz['normals'])
    print(f"Init pcd override: {init_pcd.points.shape[0]} points "
          f"from {args.init_pcd} (COLMAP had "
          f"{scene_info.point_cloud.points.shape[0]})")
if args.random_init > 0:
    from data.types import BasicPointCloud
    cam_centers = np.stack([
        np.linalg.inv(cam.world_view_transform.T.cpu().numpy())[:3, 3]
        for cam in train_cameras])
    r_max = float(np.linalg.norm(cam_centers, axis=1).max()) * args.random_init_radius
    n = args.random_init
    n_shell = int(n * args.random_init_shell_frac)
    pts = np.random.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    radii = r_max * np.random.rand(n, 1) ** (1 / 3)  # uniform in ball
    if n_shell > 0:
        # 1/r^2 radial density: r = r_max / u, u ~ U(1/extent, 1]
        u = np.random.uniform(1.0 / args.random_init_shell_extent, 1.0, (n_shell, 1))
        radii[:n_shell] = r_max / u
    pts *= radii
    init_pcd = BasicPointCloud(points=pts,
                               colors=np.random.rand(n, 3) * 0.01,
                               normals=np.zeros((n, 3)))
    print(f"Random init: {n} points in ball r={r_max:.2f} "
          f"(ignoring {scene_info.point_cloud.points.shape[0]}-point COLMAP cloud)")

if len(args.ckpt) > 0:
    model = Model.load_ckpt(Path(args.ckpt), device)
else:
    model = Model.init_from_pcd(init_pcd, train_cameras, device,
                                current_sh_deg = args.max_sh_deg if args.sh_interval <= 0 else 0,
                                **args.as_dict())
min_t = args.min_t
model.detach_geo_normals = args.detach_geo_normals

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
dschedule_decimate = list(range(args.decimate_start, args.decimate_end, args.decimate_interval))
dschedule_cull = (list(range(args.cull_start, args.cull_end, args.cull_interval))
                  if args.cull_interval > 0 else [])

# print("Encoding LR")
# xs = list(range(args.iterations))
# ys = [tet_optim.encoder_scheduler_args(x) for x in xs]
# fig = tpl.figure()
# fig.plot(xs, ys, width=150, height=20)
# fig.show()

densification_sampler = SimpleSampler(len(train_cameras), args.num_samples, device)

# Densify criteria: active + shadow-compare set. DWR observes gradients via a
# tensor hook per iteration; the others are stateless until decision time.
_crit_spec = ",".join(dict.fromkeys(
    [args.densify_criterion] +
    [s.strip() for s in args.densify_shadow.split(",") if s.strip()]))
densify_criteria = build_criteria(_crit_spec, model)

# video_writer = imageio.get_writer(str(args.output_path / "training.mp4"), fps=10)

from utils import section_prof
from utils.section_prof import section
from utils.model_query import prepare_query

progress_bar = tqdm(range(args.iterations))
torch.cuda.empty_cache()
for iteration in progress_bar:
    if args.profile_iters > 0:
        if iteration == args.profile_start:
            section_prof.enable()
        elif iteration == args.profile_start + args.profile_iters:
            section_prof.print_report(
                header=f"iters {args.profile_start}-{iteration}, "
                       f"#V={model.vertices.shape[0]}, "
                       f"#T={model.indices.shape[0]}")
            import sys
            sys.exit(0)
    do_delaunay = iteration % args.delaunay_interval == 0 and iteration < args.freeze_start
    do_freeze = iteration == args.freeze_start
    do_cloning = iteration in dschedule
    do_sh_up = not args.sh_interval == 0 and iteration % args.sh_interval == 0 and iteration > 0
    do_sh_step = iteration % args.sh_step == 0
    do_decimation = iteration in dschedule_decimate and not model.frozen
    # Guard against the freeze collision: do_* is computed before the freeze
    # block runs this same iteration, so a schedule hit on freeze_start would
    # otherwise run mesh surgery on the frozen model (frozen.py has no such path).
    do_cull = iteration in dschedule_cull and not model.frozen and iteration != args.freeze_start

    if do_delaunay or do_freeze:
        st = time.time()
        with section('delaunay_update'):
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
    with section('render_fwd_total'):
        # Query-once: the frustum mask + single iNGP pass happen here; the
        # rasterizer, geometric vertex normals, and densify criteria all
        # consume the same tensors (utils/model_query.py).
        query = prepare_query(camera, model, tile_size=args.tile_size,
                              min_t=args.min_t,
                              stochastic_subpixel=args.stochastic_subpixel,
                              stochastic_seed=iteration)
        render_pkg = render(camera, model, query=query,
                            ray_jitter=ray_jitter, **args.as_dict())
    if iteration < args.densify_end:
        with section('criteria_observe'):
            for _crit in densify_criteria.values():
                _crit.observe(model, render_pkg)
    image = render_pkg['render']

    _loss_section = section('loss_terms'); _loss_section.__enter__()
    l1_loss = ((target - image).abs() * gt_mask).mean()
    l2_loss = ((target - image)**2 * gt_mask).mean()
    reg = tet_optim.regularizer(render_pkg, **args.as_dict())
    ssim_loss = (1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))).clip(min=0, max=1)
    dl_loss = render_pkg.get('distortion_loss', 0.0)
    # norm_loss = calculate_norm_loss(render_pkg['xyzd'], camera.fx, camera.fy)
    loss = (1-args.lambda_ssim)*l1_loss + \
           args.lambda_ssim*ssim_loss + \
           reg + \
           args.lambda_opacity * (1-render_pkg['alpha']).mean() + \
           (args.lambda_dist * (10.0 ** (-3.0 * max(0.0, 1.0 - iteration / args.lambda_dist_ramp))
                                if args.lambda_dist_ramp > 0 else args.lambda_dist)) * dl_loss
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
    _loss_section.__exit__(None, None, None)

    if args.profile_iters > 0 and query is not None and query.cell_values.requires_grad:
        # Split the opaque backward at the cell_values grad boundary:
        # autograd reaches it after the loss+rasterizer backward and before
        # the iNGP (hash grid + heads) and normals-scatter backward.
        _ev_s = torch.cuda.Event(enable_timing=True)
        _ev_m = torch.cuda.Event(enable_timing=True)
        _ev_e = torch.cuda.Event(enable_timing=True)

        def _mark_cv_grad(g, _ev=_ev_m):
            _ev.record()
            return None

        query.cell_values.register_hook(_mark_cv_grad)
        with section('backward'):
            _ev_s.record()
            loss.backward()
            _ev_e.record()
        if section_prof._ENABLED:
            section_prof._records.setdefault('backward/loss+raster', []).append((_ev_s, _ev_m))
            section_prof._records.setdefault('backward/ingp+normals', []).append((_ev_m, _ev_e))
    else:
        with section('backward'):
            loss.backward()

    # Gradient diagnostics. Reports per-parameter grad stats and -- importantly --
    # flags params that are in no graph (grad is None) or whose grad is all-zero.
    # This is the usual symptom right after freezing: the model swaps its iNGP
    # backbone for per-tet parameters, so anything still keyed off `backbone`
    # silently stops reporting. Works for both the base Model (backbone params)
    # and the FrozenTetModel (per-tet params live directly on the model).
    if iteration % 50 == 0:
        if hasattr(model, 'backbone'):
            named, tag, first_only = model.backbone.named_parameters(), 'backbone', True
        else:
            named, tag, first_only = model.named_parameters(), 'frozen', False
        reported = 0
        for name, p in named:
            if not p.requires_grad:
                continue
            if p.grad is None:
                print(f"  [grad/{tag}] {name}: NO GRAD (grad is None) -- not connected to loss?")
            elif p.grad.abs().max() == 0:
                print(f"  [grad/{tag}] {name}: ZERO grad -- detached or no contribution")
            else:
                g = p.grad
                print(f"  [grad/{tag}] {name}: mean={g.mean():.3e} std={g.std():.3e} max={g.abs().max():.3e}")
                reported += 1
                if first_only:
                    break

    # tet_optim.clip_grad_norm_(args.max_norm)

    with section('optimizer'):
        tet_optim.main_step()
        tet_optim.main_zero_grad()

    if do_sh_step and tet_optim.sh_optim is not None:
        tet_optim.sh_optim.step()
        tet_optim.sh_optim.zero_grad()

    # Vertex optimization must keep running after freezing. Pre-freeze it steps
    # on the Delaunay cadence (gradients accumulate over the interval); post-freeze
    # there are no Delaunay updates, but interior_vertices / interior_vertex_normals
    # are still trainable nn.Parameters, so they must be stepped (and their grads
    # cleared) every iteration -- otherwise grads pile up unapplied and the frozen
    # geometry never moves. `model.frozen` is unreliable here (FrozenTetModel sets
    # it False), so gate on the iteration count instead.
    do_vertex_step = do_delaunay or iteration >= args.freeze_start
    if do_vertex_step:
        tet_optim.vertex_optim.step()
        tet_optim.vertex_optim.zero_grad()

        # MCMC-style SGLD noise on vertex positions -- live (pre-freeze) phase only.
        if do_delaunay and args.noise_lr > 0 and not model.frozen:
            with torch.no_grad():
                vlr = tet_optim.vertices_lr
                noise = torch.randn_like(model.interior_vertices) * args.noise_lr * vlr
                model.interior_vertices.data.add_(noise)

    tet_optim.update_learning_rate(iteration)

    # Periodic crash-safe checkpoint: long runs (60k iters, 10M+ tets) have died
    # hours in with async CUDA illegal-address errors, losing everything because
    # the only save was at the end. Write-then-rename keeps ckpt.pth valid even
    # if the process dies mid-save.
    if args.ckpt_interval > 0 and iteration > 0 and iteration % args.ckpt_interval == 0:
        with torch.no_grad():
            sd = model.state_dict()
            sd['indices'] = model.indices
            sd['empty_indices'] = model.empty_indices
            tmp_path = args.output_path / "ckpt.pth.tmp"
            torch.save(sd, tmp_path)
            os.replace(tmp_path, args.output_path / "ckpt.pth")

    if do_sh_up:
        model.sh_up()

    # if iteration % 10 == 0:
    #     with torch.no_grad():
    #         render_pkg = render(sample_camera, model, min_t=min_t, tile_size=args.tile_size)
    #         sample_image = render_pkg['render']
    #         sample_image = sample_image.permute(1, 2, 0)
    #         sample_image = (sample_image.detach().cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
    #         video_writer.append_data(pad_image2even(sample_image))

    if do_cloning and not model.frozen:
        with torch.no_grad():
            sampled_cams = [train_cameras[i] for i in densification_sampler.nextids()]

            render_pkg = render(sample_camera, model, min_t=min_t, tile_size=args.tile_size)
            sample_image = render_pkg['render']
            sample_image = sample_image.permute(1, 2, 0)
            sample_image = (sample_image.detach().cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)

            pred_normal = depth_to_normal(render_pkg['xyzd'][..., 3:], camera)
            # vis_depth, _ = visualize_depth_numpy(render_pkg['xyzd'][..., 3:].cpu().numpy())
            vis_normal = (render_pkg['xyzd'][..., :3] * 127 + 128).clamp(0, 255).byte().cpu().numpy()
            vis_pred_normal = (pred_normal * 127 + 128).clamp(0, 255).byte().cpu().numpy()
            imageio.imwrite(args.output_path / f"pred_normal{iteration}.png",
                            vis_pred_normal)
            # imageio.imwrite(args.output_path / f"normal{iteration}.png",
            #                 vis_normal)
            # imageio.imwrite(args.output_path / f"depth{iteration}.png",
            #                 vis_depth)
            # video_writer.append_data(pad_image2even(sample_image))
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)

            gc.collect()
            torch.cuda.empty_cache()
            model.eval()
            # Error-vote stats cost num_samples extra renders -- only pay when
            # the legacy criterion participates (active or shadow).
            stats = None
            if "error_votes" in densify_criteria:
                stats = collect_render_stats(sampled_cams, model, args, device)
                densify_criteria["error_votes"].set_stats(stats)
            # target_addition = targets[dschedule.index(iteration)] - model.vertices.shape[0]
            target_addition = args.budget - model.vertices.shape[0]
            if args.densify_max_per_round > 0:
                target_addition = min(target_addition,
                                      args.densify_max_per_round)

            score_map = {
                name: crit.scores(model, camera=sample_camera)
                for name, crit in densify_criteria.items()
            }
            if len(score_map) > 1:
                cmp = compare_criteria(
                    score_map, k=max(1, min(int(target_addition), 100_000)))
                print(" | ".join(
                    f"{k.split('/')[-1]}={v:.3f}" for k, v in cmp.items()))
            model.train()

            if args.densify_criterion == "error_votes":
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
            else:
                crit = densify_criteria[args.densify_criterion]
                n_added = apply_scored_densification(
                    model, tet_optim, crit,
                    score_map[args.densify_criterion],
                    target_addition, args)
                print(f"#Split[{args.densify_criterion}]: {n_added} "
                      f"| verts {model.vertices.shape[0]}")
            for crit in densify_criteria.values():
                crit.reset()
            # tet_optim.prune(**args.as_dict())
            del stats
            gc.collect()
            torch.cuda.empty_cache()

    if do_decimation:
        with torch.no_grad():
            n_removed = apply_decimation(model, tet_optim, args, device)
            gc.collect()
            torch.cuda.empty_cache()

    if do_cull:
        with torch.no_grad():
            model.eval()
            sampled_cams = [train_cameras[i] for i in densification_sampler.nextids()]
            cull_occluded_tets(sampled_cams, model, tet_optim, args, iteration, device)
            model.train()
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
        # "DL": repr(f"{dl_loss:>5.2f}"),
    })

    if iteration % 500 == 0 and hasattr(model, 'log_kernel_diagnostics'):
        diag = model.log_kernel_diagnostics()
        print(f"\n[iter {iteration}] Shell kernel diagnostics:")
        print(f"  CC outside tet: {diag['cc_outside_pct']:.1f}%")
        print(f"  Own weights: mean={diag['own_w_mean']:.4f}, "
              f"std_within_tet={diag['own_w_std_across_tet']:.4f}")
        print(f"  Flap weights: mean={diag['flap_w_mean']:.4f}, "
              f"own_frac={diag['own_frac_mean']:.3f}")
        print(f"  Entropy: {diag['entropy_mean']:.3f} / {math.log(8):.3f} "
              f"({diag['entropy_ratio']:.1%} of uniform)")
        print(f"  Vtx weight sums: mean={diag['vtx_weight_sum_mean']:.2f}, "
              f"std={diag['vtx_weight_sum_std']:.2f}, "
              f"min={diag['vtx_weight_sum_min']:.2f}, "
              f"max={diag['vtx_weight_sum_max']:.2f}")
        print(f"  Vtx tet count: mean={diag['vtx_tet_count_mean']:.0f}, "
              f"std={diag['vtx_tet_count_std']:.0f}, "
              f"range=[{diag['vtx_tet_count_min']:.0f}, {diag['vtx_tet_count_max']:.0f}]")
        if 'val_grad_mean' in diag:
            print(f"  Val grad: mean={diag['val_grad_mean']:.6f}, "
                  f"std={diag['val_grad_std']:.6f}, "
                  f"max={diag['val_grad_max']:.6f}")
        if 'pos_grad_mean' in diag:
            print(f"  Pos grad: mean={diag['pos_grad_mean']:.6f}, "
                  f"std={diag['pos_grad_std']:.6f}, "
                  f"max={diag['pos_grad_max']:.6f}")

avged_psnrs = [sum(v)/len(v) for v in psnrs if len(v) == len(train_cameras)]
# video_writer.close()

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
        n_vertices = model.vertices.shape[0],
        n_interior_vertices = model.num_int_verts,
        n_tets = model.indices.shape[0],
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
