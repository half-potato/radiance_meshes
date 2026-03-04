"""Quick profiling wrapper — runs 200 iterations and reports per-section GPU time."""
import torch, time, collections

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(1)

# ── reuse the normal train.py setup ──────────────────────────────────
import cv2, os, math, random, numpy as np
from pathlib import Path, PosixPath
import sys
sys.path.append(str(Path(os.path.abspath('')).parent))

from data import loader
from utils import cam_util
from utils.train_util import render, pad_image2even, pad_hw2even, SimpleSampler
from models.ingp_color import Model, TetOptimizer
from fused_ssim import fused_ssim
from utils.args import Args
from utils.graphics_utils import calculate_norm_loss, depth_to_normals

args = Args()
args.tile_size = 4
args.image_folder = "images_4"
args.eval = False
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/profile_run/")
args.iterations = 200
args.ckpt = ""
args.resolution = 1
args.render_train = False
args.max_sh_deg = 3
args.sh_interval = 0
args.sh_step = 1
args.max_norm = 1
args.use_tcnn = False
args.encoding_lr = 3e-3
args.final_encoding_lr = 3e-4
args.network_lr = 3e-3
args.final_network_lr = 3e-4
args.hidden_dim = 64
args.scale_multi = 0.35
args.log2_hashmap_size = 23
args.per_level_scale = 2
args.L = 8
args.hashmap_dim = 8
args.base_resolution = 64
args.density_offset = -4
args.lambda_weight_decay = 1
args.percent_alpha = 0.0
args.spike_duration = 500
args.additional_attr = 0
args.g_init=1e-5
args.s_init=1e-1
args.d_init=0.1
args.c_init=1e-1
args.lr_delay = 0
args.vert_lr_delay = 0
args.vertices_lr = 1e-4
args.final_vertices_lr = 1e-6
args.vertices_lr_delay_multi = 1e-8
args.delaunay_interval = 10
args.freeze_start = 18000
args.freeze_lr = 1e-3
args.final_freeze_lr = 1e-4
args.lambda_dist = 0.0
args.lambda_norm = 0.0
args.lambda_sh = 0.0
args.lambda_opacity = 0.0
args.num_samples = 200
args.k_samples = 1
args.trunc_sigma = 0.35
args.min_tet_count = 9
args.densify_start = 2000
args.densify_end = 16000
args.densify_interval = 500
args.budget = 2_000_000
args.within_thresh = 0.3 / 2.7
args.total_thresh = 2.0
args.clone_min_contrib = 5/255
args.split_min_contrib = 10/255
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

args = Args.from_namespace(args.get_parser().parse_args([
    '--dataset_path', '/optane/nerf_datasets/360/bicycle/',
    '--image_folder', 'images_4',
    '--output_path', 'output/profile_run',
    '--iterations', '200',
]))
args.output_path.mkdir(exist_ok=True, parents=True)

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device=args.data_device,
    eval=args.eval, resolution=args.resolution)
np.savetxt(str(args.output_path / "transform.txt"), scene_info.transform)

device = torch.device('cuda')
model = Model.init_from_pcd(scene_info.point_cloud, train_cameras, device,
                            current_sh_deg=args.max_sh_deg if args.sh_interval <= 0 else 0,
                            **args.as_dict())
tet_optim = TetOptimizer(model, **args.as_dict())

# ── profiling infrastructure ─────────────────────────────────────────
timings = collections.defaultdict(list)

def timed_section(name):
    """Context manager that records GPU time for a named section."""
    class _Timer:
        def __enter__(self):
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return self
        def __exit__(self, *a):
            self.end.record()
            torch.cuda.synchronize()
            timings[name].append(self.start.elapsed_time(self.end))
    return _Timer()

# ── warmup (5 iters, not counted) ───────────────────────────────────
inds = list(range(len(train_cameras)))
random.shuffle(inds)
print("Warming up...")
for i in range(5):
    camera = train_cameras[inds[i]]
    target = camera.original_image.cuda()
    gt_mask = camera.gt_alpha_mask.cuda()
    ray_jitter = torch.rand((camera.image_height, camera.image_width, 2), device=device)
    render_pkg = render(camera, model, ray_jitter=ray_jitter, **args.as_dict())
    image = render_pkg['render']
    l1_loss = ((target - image).abs() * gt_mask).mean()
    l1_loss.backward()
    tet_optim.main_step()
    tet_optim.main_zero_grad()

# ── profiled run ─────────────────────────────────────────────────────
N_ITERS = 200
inds = list(range(len(train_cameras)))
random.shuffle(inds)
print(f"Profiling {N_ITERS} iterations...")
torch.cuda.synchronize()
wall_start = time.time()

for iteration in range(N_ITERS):
    do_delaunay = iteration % args.delaunay_interval == 0

    if do_delaunay:
        with timed_section("delaunay_update"):
            tet_optim.update_triangulation(
                density_threshold=0, alpha_threshold=0, high_precision=False)

    if len(inds) == 0:
        inds = list(range(len(train_cameras)))
        random.shuffle(inds)
    ind = inds.pop()
    camera = train_cameras[ind]

    with timed_section("image_to_gpu"):
        target = camera.original_image.cuda(non_blocking=True)
        gt_mask = camera.gt_alpha_mask.cuda(non_blocking=True)

    with timed_section("ray_jitter"):
        ray_jitter = torch.rand((camera.image_height, camera.image_width, 2), device=device)

    with timed_section("render_total"):
        render_pkg = render(camera, model, ray_jitter=ray_jitter, **args.as_dict())
        image = render_pkg['render']

    with timed_section("loss_compute"):
        l1_loss = ((target - image).abs() * gt_mask).mean()
        l2_loss = ((target - image)**2 * gt_mask).mean()
        reg = tet_optim.regularizer(render_pkg, **args.as_dict())
        ssim_loss = (1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))).clip(min=0, max=1)
        loss = (1-args.lambda_ssim)*l1_loss + \
               args.lambda_ssim*ssim_loss + \
               reg + \
               args.lambda_opacity * (1-render_pkg['alpha']).mean()

    with timed_section("backward"):
        loss.backward()

    with timed_section("optim_step"):
        tet_optim.main_step()
        tet_optim.main_zero_grad()
        if do_delaunay:
            tet_optim.vertex_optim.step()
            tet_optim.vertex_optim.zero_grad()

    with timed_section("lr_update"):
        tet_optim.update_learning_rate(iteration)

torch.cuda.synchronize()
wall_end = time.time()

# ── report ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PROFILING RESULTS  ({N_ITERS} iterations)")
print(f"{'='*60}")
print(f"Wall time: {wall_end - wall_start:.1f}s  ({N_ITERS/(wall_end - wall_start):.2f} it/s)\n")

total_gpu_ms = 0
rows = []
for name, vals in sorted(timings.items(), key=lambda x: -sum(x[1])):
    total = sum(vals)
    avg = total / len(vals)
    total_gpu_ms += total
    rows.append((name, total, avg, len(vals)))

print(f"{'Section':<20} {'Total ms':>10} {'Avg ms':>10} {'Calls':>6} {'% of GPU':>9}")
print(f"{'-'*55}")
for name, total, avg, count in rows:
    pct = 100.0 * total / total_gpu_ms if total_gpu_ms > 0 else 0
    print(f"{name:<20} {total:>10.1f} {avg:>10.2f} {count:>6} {pct:>8.1f}%")
print(f"{'-'*55}")
print(f"{'TOTAL GPU':<20} {total_gpu_ms:>10.1f}")
