"""Evaluate a trained radiance-mesh checkpoint via the slang ray tracer.

Mirrors evaluate.py's metric flow (PSNR / SSIM / LPIPS over the test set,
metrics.json + per-frame PNGs) but renders with TetrahedralRayTrace instead of
the tile-based rasterizer. Walking adjacency from the camera's containing tet
should converge to the same image as the rasterizer up to floating-point and
end-of-walk transparency differences -- so PSNR is also a regression check on
the kernel.
"""

import json
import time
from pathlib import Path

import imageio
import mediapy
import numpy as np
import torch
from fused_ssim import fused_ssim
from icecream import ic
from submodules.lpipsPyTorch import LPIPSEval
from tqdm import tqdm

from data import loader
from delaunay_rasterization.internal.ray_trace import TetrahedralRayTrace
from dtlookup import TetrahedraLookup
from utils import cam_util
from utils.args import Args
from utils.test_util import psnr
from utils.topo_utils import build_adj


args = Args()
args.tile_size = 4
args.image_folder = "images_4"
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/test/")
args.eval = True
args.use_ply = False
args.render_train = False
args.min_t = 0.2
args.max_iters = 256
args = Args.from_namespace(args.get_parser().parse_args())

device = torch.device('cuda')
if args.use_ply:
    from models.tet_color import Model
    model = Model.load_ply(args.output_path / "ckpt.ply", device)
else:
    # Current iNGP-style checkpoints use models.ingp_color; frozen-geometry
    # checkpoints use models.frozen. Try each in order and use whichever
    # loads cleanly.
    from models.ingp_color import Model as IngpModel
    from models.frozen import FrozenTetModel

    last_err = None
    model = None
    for cls in (IngpModel, FrozenTetModel):
        try:
            model = cls.load_ckpt(args.output_path, device)
            print(f"Loaded checkpoint with {cls.__module__}.{cls.__name__}")
            break
        except Exception as e:
            last_err = e
    if model is None:
        raise RuntimeError(
            f"Could not load checkpoint at {args.output_path}: {last_err}")
model.eval()
# load_ckpt resets current_sh_deg to the constructor default (2), which is
# wrong when max_sh_deg<2 (eval_sh assert fires). Clamp.
if hasattr(model, 'current_sh_deg') and hasattr(model, 'max_sh_deg'):
    model.current_sh_deg = min(model.current_sh_deg, model.max_sh_deg)

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cpu", eval=args.eval)

ic(model.min_t, args.min_t)
min_t = float(args.min_t) if args.min_t is not None else float(model.min_t)

# Walk the full mesh -- active + empty -- so a ray that starts in an empty
# "sky" tet can still walk into the populated tets via adjacency.
full_indices = torch.cat([model.indices, model.empty_indices], dim=0).contiguous()
n_active = int(model.indices.shape[0])
n_empty = int(model.empty_indices.shape[0])
print(f"Mesh: {n_active} active tets + {n_empty} empty tets = {full_indices.shape[0]} total")

tet_adj = build_adj(model.vertices.detach(), full_indices.long(), device='cuda').int().contiguous()
lookup = TetrahedraLookup(full_indices, model.vertices.detach(), 256)

splits = [('test', test_cameras)]
if args.render_train:
    splits = [('train', train_cameras), ('test', test_cameras)]

lpips_eval = LPIPSEval(net_type='vgg', device='cuda')

results = {}
short_results = {}
out_root = args.output_path / "ray_traced"
out_root.mkdir(parents=True, exist_ok=True)

for split, cameras in splits:
    # Camera-space ray dirs are constant across frames if intrinsics match;
    # cache them off the first camera and reuse.
    cam_dirs = cameras[0].get_camera_space_directions('cuda')

    pred_dir = out_root / "pred" / split
    gt_dir = out_root / "gt" / split
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    psnrs, ssims, lpipss, fpss = [], [], [], []

    for idx, camera in enumerate(tqdm(cameras, desc=f"RT {split}")):
        with torch.no_grad():
            # Activated cell values for the active tets, zeros for empties.
            _, active_features = model.get_cell_values(camera)
            cell_values = torch.cat([
                active_features,
                torch.zeros(
                    (n_empty, active_features.shape[1]),
                    device=device, dtype=active_features.dtype),
            ], dim=0).contiguous()

            rays = camera.get_world_space_rays(cam_dirs, 'cuda')
            # Shift origins by min_t so the per-ray start-tet lookup matches
            # the same point the kernel will use for its first eval_tet_rt
            # (kernel applies a min_t=0 shift, i.e. no shift, since we did it
            # here). Otherwise the kernel can land just past the start tet
            # and exit at iter 1.
            ray_d_norm = rays[:, 3:6] / rays[:, 3:6].norm(dim=1, keepdim=True).clamp_min(1e-8)
            shifted_origin = rays[:, :3] + ray_d_norm * min_t
            start_tet_ids = lookup.lookup(shifted_origin.contiguous()).to(torch.int32)

            torch.cuda.synchronize()
            t0 = time.time()
            output, aux = TetrahedralRayTrace.apply(
                rays, full_indices, model.vertices.detach(), cell_values,
                tet_adj, start_tet_ids, 0.0, int(args.max_iters))
            torch.cuda.synchronize()
            fpss.append(1.0 / max(time.time() - t0, 1e-6))

            H, W = camera.image_height, camera.image_width
            image_chw = (output[:, :3]
                         .reshape(H, W, 3)
                         .permute(2, 0, 1)
                         .clamp(0.0, 1.0)
                         .contiguous()
                         .unsqueeze(0))

            gt = camera.original_image.to(device).unsqueeze(0)
            if gt.shape[1] != 3:
                gt = gt[:, :3]

            psnr_val = psnr(image_chw, gt).item()
            ssim_val = fused_ssim(image_chw, gt).item()
            lpips_val = lpips_eval.criterion(2 * image_chw - 1, 2 * gt - 1).item()

            psnrs.append(psnr_val)
            ssims.append(ssim_val)
            lpipss.append(lpips_val)
            results[f"{split}_{idx:04d}"] = {
                "PSNR": psnr_val, "SSIM": ssim_val, "LPIPS": lpips_val,
            }

            imageio.imwrite(
                pred_dir / f"{idx:04d}.png",
                (image_chw[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            imageio.imwrite(
                gt_dir / f"{idx:04d}.png",
                (gt[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    means = {
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
        "LPIPS": float(np.mean(lpipss)),
        "FPS": float(np.mean(fpss)),
    }
    results[f"{split}_mean"] = means
    short_results.update({f"{split}_{k}": v for k, v in means.items()})

    print(f"\n{split.upper()} (ray traced):")
    print(f"  PSNR : {means['PSNR']:>12.7f}")
    print(f"  SSIM : {means['SSIM']:>12.7f}")
    print(f"  LPIPS: {means['LPIPS']:>12.7f}")
    print(f"  FPS  : {means['FPS']:>12.4f}")

with open(args.output_path / "metrics_ray_traced.json", "w") as f:
    json.dump(results, f, indent=4)
print(f"Metrics saved to {args.output_path / 'metrics_ray_traced.json'}")

# Rotating camera fly-through, mirroring evaluate.py.
with torch.no_grad():
    epath = cam_util.generate_cam_path(train_cameras, 400)
    eimages = []
    cam_dirs = epath[0].get_camera_space_directions('cuda')
    for camera in tqdm(epath, desc="Fly-through"):
        _, active_features = model.get_cell_values(camera)
        cell_values = torch.cat([
            active_features,
            torch.zeros(
                (n_empty, active_features.shape[1]),
                device=device, dtype=active_features.dtype),
        ], dim=0).contiguous()

        rays = camera.get_world_space_rays(cam_dirs, 'cuda')
        ray_d_norm = rays[:, 3:6] / rays[:, 3:6].norm(dim=1, keepdim=True).clamp_min(1e-8)
        shifted_origin = rays[:, :3] + ray_d_norm * min_t
        start_tet_ids = lookup.lookup(shifted_origin.contiguous()).to(torch.int32)

        output, _ = TetrahedralRayTrace.apply(
            rays, full_indices, model.vertices.detach(), cell_values,
            tet_adj, start_tet_ids, 0.0, int(args.max_iters))
        H, W = camera.image_height, camera.image_width
        eimages.append(
            output[:, :3].reshape(H, W, 3).clamp(0.0, 1.0).cpu().numpy())

mediapy.write_video(args.output_path / "rotating_rt.mp4", eimages)
print(f"Fly-through saved to {args.output_path / 'rotating_rt.mp4'}")
