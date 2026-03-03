"""
Decimation Diagnostic Script
==============================
Loads a checkpoint and runs iterative edge-star decimation, measuring
PSNR after each round to verify that the vectorized scoring + collapse
pipeline works correctly.

Usage:
  uv run test_decimation.py --ckpt output/bicycle_exp1_densify --dataset /data/nerf_datasets/360/bicycle
  uv run test_decimation.py --ckpt output/bicycle_exp1_densify --dataset /data/nerf_datasets/360/bicycle --steps 3 --count 10000
"""

import gc
import json
import math
import random
import argparse
import time
from pathlib import Path

import torch
import numpy as np

from data import loader
from models.ingp_color import Model, TetOptimizer
from utils.args import Args
from utils.train_util import render
from utils.decimation import apply_decimation
from fused_ssim import fused_ssim

torch.set_num_threads(1)


def load_model_and_config(ckpt_path, device):
    config = Args.load_from_json(str(Path(ckpt_path) / "config.json"))
    model = Model.load_ckpt(Path(ckpt_path), device)
    return model, config


@torch.no_grad()
def compute_mean_psnr(model, cameras, tile_size=4, min_t=0.4, max_cameras=-1):
    model.eval()
    if max_cameras > 0 and max_cameras < len(cameras):
        indices = list(range(len(cameras)))
        random.shuffle(indices)
        cameras = [cameras[i] for i in indices[:max_cameras]]

    psnrs = []
    for cam in cameras:
        target = cam.original_image.cuda()
        gt_mask = cam.gt_alpha_mask.cuda()
        render_pkg = render(cam, model, tile_size=tile_size, min_t=min_t)
        image = render_pkg['render']

        l2 = ((target - image) ** 2 * gt_mask).mean().item()
        psnr_val = -20 * math.log10(math.sqrt(max(l2, 1e-10)))
        psnrs.append(psnr_val)

        del render_pkg, image, target, gt_mask
        torch.cuda.empty_cache()

    return np.mean(psnrs)


def parse_args():
    parser = argparse.ArgumentParser(description="Decimation diagnostics")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Checkpoint directory")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset path (overrides config)")
    parser.add_argument("--image-folder", type=str, default="",
                        help="Image folder (overrides config)")
    parser.add_argument("--output-dir", type=str,
                        default="output/decimation_diagnostics",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of decimation rounds")
    parser.add_argument("--count", type=int, default=5000,
                        help="Vertices to remove per step")
    parser.add_argument("--max-eval-cameras", type=int, default=-1,
                        help="Max cameras for PSNR eval (-1 = all test cameras)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    cli_args = parse_args()

    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)

    device = torch.device('cuda')
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = Args.load_from_json(str(Path(cli_args.ckpt) / "config.json"))
    config.decimate_count = cli_args.count

    dataset_path = cli_args.dataset if cli_args.dataset else config.dataset_path
    image_folder = cli_args.image_folder if cli_args.image_folder else config.image_folder

    # Load dataset
    print(f"Loading dataset from {dataset_path} ({image_folder})...")
    train_cameras, test_cameras, scene_info = loader.load_dataset(
        Path(dataset_path), image_folder, data_device='cpu',
        eval=config.eval if hasattr(config, 'eval') else False,
        resolution=config.resolution if hasattr(config, 'resolution') else 1)
    eval_cameras = test_cameras if test_cameras else train_cameras
    print(f"  {len(train_cameras)} train, {len(test_cameras)} test cameras")

    # Load model
    print(f"Loading checkpoint from {cli_args.ckpt}...")
    model, _ = load_model_and_config(cli_args.ckpt, device)
    tet_optim = TetOptimizer(model, **config.as_dict())
    print(f"  {model.vertices.shape[0]} vertices, {model.indices.shape[0]} tets, "
          f"{model.num_int_verts} interior")

    # Baseline PSNR
    print("\nMeasuring baseline PSNR...")
    baseline_psnr = compute_mean_psnr(
        model, eval_cameras,
        tile_size=config.tile_size, min_t=config.min_t,
        max_cameras=cli_args.max_eval_cameras)
    print(f"  Baseline: {baseline_psnr:.4f} dB")

    # Run decimation steps
    results = []
    results.append(dict(
        step=0,
        n_verts=model.vertices.shape[0],
        n_int_verts=model.num_int_verts,
        n_tets=model.indices.shape[0],
        psnr=baseline_psnr,
        time_s=0.0,
        removed=0,
    ))

    prev_psnr = baseline_psnr
    print(f"\n{'Step':>4s} | {'#V':>8s} | {'#IntV':>8s} | {'#T':>9s} | "
          f"{'PSNR':>8s} | {'dPSNR':>8s} | {'Time':>6s} | {'Removed':>7s}")
    print("-" * 78)
    print(f"{'0':>4s} | {model.vertices.shape[0]:>8d} | {model.num_int_verts:>8d} | "
          f"{model.indices.shape[0]:>9d} | {baseline_psnr:>8.4f} | {'---':>8s} | "
          f"{'---':>6s} | {'---':>7s}")

    for step in range(1, cli_args.steps + 1):
        st = time.time()
        n_removed = apply_decimation(model, tet_optim, config, device)
        elapsed = time.time() - st

        psnr = compute_mean_psnr(
            model, eval_cameras,
            tile_size=config.tile_size, min_t=config.min_t,
            max_cameras=cli_args.max_eval_cameras)
        delta = psnr - prev_psnr

        results.append(dict(
            step=step,
            n_verts=model.vertices.shape[0],
            n_int_verts=model.num_int_verts,
            n_tets=model.indices.shape[0],
            psnr=psnr,
            time_s=elapsed,
            removed=n_removed,
        ))

        print(f"{step:>4d} | {model.vertices.shape[0]:>8d} | {model.num_int_verts:>8d} | "
              f"{model.indices.shape[0]:>9d} | {psnr:>8.4f} | {delta:>+8.4f} | "
              f"{elapsed:>5.1f}s | {n_removed:>7d}")
        prev_psnr = psnr

        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    total_removed = sum(r['removed'] for r in results)
    total_time = sum(r['time_s'] for r in results)
    final_psnr = results[-1]['psnr']
    print(f"\n=== Summary ===")
    print(f"Total removed: {total_removed} vertices in {total_time:.1f}s")
    print(f"PSNR: {baseline_psnr:.4f} -> {final_psnr:.4f} "
          f"({final_psnr - baseline_psnr:+.4f} dB)")
    print(f"Vertices: {results[0]['n_verts']} -> {results[-1]['n_verts']}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(dict(
            ckpt=cli_args.ckpt,
            count_per_step=cli_args.count,
            steps=results,
        ), f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
