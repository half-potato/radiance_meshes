"""Test different min_t values to diagnose floater location (near camera vs scene geometry)."""
from utils.train_util import render
import torch
from tqdm import tqdm
from pathlib import Path
from data import loader
from utils.args import Args
from fused_ssim import fused_ssim
import json
import numpy as np

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

args = Args()
args.tile_size = 4
args.image_folder = "images_4"
args.dataset_path = Path("/data/nerf_datasets/360/bicycle")
args.output_path = Path("output/bicycle_ifimages_4_i4000_lel0.0_dc0_ds3000_di1000_di200_nl0.0_dt0.0_lo0.0_do-3_ld0.0_lsb0.0_ls0.2_v13")
args.eval = True
args = Args.from_namespace(args.get_parser().parse_args())

device = torch.device('cuda')
from models.ingp_color import Model
from models.frozen import FrozenTetModel
try:
    model = Model.load_ckpt(args.output_path, device)
except:
    model = FrozenTetModel.load_ckpt(args.output_path, device)

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cpu", eval=args.eval)

# Test different min_t values
min_t_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
problem_views = ['test_0018', 'test_0021', 'test_0009']

print("=" * 80)
print(f"Testing min_t values on checkpoint: {args.output_path.name}")
print(f"Model was trained with min_t = {model.min_t}")
print("=" * 80)

results = {}
for min_t in min_t_values:
    print(f"\nTesting min_t = {min_t:.1f}")
    psnrs = {}

    for idx, camera in enumerate(tqdm(test_cameras, desc=f"min_t={min_t:.1f}")):
        with torch.no_grad():
            render_pkg = render(camera, model, tile_size=args.tile_size, min_t=min_t)
            image = render_pkg['render'].clip(min=0, max=1).unsqueeze(0)
            gt = camera.original_image.cuda().unsqueeze(0)
            psnr_val = psnr(image, gt).item()

            view_name = f"test_{idx:04d}"
            psnrs[view_name] = psnr_val

    results[min_t] = psnrs

    # Print summary
    overall = np.mean(list(psnrs.values()))
    problem = np.mean([psnrs[v] for v in problem_views])
    other = np.mean([v for k, v in psnrs.items() if k not in problem_views])
    print(f"  Overall: {overall:.2f} dB, Problem: {problem:.2f} dB, Other: {other:.2f} dB, Gap: {problem-other:+.2f} dB")

# Analysis
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'min_t':<8} {'Overall':>9} {'Problem':>9} {'Other':>9} {'Gap':>8}")
print("-" * 80)

baseline_overall = np.mean(list(results[0.4].values()))  # 0.4 is training value
for min_t in min_t_values:
    psnrs = results[min_t]
    overall = np.mean(list(psnrs.values()))
    problem = np.mean([psnrs[v] for v in problem_views])
    other = np.mean([v for k, v in psnrs.items() if k not in problem_views])
    gap = problem - other

    marker = " *" if min_t == 0.4 else ""
    print(f"{min_t:<8.1f} {overall:>9.2f} {problem:>9.2f} {other:>9.2f} {gap:>+8.2f}{marker}")

# Per-view analysis for problem views
print("\n" + "=" * 80)
print("PROBLEM VIEWS DETAIL")
print("=" * 80)

for view in problem_views:
    print(f"\n{view}:")
    print(f"{'min_t':<8} {'PSNR':>9} {'Δ vs 0.4':>11}")
    print("-" * 30)
    baseline = results[0.4][view]
    for min_t in min_t_values:
        psnr_val = results[min_t][view]
        delta = psnr_val - baseline
        marker = " *" if min_t == 0.4 else ""
        print(f"{min_t:<8.1f} {psnr_val:>9.2f} {delta:>+11.2f}{marker}")

# Save results
output_file = args.output_path / "tmin_sweep.json"
json_results = {str(k): v for k, v in results.items()}
with open(output_file, 'w') as f:
    json.dump(json_results, f, indent=2)
print(f"\nResults saved to {output_file}")

# Conclusion
print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("\nIf PSNR improves with higher min_t:")
print("  → Floaters are NEAR CAMERA (skipping near-field artifacts helps)")
print("\nIf PSNR degrades with higher min_t:")
print("  → Floaters are SCENE GEOMETRY (grass), skipping them hurts")
print("\nIf PSNR is stable across min_t:")
print("  → Issue is not related to depth/floaters")
