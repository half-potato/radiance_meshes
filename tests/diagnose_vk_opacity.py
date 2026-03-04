"""Diagnose VK opacity discrepancy by analyzing the aux buffer.

The aux buffer contains (t_min, t_max, optical_depth, dist) for the LAST
fragment rendered at each pixel (nearest tet in back-to-front order).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

from data import loader
from models.ingp_color import Model
from utils.render_vk import create_vk_renderer, compute_vk_tensors
from rmesh_wgpu import RMeshRenderer
from rmesh_wgpu.autograd import RMeshForward

torch.set_num_threads(1)

# --- Config ---
dataset_path = Path("/optane/nerf_datasets/360/bicycle")
image_folder = "images_8"
resolution = 4
device = torch.device("cuda")

# --- Load data ---
train_cameras, test_cameras, scene_info = loader.load_dataset(
    dataset_path, image_folder, data_device="cpu", eval=True, resolution=resolution)

model = Model.init_from_pcd(
    scene_info.point_cloud, train_cameras, device,
    current_sh_deg=0, max_sh_deg=0,
    ablate_circumsphere=True, ablate_gradient=False,
    density_offset=-3, use_tcnn=False)
model.eval()

camera = train_cameras[0]
W, H = camera.image_width, camera.image_height
print(f"Camera: {W}x{H}")
print(f"Model: {model.vertices.shape[0]} verts, {model.indices.shape[0]} tets")

# --- Create VK renderer ---
with torch.no_grad():
    vk_renderer = create_vk_renderer(model, W, H)
    vertices_flat, sh_coeffs, densities, color_grads = compute_vk_tensors(model)

    vp = camera.full_proj_transform.to(device)
    inv_vp = torch.inverse(vp)
    cam_pos = camera.camera_center.to(device)

    # Forward render
    image_rgba = RMeshForward.apply(
        vk_renderer, cam_pos, vp, inv_vp,
        vertices_flat, sh_coeffs, densities, color_grads
    )

    # Read aux buffer
    aux = vk_renderer.read_aux()  # numpy [H, W, 4]
    aux = torch.from_numpy(np.array(aux))

print(f"\n=== RGBA Output ===")
print(f"  RGB: range=[{image_rgba[..., :3].min():.6f}, {image_rgba[..., :3].max():.6f}], mean={image_rgba[..., :3].mean():.6f}")
print(f"  Alpha: range=[{image_rgba[..., 3].min():.6f}, {image_rgba[..., 3].max():.6f}], mean={image_rgba[..., 3].mean():.6f}")

print(f"\n=== Aux Buffer (nearest tet per pixel) ===")
t_min = aux[..., 0]
t_max = aux[..., 1]
od = aux[..., 2]
dist = aux[..., 3]

# Filter to pixels that have data (dist > 0)
has_data = dist > 0
n_total = dist.numel()
n_hit = has_data.sum().item()
print(f"  Pixels with data: {n_hit}/{n_total} ({100*n_hit/n_total:.1f}%)")

if n_hit > 0:
    t_min_hit = t_min[has_data]
    t_max_hit = t_max[has_data]
    od_hit = od[has_data]
    dist_hit = dist[has_data]

    print(f"\n  t_min: range=[{t_min_hit.min():.6f}, {t_min_hit.max():.6f}], mean={t_min_hit.mean():.6f}, median={t_min_hit.median():.6f}")
    print(f"  t_max: range=[{t_max_hit.min():.6f}, {t_max_hit.max():.6f}], mean={t_max_hit.mean():.6f}, median={t_max_hit.median():.6f}")
    print(f"  dist:  range=[{dist_hit.min():.6f}, {dist_hit.max():.6f}], mean={dist_hit.mean():.6f}, median={dist_hit.median():.6f}")
    print(f"  OD:    range=[{od_hit.min():.6f}, {od_hit.max():.6f}], mean={od_hit.mean():.6f}, median={od_hit.median():.6f}")

    # Implied density = OD / dist
    implied_density = od_hit / (dist_hit + 1e-10)
    print(f"  implied density: mean={implied_density.mean():.4f}, median={implied_density.median():.4f}")

    # Check for suspicious patterns
    print(f"\n  Negative t_min: {(t_min_hit < 0).sum().item()}")
    print(f"  t_min > t_max: {(t_min_hit > t_max_hit).sum().item()}")
    print(f"  dist == 0 but has_data: impossible by filter")
    print(f"  OD < 0: {(od_hit < 0).sum().item()}")

    # Percentiles
    for p in [1, 5, 25, 50, 75, 95, 99]:
        pct = np.percentile(dist_hit.numpy(), p)
        print(f"  dist p{p}: {pct:.6f}")

    # What alpha does this nearest tet produce?
    nearest_alpha = 1 - torch.exp(-od_hit)
    print(f"\n  Nearest tet alpha: mean={nearest_alpha.mean():.6f}, median={nearest_alpha.median():.6f}")

    # Compare with actual pixel alpha
    alpha = image_rgba[..., 3]
    alpha_hit = alpha[has_data.to(device)]
    print(f"  Actual pixel alpha (all tets): mean={alpha_hit.mean():.6f}")
    print(f"  Ratio (actual/nearest): {alpha_hit.mean() / nearest_alpha.mean():.4f}")

# --- Also check densities that were passed in ---
print(f"\n=== Density Stats ===")
dens_np = densities.detach().cpu().numpy()
print(f"  Range: [{dens_np.min():.6f}, {dens_np.max():.6f}]")
print(f"  Mean: {dens_np.mean():.6f}, Median: {np.median(dens_np):.6f}")
print(f"  Non-positive: {(dens_np <= 0).sum()}")

# --- Render with 10x density to see if alpha scales correctly ---
print(f"\n=== 10x Density Test ===")
with torch.no_grad():
    densities_10x = densities * 10
    vk_renderer.update_densities(densities_10x.detach().cpu().numpy())
    image_10x = RMeshForward.apply(
        vk_renderer, cam_pos, vp, inv_vp,
        vertices_flat, sh_coeffs, densities_10x, color_grads
    )
    alpha_10x = image_10x[..., 3]
    print(f"  1x density alpha mean: {image_rgba[..., 3].mean():.6f}")
    print(f"  10x density alpha mean: {alpha_10x.mean():.6f}")
    # If alpha = 1-exp(-OD), then 10x density → 10x OD
    # Expected: alpha_10x = 1 - exp(-10*OD) = 1 - (1-alpha_1x)^10
    alpha_1x_mean = image_rgba[..., 3].mean().item()
    expected_10x = 1 - (1 - alpha_1x_mean) ** 10
    print(f"  Expected 10x alpha (from 1x): {expected_10x:.6f}")
    # Restore original densities
    vk_renderer.update_densities(densities.detach().cpu().numpy())
