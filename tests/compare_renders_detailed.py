"""Detailed comparison of VK vs Slang renders.

Saves side-by-side images and checks per-pixel OD distribution.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

from data import loader
from models.ingp_color import Model
from utils.train_util import render, RenderGrid

torch.set_num_threads(1)

dataset_path = Path("/optane/nerf_datasets/360/bicycle")
image_folder = "images_8"
resolution = 4
device = torch.device("cuda")

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

# Check vertex extent
verts = model.vertices.detach().cpu().numpy()
print(f"\nVertex extent: min={verts.min(axis=0)}, max={verts.max(axis=0)}")
print(f"Vertex range: {verts.max(axis=0) - verts.min(axis=0)}")
print(f"Scene center: {verts.mean(axis=0)}")

# Camera info
cam_pos = camera.camera_center.cpu().numpy()
print(f"Camera position: {cam_pos}")
print(f"Distance from center: {np.linalg.norm(cam_pos - verts.mean(axis=0)):.3f}")

# === Slang render ===
with torch.no_grad():
    slang_pkg = render(camera, model, min_t=0.4, tile_size=4)
    slang_T = slang_pkg['alpha']  # TRANSMITTANCE
    slang_rgb = slang_pkg['render']  # [3,H,W]
    slang_opacity = 1 - slang_T
    slang_OD = -torch.log(slang_T.clamp(min=1e-10))

# === VK render ===
from utils.render_vk import create_vk_renderer, compute_vk_tensors, make_vk_vp
from rmesh_wgpu.autograd import RMeshForward

with torch.no_grad():
    vk_renderer = create_vk_renderer(model, W, H)
    verts_flat, sh_coeffs, densities, color_grads = compute_vk_tensors(model)

    vp, inv_vp, cam_pos_t = make_vk_vp(camera, model, device)

    image_rgba = RMeshForward.apply(
        vk_renderer, cam_pos_t, vp, inv_vp,
        verts_flat, sh_coeffs, densities, color_grads
    )
    vk_alpha = image_rgba[..., 3].clamp(0, 1)
    vk_rgb = image_rgba[..., :3].clamp(0, 1)
    vk_OD = -torch.log((1 - vk_alpha).clamp(min=1e-10))

    # Read aux buffer
    aux = vk_renderer.read_aux()
    aux = torch.from_numpy(np.array(aux))
    vk_instance_count = vk_renderer.read_instance_count()

print(f"\n=== Summary ===")
print(f"  Slang total OD per pixel: {slang_OD.mean():.4f}")
print(f"  VK total OD per pixel:    {vk_OD.mean():.4f}")
print(f"  VK visible tets: {vk_instance_count}")
print(f"  Ratio: {slang_OD.mean()/vk_OD.mean():.2f}x")

# Per-pixel analysis
slang_OD_np = slang_OD.cpu().numpy()
vk_OD_np = vk_OD.cpu().numpy()

# Find pixels with significant OD in both
slang_hit = slang_OD_np > 0.01
vk_hit = vk_OD_np > 0.01
both_hit = slang_hit & vk_hit

print(f"\n  Pixels with Slang OD > 0.01: {slang_hit.sum()}")
print(f"  Pixels with VK OD > 0.01:    {vk_hit.sum()}")
print(f"  Pixels with both > 0.01:     {both_hit.sum()}")

if both_hit.sum() > 0:
    ratio_at_shared = slang_OD_np[both_hit] / (vk_OD_np[both_hit] + 1e-10)
    print(f"  At shared pixels: Slang/VK ratio = {ratio_at_shared.mean():.2f}x (median {np.median(ratio_at_shared):.2f}x)")

# Check aux buffer for VK hit tets per pixel
aux_dist = aux[..., 3].numpy()
vk_has_data = aux_dist > 0
print(f"\n  VK pixels with any fragment:  {vk_has_data.sum()}")
print(f"  VK aux dist range: [{aux_dist[vk_has_data].min():.6f}, {aux_dist[vk_has_data].max():.6f}]")
print(f"  VK aux dist mean:  {aux_dist[vk_has_data].mean():.6f}")

# Per-tet path length stats
# VK: total_dist = total_OD / density. With uniform density, this is straightforward.
mean_density = densities.mean().item()
vk_total_dist = vk_OD_np / mean_density
slang_total_dist = slang_OD_np / mean_density

print(f"\n  Density: {mean_density:.6f}")
print(f"  Slang total path/pixel: mean={slang_total_dist[slang_hit].mean():.2f}")
print(f"  VK total path/pixel:    mean={vk_total_dist[vk_hit].mean():.2f}")

# Save images for visual comparison
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Slang
slang_img = slang_rgb.cpu().permute(1, 2, 0).numpy().clip(0, 1)
axes[0, 0].imshow(slang_img)
axes[0, 0].set_title(f'Slang RGB (mean OD={slang_OD.mean():.3f})')

axes[0, 1].imshow(slang_OD_np, cmap='hot', vmin=0, vmax=3)
axes[0, 1].set_title('Slang OD per pixel')

axes[0, 2].imshow(slang_opacity.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title('Slang opacity')

# VK
vk_img = vk_rgb.cpu().numpy().clip(0, 1)
axes[1, 0].imshow(vk_img)
axes[1, 0].set_title(f'VK RGB (mean OD={vk_OD.mean():.3f})')

axes[1, 1].imshow(vk_OD_np, cmap='hot', vmin=0, vmax=3)
axes[1, 1].set_title('VK OD per pixel')

axes[1, 2].imshow(vk_alpha.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
axes[1, 2].set_title('VK opacity')

plt.tight_layout()
plt.savefig('/home/dronelab/delaunay_rasterization/tests/compare_renders.png', dpi=100)
print(f"\nSaved comparison image to tests/compare_renders.png")

# Histogram of OD values
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(slang_OD_np.flatten(), bins=50, alpha=0.7, label='Slang', range=(0, 5))
ax1.hist(vk_OD_np.flatten(), bins=50, alpha=0.7, label='VK', range=(0, 5))
ax1.set_xlabel('OD per pixel')
ax1.set_ylabel('Count')
ax1.legend()
ax1.set_title('OD distribution')

if both_hit.sum() > 0:
    ax2.hist(ratio_at_shared, bins=50, range=(0, 50))
    ax2.set_xlabel('Slang/VK OD ratio')
    ax2.set_ylabel('Count')
    ax2.set_title('Per-pixel ratio at shared hits')

plt.tight_layout()
plt.savefig('/home/dronelab/delaunay_rasterization/tests/compare_od_hist.png', dpi=100)
print(f"Saved OD histogram to tests/compare_od_hist.png")
