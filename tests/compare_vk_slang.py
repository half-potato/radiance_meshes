"""Compare VK (wgpu) and Slang renderer outputs for the same model+camera.

Prints per-channel statistics and saves side-by-side images.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import imageio
from pathlib import Path

from data import loader
from models.ingp_color import Model
from utils.train_util import render
from utils.render_vk import render_vk, create_vk_renderer, compute_vk_tensors

torch.set_num_threads(1)

# --- Config ---
dataset_path = Path("/optane/nerf_datasets/360/bicycle")
image_folder = "images_8"
resolution = 4
max_sh_deg = 0
output_dir = Path("output/compare_vk_slang")
output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda")

# --- Load data ---
train_cameras, test_cameras, scene_info = loader.load_dataset(
    dataset_path, image_folder, data_device="cpu", eval=True, resolution=resolution)

# --- Init model ---
model = Model.init_from_pcd(
    scene_info.point_cloud, train_cameras, device,
    current_sh_deg=max_sh_deg,
    max_sh_deg=max_sh_deg,
    ablate_circumsphere=True,
    ablate_gradient=False,
    density_offset=-3,
    use_tcnn=False,
)
model.eval()

camera = train_cameras[0]
print(f"Camera: {camera.image_width}x{camera.image_height}")
print(f"Model: {model.vertices.shape[0]} vertices, {model.indices.shape[0]} tets")
print(f"SH degree: {model.current_sh_deg}, max: {model.max_sh_deg}")

# --- Slang render ---
with torch.no_grad():
    slang_pkg = render(camera, model, min_t=0.4, tile_size=4)
    slang_rgb = slang_pkg['render']  # [3, H, W]
    slang_alpha = slang_pkg['alpha']  # [H, W]

print("\n=== Slang Render ===")
print(f"  RGB shape: {slang_rgb.shape}, range: [{slang_rgb.min():.4f}, {slang_rgb.max():.4f}]")
print(f"  RGB mean per channel: R={slang_rgb[0].mean():.4f} G={slang_rgb[1].mean():.4f} B={slang_rgb[2].mean():.4f}")
print(f"  Alpha shape: {slang_alpha.shape}, range: [{slang_alpha.min():.4f}, {slang_alpha.max():.4f}], mean={slang_alpha.mean():.4f}")

# --- VK render ---
with torch.no_grad():
    vk_renderer = create_vk_renderer(model, camera.image_width, camera.image_height)
    vk_pkg = render_vk(camera, model, vk_renderer)
    vk_rgb = vk_pkg['render']  # [3, H, W]
    vk_alpha = vk_pkg['alpha']  # [H, W]

print("\n=== VK Render ===")
print(f"  RGB shape: {vk_rgb.shape}, range: [{vk_rgb.min():.4f}, {vk_rgb.max():.4f}]")
print(f"  RGB mean per channel: R={vk_rgb[0].mean():.4f} G={vk_rgb[1].mean():.4f} B={vk_rgb[2].mean():.4f}")
print(f"  Alpha shape: {vk_alpha.shape}, range: [{vk_alpha.min():.4f}, {vk_alpha.max():.4f}], mean={vk_alpha.mean():.4f}")

# --- Intermediate tensors ---
with torch.no_grad():
    vertices_flat, sh_coeffs, densities, color_grads = compute_vk_tensors(model)
    nc = (model.current_sh_deg + 1) ** 2
    n_tets = model.indices.shape[0]

    print(f"\n=== VK Tensors ===")
    print(f"  SH coeffs shape: {sh_coeffs.shape}, nc={nc}")
    sh_reshaped = sh_coeffs.reshape(n_tets, nc * 3)
    dc_r = sh_reshaped[:, 0]  # DC for R channel
    print(f"  DC R: mean={dc_r.mean():.4f}, std={dc_r.std():.4f}, range=[{dc_r.min():.4f}, {dc_r.max():.4f}]")
    print(f"  Densities: mean={densities.mean():.4f}, std={densities.std():.4f}, range=[{densities.min():.4f}, {densities.max():.4f}]")
    print(f"  Color grads: mean={color_grads.mean():.6f}, std={color_grads.std():.6f}")

    # What color does the SH pipeline produce? C0 * DC + 0.5
    expected_color = 0.28209479177387814 * dc_r + 0.5
    print(f"  Expected color (C0*DC+0.5) R: mean={expected_color.mean():.4f}")

    # What color after softplus?
    import torch.nn.functional as F
    sp_color = F.softplus(expected_color, beta=10)
    print(f"  After softplus (beta=10) R: mean={sp_color.mean():.4f}")

# --- Also check the raw RGBA from VK ---
print("\n=== Raw VK RGBA ===")
with torch.no_grad():
    from rmesh_wgpu import RMeshRenderer
    from rmesh_wgpu.autograd import RMeshForward

    vp = camera.full_proj_transform.to(device)
    inv_vp = torch.inverse(vp)
    cam_pos = camera.camera_center.to(device)

    image_rgba = RMeshForward.apply(
        vk_renderer, cam_pos, vp, inv_vp,
        vertices_flat, sh_coeffs, densities, color_grads
    )
    print(f"  RGBA shape: {image_rgba.shape}")
    print(f"  R: range=[{image_rgba[...,0].min():.4f}, {image_rgba[...,0].max():.4f}], mean={image_rgba[...,0].mean():.4f}")
    print(f"  G: range=[{image_rgba[...,1].min():.4f}, {image_rgba[...,1].max():.4f}], mean={image_rgba[...,1].mean():.4f}")
    print(f"  B: range=[{image_rgba[...,2].min():.4f}, {image_rgba[...,2].max():.4f}], mean={image_rgba[...,2].mean():.4f}")
    print(f"  A: range=[{image_rgba[...,3].min():.4f}, {image_rgba[...,3].max():.4f}], mean={image_rgba[...,3].mean():.4f}")

# --- Comparison ---
print("\n=== Comparison ===")
diff = (slang_rgb - vk_rgb).abs()
print(f"  L1 diff: {diff.mean():.4f}")
print(f"  Max diff: {diff.max():.4f}")
print(f"  Alpha diff: {(slang_alpha - vk_alpha).abs().mean():.4f}")

# --- Save images ---
def save_img(tensor, path):
    if tensor.dim() == 3 and tensor.shape[0] <= 4:
        tensor = tensor.permute(1, 2, 0)
    img = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    imageio.imwrite(str(path), img)

save_img(slang_rgb, output_dir / "slang_rgb.png")
save_img(vk_rgb, output_dir / "vk_rgb.png")
save_img(slang_alpha.unsqueeze(0), output_dir / "slang_alpha.png")
save_img(vk_alpha.unsqueeze(0), output_dir / "vk_alpha.png")

# Scale up for visibility
save_img((slang_rgb * 5).clamp(0, 1), output_dir / "slang_rgb_5x.png")
save_img((vk_rgb * 5).clamp(0, 1), output_dir / "vk_rgb_5x.png")

# Diff image
save_img((diff * 10).clamp(0, 1), output_dir / "diff_10x.png")

gt = camera.original_image.to(device)
save_img(gt, output_dir / "gt.png")

print(f"\nImages saved to {output_dir}/")
