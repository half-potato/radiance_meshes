"""Diagnose tiled forward rendering vs hardware raster.

Usage: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python tests/diagnose_tiled_fwd.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

from data import loader
from models.ingp_color import Model
from utils.render_vk import create_vk_renderer, make_vk_vp
from utils.args import Args

device = torch.device("cuda:0")

# Copy args from train_vk.py
args = Args()
args.dataset_path = '/optane/nerf_datasets/360/bicycle'
args.image_folder = 'images_8'
args.max_sh_deg = 0
args.budget = 2_000_000
args.resolution = 1
args.data_device = 'cpu'
args.use_tcnn = False
args.encoding_lr = 3e-3
args.network_lr = 3e-3
args.hidden_dim = 64
args.scale_multi = 0.35
args.log2_hashmap_size = 23
args.per_level_scale = 2
args.L = 8
args.hashmap_dim = 8
args.base_resolution = 64
args.density_offset = -3
args.g_init = 1e-5
args.s_init = 1e-1
args.d_init = 0.1
args.c_init = 1e-1
args.ablate_gradient = False
args.ablate_circumsphere = True
args.ablate_downweighing = False
args.lambda_weight_decay = 1
args.percent_alpha = 0.0
args.spike_duration = 500
args.additional_attr = 0
args.final_encoding_lr = 3e-4
args.final_network_lr = 3e-4

# Load data
train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device=args.data_device,
    eval=True, resolution=args.resolution)

# Create model
model = Model.init_from_pcd(scene_info.point_cloud, train_cameras, device,
                            current_sh_deg=0, **args.as_dict())

cam = train_cameras[0]
W, H = cam.image_width, cam.image_height
print(f"Image: {W}x{H}")
print(f"Vertices: {model.vertices.shape[0]}, Tets: {model.indices.shape[0]}")

# Create renderer
renderer = create_vk_renderer(model, W, H)

# Camera
vp, inv_vp, cam_pos = make_vk_vp(cam, model, device)
cam_np = cam_pos.detach().cpu().numpy().ravel().astype(np.float32)
vp_np = vp.detach().cpu().numpy().ravel().astype(np.float32)
inv_vp_np = inv_vp.detach().cpu().numpy().ravel().astype(np.float32)

# Hardware raster
print("\n=== Hardware Raster Forward ===")
img_hw = renderer.forward(cam_np, vp_np, inv_vp_np)
print(f"  RGB: min={img_hw[:,:,:3].min():.4f} max={img_hw[:,:,:3].max():.4f} mean={img_hw[:,:,:3].mean():.6f}")
print(f"  Alpha: min={img_hw[:,:,3].min():.4f} max={img_hw[:,:,3].max():.4f} mean={img_hw[:,:,3].mean():.4f}")
print(f"  Non-zero alpha: {(img_hw[:,:,3] > 0.01).sum()} / {H*W}")
Image.fromarray((np.clip(img_hw[:,:,:3], 0, 1) * 255).astype(np.uint8)).save("tests/diag_hw_raster.png")

# Tiled forward
print("\n=== Tiled Forward ===")
img_tiled = renderer.forward_tiled(cam_np, vp_np, inv_vp_np)
print(f"  RGB: min={img_tiled[:,:,:3].min():.4f} max={img_tiled[:,:,:3].max():.4f} mean={img_tiled[:,:,:3].mean():.6f}")
print(f"  Alpha: min={img_tiled[:,:,3].min():.4f} max={img_tiled[:,:,3].max():.4f} mean={img_tiled[:,:,3].mean():.4f}")
print(f"  Non-zero alpha: {(img_tiled[:,:,3] > 0.01).sum()} / {H*W}")
Image.fromarray((np.clip(img_tiled[:,:,:3], 0, 1) * 255).astype(np.uint8)).save("tests/diag_tiled.png")

# Visible count
vis_count = renderer.read_instance_count()
print(f"\n  Visible tet count: {vis_count} / {model.indices.shape[0]}")

# Behind-camera analysis
verts = model.vertices
indices = model.indices
vp_dev = vp.to(device)
ones = torch.ones(verts.shape[0], 1, device=device)
homo = torch.cat([verts, ones], dim=1)
clip = homo @ vp_dev
behind = clip[:, 3] <= 0
print(f"\n=== Behind-Camera Analysis ===")
print(f"  Vertices behind: {behind.sum().item()} / {verts.shape[0]}")

v_behind = behind[indices]
any_behind = v_behind.any(dim=1)
all_behind = v_behind.all(dim=1)
print(f"  Tets any_behind: {any_behind.sum().item()} / {indices.shape[0]}")
print(f"  Tets all_behind: {all_behind.sum().item()} / {indices.shape[0]}")

tiles_x = (W + 15) // 16
tiles_y = (H + 15) // 16
num_tiles = tiles_x * tiles_y
n_behind = any_behind.sum().item()
print(f"\n=== Buffer Overflow Analysis ===")
print(f"  Tiles: {tiles_x}x{tiles_y} = {num_tiles}")
print(f"  Behind-camera pairs (now culled): {n_behind} x 0 = 0")
buf_size = 1
while buf_size < max(indices.shape[0] * 4, num_tiles):
    buf_size *= 2
print(f"  max_pairs_pow2: {buf_size:,}")
print(f"  OVERFLOW: {'YES!' if n_behind * num_tiles > buf_size else 'no'}")
if buf_size > 0:
    print(f"  Overflow ratio: {n_behind * num_tiles / buf_size:.1f}x")
