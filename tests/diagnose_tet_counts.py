"""Compare tet counts between Slang and VK renderers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

from data import loader
from models.ingp_color import Model
from utils.train_util import render, RenderGrid
from delaunay_rasterization.internal.alphablend_tiled_slang_interp import AlphaBlendTiledRender
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader

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

# === Slang: check tile shader output ===
with torch.no_grad():
    tile_size = 4
    render_grid = RenderGrid(H, W, tile_height=tile_size, tile_width=tile_size)
    tcam = dict(
        tile_height=tile_size,
        tile_width=tile_size,
        grid_height=render_grid.grid_height,
        grid_width=render_grid.grid_width,
        min_t=0.4,
        **camera.to_dict(device)
    )
    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _ = vertex_and_tile_shader(
        model.indices, model.vertices, tcam, render_grid)

    n_visible = mask.sum().item()
    n_total = model.indices.shape[0]
    print(f"\n=== Slang Tile Shader ===")
    print(f"  Visible tets (mask): {n_visible}/{n_total} ({100*n_visible/n_total:.1f}%)")
    print(f"  sorted_tetra_idx shape: {sorted_tetra_idx.shape}")
    print(f"  tile_ranges shape: {tile_ranges.shape}")

    # Total tet-tile pairs (how many tet evaluations across all tiles)
    n_tiles = render_grid.grid_height * render_grid.grid_width
    total_tet_tile = sorted_tetra_idx.shape[0]
    print(f"  Total tiles: {n_tiles}")
    print(f"  Total tet-tile pairs: {total_tet_tile}")
    print(f"  Avg tets per tile: {total_tet_tile / n_tiles:.1f}")

    # Per-tile counts
    tile_counts = []
    for t in range(n_tiles):
        start, end = tile_ranges[t]
        tile_counts.append(end.item() - start.item())
    tile_counts = np.array(tile_counts)
    print(f"  Tile tet counts: min={tile_counts.min()}, max={tile_counts.max()}, median={np.median(tile_counts):.0f}, mean={tile_counts.mean():.1f}")

    # Render to get alpha
    slang_pkg = render(camera, model, min_t=0.4, tile_size=4)
    slang_alpha = slang_pkg['alpha']  # this is TRANSMITTANCE = exp(-total_OD)
    slang_opacity = 1 - slang_alpha
    print(f"  Slang transmittance: mean={slang_alpha.mean():.4f}")
    print(f"  Slang opacity (1-T): mean={slang_opacity.mean():.4f}")

    # Total OD
    slang_log_T = torch.log(slang_alpha.clamp(min=1e-10))
    slang_total_OD = -slang_log_T
    print(f"  Slang total OD per pixel: mean={slang_total_OD.mean():.4f}")

    # Get density from cell_values
    cell_values = slang_pkg['cell_values']
    slang_density = cell_values[:, 0]
    print(f"\n  Slang density: mean={slang_density.mean():.6f}, range=[{slang_density.min():.6f}, {slang_density.max():.6f}]")

    # Implied total dist
    mean_density = slang_density.mean().item()
    mean_total_OD = slang_total_OD.mean().item()
    implied_total_dist = mean_total_OD / mean_density if mean_density > 0 else 0
    print(f"  Implied total dist per pixel: {implied_total_dist:.2f}")

# === VK ===
print(f"\n=== VK Renderer ===")
from utils.render_vk import create_vk_renderer, compute_vk_tensors
with torch.no_grad():
    vk_renderer = create_vk_renderer(model, W, H)
    verts_flat, sh_coeffs, densities, color_grads = compute_vk_tensors(model)
    print(f"  VK density: mean={densities.mean():.6f}, range=[{densities.min():.6f}, {densities.max():.6f}]")

    from rmesh_wgpu.autograd import RMeshForward
    vp = camera.full_proj_transform.to(device)
    inv_vp = torch.inverse(vp)
    cam_pos = camera.camera_center.to(device)

    image_rgba = RMeshForward.apply(
        vk_renderer, cam_pos, vp, inv_vp,
        verts_flat, sh_coeffs, densities, color_grads
    )
    vk_opacity = image_rgba[..., 3].clamp(0, 1)
    vk_total_OD = -torch.log(1 - vk_opacity.clamp(max=1-1e-10))
    print(f"  VK opacity: mean={vk_opacity.mean():.4f}")
    print(f"  VK total OD per pixel: mean={vk_total_OD.mean():.4f}")

    vk_implied_total_dist = vk_total_OD.mean().item() / densities.mean().item()
    print(f"  VK implied total dist per pixel: {vk_implied_total_dist:.2f}")

    # Read instance count (number of tets that passed frustum culling)
    vk_visible = vk_renderer.read_instance_count()
    print(f"  VK visible tets (instance_count): {vk_visible}/{n_total} ({100*vk_visible/n_total:.1f}%)")

    print(f"\n=== Comparison ===")
    print(f"  Visible tets: Slang={n_visible} vs VK={vk_visible} (ratio={n_visible/max(vk_visible,1):.2f}x)")
    print(f"  Density match: Slang={slang_density.mean():.6f} vs VK={densities.mean():.6f}")
    print(f"  Opacity: Slang={slang_opacity.mean():.4f} vs VK={vk_opacity.mean():.4f}")
    print(f"  Total OD: Slang={slang_total_OD.mean():.4f} vs VK={vk_total_OD.mean():.4f}")
    print(f"  Ratio (Slang/VK): {slang_total_OD.mean() / vk_total_OD.mean():.2f}x")
