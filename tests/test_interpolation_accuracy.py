"""Test whether the VK vertex shader's interpolated plane parameters
give correct ray-tet intersection results.

Compares:
1. Exact per-pixel ray-tet intersection (Slang approach)
2. Interpolated plane numerators/denominators from vertex shader (VK approach)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from data import loader
from models.ingp_color import Model
from utils.topo_utils import calculate_circumcenters_torch
from pathlib import Path

torch.set_num_threads(1)

# --- Load scene ---
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
cam_pos = camera.camera_center.to(device)  # [3]
W, H = camera.image_width, camera.image_height
print(f"Camera: {W}x{H}, cam_pos={cam_pos}")
print(f"Model: {model.vertices.shape[0]} verts, {model.indices.shape[0]} tets")

vertices = model.vertices  # [V, 3]
indices = model.indices     # [T, 4]

# VP matrix
vp_t = camera.full_proj_transform.to(device)  # VP^T
vp = vp_t.T  # VP

TET_FACES = [
    [0, 2, 1],
    [1, 2, 3],
    [0, 3, 2],
    [3, 0, 1],
]

def project_to_ndc(pos, vp):
    """Project world position to NDC."""
    clip = vp @ torch.cat([pos, torch.ones(1, device=pos.device)])
    inv_w = 1.0 / (clip[3] + 1e-6)
    return clip[:3] * inv_w, clip[3]


def exact_ray_tet_intersect(ray_o, ray_d, v0, v1, v2, v3):
    """Exact ray-tet intersection via plane clipping (Slang approach)."""
    faces = [(v0, v1, v2, v3), (v0, v1, v3, v2), (v0, v2, v3, v1), (v1, v2, v3, v0)]

    t_enter = -1e30
    t_exit = 1e30

    for va, vb, vc, vopp in faces:
        n = torch.cross(vb - va, vc - va)
        # Flip to point outward (away from opposite vertex)
        if torch.dot(n, vopp - va) > 0:
            n = -n

        dist = -torch.dot(n, va - ray_o)
        denom = torch.dot(n, ray_d)

        if abs(denom) < 1e-10:
            if dist > 0:
                return 0.0, 0.0, 0.0  # Outside
            continue

        tplane = -dist / denom
        if denom < 0:
            t_enter = max(t_enter, tplane.item())
        else:
            t_exit = min(t_exit, tplane.item())

    if t_enter > t_exit or t_exit <= 0:
        return 0.0, 0.0, 0.0

    t_enter = max(t_enter, 0.0)
    dist = max(t_exit - t_enter, 0.0)
    return t_enter, t_exit, dist


def vk_interpolated_intersect(pixel_ndc, face_idx, tet_verts, cam):
    """Simulate VK vertex shader + fragment shader for one face.

    Args:
        pixel_ndc: [2] NDC coordinates of pixel
        face_idx: which face we're rasterizing (0-3)
        tet_verts: [4, 3] world positions of tet vertices
        cam: [3] camera position

    Returns:
        t_min, t_max, dist from VK approach
    """
    face = TET_FACES[face_idx]
    face_world_pos = tet_verts[face]  # [3, 3] - the 3 vertices of this face

    # --- Vertex shader computation at each vertex ---
    ray_dirs = []
    all_numerators = []
    all_denominators = []

    for vi in range(3):
        world_pos = face_world_pos[vi]
        ray_dir = torch.nn.functional.normalize(world_pos - cam, dim=0)
        ray_dirs.append(ray_dir)

        numerators = torch.zeros(4, device=cam.device)
        denominators = torch.zeros(4, device=cam.device)

        for fi in range(4):
            f = TET_FACES[fi]
            va = tet_verts[f[0]]
            vb = tet_verts[f[1]]
            vc = tet_verts[f[2]]
            n = torch.cross(vc - va, vb - va)

            numerators[fi] = torch.dot(n, va - cam)
            denominators[fi] = torch.dot(n, ray_dir)

        all_numerators.append(numerators)
        all_denominators.append(denominators)

    # --- Compute barycentric coordinates of pixel within face ---
    # Project face vertices to screen
    face_screen = []
    face_clip_w = []
    for vi in range(3):
        ndc, w = project_to_ndc(face_world_pos[vi], vp)
        face_screen.append(ndc[:2])
        face_clip_w.append(w)

    # Compute barycentric coords in screen space
    p = pixel_ndc
    v0_s = face_screen[0]
    v1_s = face_screen[1]
    v2_s = face_screen[2]

    d00 = torch.dot(v1_s - v0_s, v1_s - v0_s)
    d01 = torch.dot(v1_s - v0_s, v2_s - v0_s)
    d11 = torch.dot(v2_s - v0_s, v2_s - v0_s)
    d20 = torch.dot(p - v0_s, v1_s - v0_s)
    d21 = torch.dot(p - v0_s, v2_s - v0_s)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return 0.0, 0.0, 0.0

    bary_v = (d11 * d20 - d01 * d21) / denom
    bary_w = (d00 * d21 - d01 * d20) / denom
    bary_u = 1.0 - bary_v - bary_w

    # Check if inside face
    if bary_u < -0.01 or bary_v < -0.01 or bary_w < -0.01:
        return 0.0, 0.0, 0.0

    # --- Perspective-correct interpolation ---
    # GPU rasterizer uses perspective-correct interpolation:
    # attr_interp = (attr0*bary0/w0 + attr1*bary1/w1 + attr2*bary2/w2) / (bary0/w0 + bary1/w1 + bary2/w2)
    w0 = face_clip_w[0].item()
    w1 = face_clip_w[1].item()
    w2 = face_clip_w[2].item()

    # Perspective-correct weights
    pc0 = bary_u.item() / w0
    pc1 = bary_v.item() / w1
    pc2 = bary_w.item() / w2
    denom_pc = pc0 + pc1 + pc2
    if abs(denom_pc) < 1e-10:
        return 0.0, 0.0, 0.0
    pc0 /= denom_pc
    pc1 /= denom_pc
    pc2 /= denom_pc

    # Interpolate ray_dir
    interp_ray_dir = ray_dirs[0] * pc0 + ray_dirs[1] * pc1 + ray_dirs[2] * pc2

    # Interpolate numerators and denominators
    interp_num = all_numerators[0] * pc0 + all_numerators[1] * pc1 + all_numerators[2] * pc2
    interp_den = all_denominators[0] * pc0 + all_denominators[1] * pc1 + all_denominators[2] * pc2

    # --- Fragment shader ---
    d = torch.norm(interp_ray_dir)
    plane_denom = interp_den / d
    all_t = interp_num / plane_denom

    neg_inf = torch.tensor(-3.4e38, device=cam.device)
    pos_inf = torch.tensor(3.4e38, device=cam.device)

    t_enter_vals = torch.where(plane_denom > 0, all_t, neg_inf)
    t_exit_vals = torch.where(plane_denom < 0, all_t, pos_inf)

    t_min = t_enter_vals.max().item()
    t_max = t_exit_vals.min().item()

    dist = max(t_max - t_min, 0.0)
    return t_min, t_max, dist


# --- Pick some representative tets ---
# Find tets that are visible (project within screen bounds)
print("\nFinding visible tets...")

visible_tets = []
with torch.no_grad():
    for tet_id in range(min(indices.shape[0], 50000)):
        v = vertices[indices[tet_id]]  # [4, 3]
        ndcs = []
        behind = False
        for vi in range(4):
            ndc, w = project_to_ndc(v[vi], vp)
            if w < 0:
                behind = True
                break
            ndcs.append(ndc)
        if behind:
            continue
        ndcs = torch.stack(ndcs)
        min_xy = ndcs[:, :2].min(dim=0).values
        max_xy = ndcs[:, :2].max(dim=0).values
        if max_xy[0] < -1 or min_xy[0] > 1 or max_xy[1] < -1 or min_xy[1] > 1:
            continue
        # Compute screen-space extent
        ext_x = (max_xy[0] - min_xy[0]).item() * W / 2
        ext_y = (max_xy[1] - min_xy[1]).item() * H / 2
        if ext_x * ext_y < 4:  # Want reasonably sized tets
            continue
        visible_tets.append((tet_id, ext_x * ext_y))
        if len(visible_tets) >= 200:
            break

visible_tets.sort(key=lambda x: -x[1])  # Sort by screen area
print(f"Found {len(visible_tets)} visible tets")

# --- Test each tet ---
print("\n=== Comparing interpolated vs exact intersection ===")
total_exact_dist = 0.0
total_vk_dist = 0.0
n_tested = 0
n_exact_hit = 0
n_vk_hit = 0
n_vk_miss_exact_hit = 0
large_errors = 0

for idx, (tet_id, area) in enumerate(visible_tets[:50]):
    v = vertices[indices[tet_id]]  # [4, 3]

    # Pick a pixel near the center of the tet's projection
    ndcs = []
    for vi in range(4):
        ndc, w = project_to_ndc(v[vi], vp)
        ndcs.append(ndc)
    ndcs = torch.stack(ndcs)
    center_ndc = ndcs[:, :2].mean(dim=0)

    # Exact ray direction for this pixel
    # Reconstruct world position from NDC using inv_vp
    inv_vp = torch.inverse(vp)
    ndc_point = torch.tensor([center_ndc[0], center_ndc[1], 0.5, 1.0], device=device)
    world_hom = inv_vp @ ndc_point
    world_pos = world_hom[:3] / world_hom[3]
    ray_d = torch.nn.functional.normalize(world_pos - cam_pos, dim=0)

    # Exact intersection
    e_tmin, e_tmax, e_dist = exact_ray_tet_intersect(cam_pos, ray_d, v[0], v[1], v[2], v[3])

    # VK interpolated intersection (try all faces, take the one that hits)
    best_vk_dist = 0.0
    best_face = -1
    vk_dists = []
    for fi in range(4):
        vk_tmin, vk_tmax, vk_dist = vk_interpolated_intersect(center_ndc, fi, v, cam_pos)
        vk_dists.append(vk_dist)
        if vk_dist > best_vk_dist:
            best_vk_dist = vk_dist
            best_face = fi

    n_tested += 1
    if e_dist > 0:
        n_exact_hit += 1
        total_exact_dist += e_dist
    if best_vk_dist > 0:
        n_vk_hit += 1
        total_vk_dist += best_vk_dist

    if e_dist > 0 and best_vk_dist == 0:
        n_vk_miss_exact_hit += 1

    if e_dist > 1e-6:
        ratio = best_vk_dist / e_dist
        if abs(ratio - 1.0) > 0.1:
            large_errors += 1

    if idx < 10 or (e_dist > 0 and abs(best_vk_dist / max(e_dist, 1e-10) - 1.0) > 0.1):
        print(f"\nTet {tet_id} (area={area:.1f}px²):")
        print(f"  Exact: t=[{e_tmin:.6f}, {e_tmax:.6f}], dist={e_dist:.6f}")
        print(f"  VK per face: dists={[f'{d:.6f}' for d in vk_dists]}")
        print(f"  VK best: face={best_face}, dist={best_vk_dist:.6f}")
        if e_dist > 1e-6:
            print(f"  Ratio VK/exact: {best_vk_dist/e_dist:.4f}")

print(f"\n=== Summary ===")
print(f"Tested: {n_tested}")
print(f"Exact hits: {n_exact_hit}")
print(f"VK hits: {n_vk_hit}")
print(f"VK missed but exact hit: {n_vk_miss_exact_hit}")
print(f"Large errors (>10%): {large_errors}")
if n_exact_hit > 0:
    print(f"Avg exact dist: {total_exact_dist/n_exact_hit:.6f}")
if n_vk_hit > 0:
    print(f"Avg VK dist: {total_vk_dist/n_vk_hit:.6f}")
if total_exact_dist > 0:
    print(f"Total dist ratio VK/exact: {total_vk_dist/total_exact_dist:.4f}")
