"""Verify VP projection matches forward render positions."""
import numpy as np

width, height = 16, 16
cam_pos = np.array([0.15, 0.1, 1.5], dtype=np.float32)
fov = 1.0
aspect = width / height
znear, zfar = 0.01, 100.0
f = 1.0 / np.tan(fov / 2.0)

proj = np.zeros((4, 4), dtype=np.float32)
proj[0, 0] = f / aspect
proj[1, 1] = f
proj[2, 2] = zfar / (zfar - znear)
proj[2, 3] = 1.0
proj[3, 2] = -(zfar * znear) / (zfar - znear)

view = np.eye(4, dtype=np.float32)
view[2, 2] = -1.0
view[3, 2] = cam_pos[2]

vp = (view @ proj).astype(np.float32)
inv_vp = np.linalg.inv(vp).astype(np.float32)

print("VP matrix (numpy, row-major):")
print(vp)
print("\nProj matrix:")
print(proj)
print("\nView matrix:")
print(view)

# shader_vp * v = vp^T @ v (column vector multiplication)
# numpy convention: v_row @ vp = clip_row
# So: clip = v_row @ vp, ndc = clip[:3] / clip[3]

vertices = {
    'v0': [0.0, 0.0, 0.0],
    'v1': [0.3, 0.0, 0.0],
    'v2': [0.15, 0.3, 0.0],
    'v3': [0.15, 0.1, 0.3],
    'v4': [0.15, 0.1, -0.3],
}

print("\nVertex projections (numpy row-vector convention: v_row @ vp):")
for name, v in vertices.items():
    v4 = np.array(v + [1.0], dtype=np.float32)
    clip = v4 @ vp
    if clip[3] <= 0:
        print(f"  {name}{v}: BEHIND CAMERA (clip.w={clip[3]:.4f})")
        continue
    ndc = clip[:3] / clip[3]
    px = (ndc[0] + 1) * 0.5 * width
    py = (ndc[1] + 1) * 0.5 * height
    print(f"  {name}{v}: clip=[{clip[0]:.4f},{clip[1]:.4f},{clip[2]:.4f},{clip[3]:.4f}] "
          f"ndc=[{ndc[0]:.4f},{ndc[1]:.4f}] pixel=({px:.1f},{py:.1f})")

# Also test shader convention: shader_vp * v_col = vp^T @ v_col
print("\nVertex projections (shader column-vector convention: vp^T @ v_col):")
vp_t = vp.T
for name, v in vertices.items():
    v4 = np.array(v + [1.0], dtype=np.float32)
    clip = vp_t @ v4
    if clip[3] <= 0:
        print(f"  {name}{v}: BEHIND CAMERA (clip.w={clip[3]:.4f})")
        continue
    ndc = clip[:3] / clip[3]
    px = (ndc[0] + 1) * 0.5 * width
    py = (ndc[1] + 1) * 0.5 * height
    print(f"  {name}{v}: clip=[{clip[0]:.4f},{clip[1]:.4f},{clip[2]:.4f},{clip[3]:.4f}] "
          f"ndc=[{ndc[0]:.4f},{ndc[1]:.4f}] pixel=({px:.1f},{py:.1f})")

# Check backward NDC -> ray direction for a few pixels
print("\nBackward NDC -> ray (using inv_vp^T @ ndc_point, i.e. shader convention):")
inv_vp_t = inv_vp.T  # = shader_inv_vp
for py_test in [5, 6, 7, 8, 9, 10]:
    for px_test in [7, 8, 9, 10]:
        ndc_x = (2.0 * (px_test + 0.5) / width) - 1.0
        ndc_y = (2.0 * (py_test + 0.5) / height) - 1.0
        near_clip = inv_vp_t @ np.array([ndc_x, ndc_y, 0, 1], dtype=np.float32)
        far_clip = inv_vp_t @ np.array([ndc_x, ndc_y, 1, 1], dtype=np.float32)
        near_w = near_clip[:3] / near_clip[3]
        far_w = far_clip[:3] / far_clip[3]
        ray_dir = far_w - near_w
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        # Check if ray intersects tet 1 (vertices 3,2,1,0)
        # Just print ray origin and direction
        print(f"  pixel ({px_test},{py_test}): ndc=({ndc_x:.3f},{ndc_y:.3f}) "
              f"ray_origin=({near_w[0]:.4f},{near_w[1]:.4f},{near_w[2]:.4f}) "
              f"ray_dir=({ray_dir[0]:.4f},{ray_dir[1]:.4f},{ray_dir[2]:.4f})")
