"""CPU reference for convex hull scanline fill of projected tetrahedra.

Tests the rasterization algorithm that will go into forward_tiled_compute.wgsl:
given 4 projected 2D points (tet vertices), compute which pixels are covered
by their convex hull.

Usage:
    uv run python tests/test_scanline_fill.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from scipy.spatial import ConvexHull, Delaunay


def convex_hull_2d(points):
    """Return vertices of 2D convex hull in order. points: (N, 2)."""
    if len(points) < 3:
        return points
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except Exception:
        return points


def scanline_fill(hull_verts, x_min, x_max, y_min, y_max):
    """Scanline fill of a convex polygon within [x_min, x_max) x [y_min, y_max).

    hull_verts: (K, 2) convex hull vertices in order (CW or CCW).
    Returns list of (px, py) pixel coordinates covered by the hull.

    For each integer row y in [y_min, y_max), find the x-range where the
    hull covers by intersecting all hull edges with the horizontal line y+0.5.
    """
    K = len(hull_verts)
    if K < 3:
        return []

    pixels = []

    for y in range(y_min, y_max):
        yc = y + 0.5  # pixel center

        # Find x-intersections of all hull edges with y = yc
        x_intersections = []
        for i in range(K):
            j = (i + 1) % K
            y0, y1 = hull_verts[i, 1], hull_verts[j, 1]
            x0, x1 = hull_verts[i, 0], hull_verts[j, 0]

            # Does this edge cross y = yc?
            if (y0 <= yc and y1 > yc) or (y1 <= yc and y0 > yc):
                # Linear interpolation for x at y = yc
                t = (yc - y0) / (y1 - y0)
                x_at_y = x0 + t * (x1 - x0)
                x_intersections.append(x_at_y)

        if len(x_intersections) < 2:
            continue

        xl = min(x_intersections)
        xr = max(x_intersections)

        # Fill pixels in [xl, xr] range, clamped to [x_min, x_max)
        px_start = max(int(np.floor(xl)), x_min)
        px_end = min(int(np.floor(xr)), x_max - 1)

        for x in range(px_start, px_end + 1):
            xc = x + 0.5
            if xc >= xl and xc <= xr:
                pixels.append((x, y))

    return pixels


def reference_fill(hull_verts, x_min, x_max, y_min, y_max):
    """Reference: use matplotlib Path to test each pixel center."""
    path = Path(np.vstack([hull_verts, hull_verts[0]]))  # close the path
    pixels = []
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            if path.contains_point((x + 0.5, y + 0.5)):
                pixels.append((x, y))
    return pixels


def project_tet(verts_3d, vp, width, height):
    """Project 4 tet vertices to pixel coordinates.

    verts_3d: (4, 3)
    vp: (4, 4) view-projection matrix (row-major, post-multiply convention:
         clip = vp @ [x,y,z,1]^T ... but we use column-major like wgpu)

    Returns: (4, 2) pixel coordinates, or None if any vertex behind camera.
    """
    pts = []
    for i in range(4):
        v = np.append(verts_3d[i], 1.0)
        clip = vp @ v
        if clip[3] <= 0:
            return None  # behind camera
        ndc = clip[:3] / clip[3]
        px = (ndc[0] + 1.0) * 0.5 * width
        py = (1.0 - ndc[1]) * 0.5 * height
        pts.append([px, py])
    return np.array(pts)


def make_vp_matrix(cam_pos, fov=1.0, width=64, height=64, znear=0.1, zfar=100.0):
    """Create a VP matrix matching the test_vk_backward.py convention."""
    aspect = width / height
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

    return (view @ proj).astype(np.float32)


def visualize_comparison(hull_2d, scanline_pixels, ref_pixels, width, height, title=""):
    """Show scanline vs reference fill side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Scanline result
    img_scan = np.zeros((height, width), dtype=np.float32)
    for x, y in scanline_pixels:
        if 0 <= x < width and 0 <= y < height:
            img_scan[y, x] = 1.0

    # Reference result
    img_ref = np.zeros((height, width), dtype=np.float32)
    for x, y in ref_pixels:
        if 0 <= x < width and 0 <= y < height:
            img_ref[y, x] = 1.0

    # Difference
    img_diff = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            s = img_scan[y, x]
            r = img_ref[y, x]
            if s > 0 and r > 0:
                img_diff[y, x] = [0, 1, 0]  # green = both
            elif s > 0:
                img_diff[y, x] = [1, 0, 0]  # red = scanline only
            elif r > 0:
                img_diff[y, x] = [0, 0, 1]  # blue = reference only

    axes[0].imshow(img_scan, cmap='gray', origin='upper')
    axes[0].set_title(f"Scanline ({len(scanline_pixels)} px)")
    hull_closed = np.vstack([hull_2d, hull_2d[0]])
    axes[0].plot(hull_closed[:, 0], hull_closed[:, 1], 'r-', linewidth=1)

    axes[1].imshow(img_ref, cmap='gray', origin='upper')
    axes[1].set_title(f"Reference ({len(ref_pixels)} px)")
    axes[1].plot(hull_closed[:, 0], hull_closed[:, 1], 'r-', linewidth=1)

    axes[2].imshow(img_diff, origin='upper')
    axes[2].set_title("Diff (green=both, red=scan only, blue=ref only)")
    axes[2].plot(hull_closed[:, 0], hull_closed[:, 1], 'r-', linewidth=1)

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def test_basic_shapes():
    """Test scanline fill on known shapes."""
    W, H = 32, 32

    print("=== Test 1: Simple triangle ===")
    tri = np.array([[5.0, 5.0], [25.0, 5.0], [15.0, 28.0]])
    hull = convex_hull_2d(tri)
    scan = scanline_fill(hull, 0, W, 0, H)
    ref = reference_fill(hull, 0, W, 0, H)
    print(f"  Scanline: {len(scan)} pixels, Reference: {len(ref)} pixels")
    scan_set = set(scan)
    ref_set = set(ref)
    missed = ref_set - scan_set
    extra = scan_set - ref_set
    print(f"  Missed: {len(missed)}, Extra: {len(extra)}")
    fig = visualize_comparison(hull, scan, ref, W, H, "Triangle")
    fig.savefig("tests/scanline_triangle.png", dpi=100)
    plt.close()

    print("\n=== Test 2: Quadrilateral (tet projection) ===")
    quad = np.array([[3.0, 10.0], [15.0, 2.0], [28.0, 12.0], [12.0, 29.0]])
    hull = convex_hull_2d(quad)
    scan = scanline_fill(hull, 0, W, 0, H)
    ref = reference_fill(hull, 0, W, 0, H)
    print(f"  Scanline: {len(scan)} pixels, Reference: {len(ref)} pixels")
    scan_set = set(scan)
    ref_set = set(ref)
    missed = ref_set - scan_set
    extra = scan_set - ref_set
    print(f"  Missed: {len(missed)}, Extra: {len(extra)}")
    fig = visualize_comparison(hull, scan, ref, W, H, "Quadrilateral")
    fig.savefig("tests/scanline_quad.png", dpi=100)
    plt.close()

    print("\n=== Test 3: Thin sliver ===")
    sliver = np.array([[2.0, 15.0], [30.0, 14.0], [16.0, 17.0], [15.0, 16.0]])
    hull = convex_hull_2d(sliver)
    scan = scanline_fill(hull, 0, W, 0, H)
    ref = reference_fill(hull, 0, W, 0, H)
    print(f"  Scanline: {len(scan)} pixels, Reference: {len(ref)} pixels")
    scan_set = set(scan)
    ref_set = set(ref)
    missed = ref_set - scan_set
    extra = scan_set - ref_set
    print(f"  Missed: {len(missed)}, Extra: {len(extra)}")
    fig = visualize_comparison(hull, scan, ref, W, H, "Thin sliver")
    fig.savefig("tests/scanline_sliver.png", dpi=100)
    plt.close()

    print("\n=== Test 4: Nearly degenerate (3 collinear + 1 offset) ===")
    degen = np.array([[5.0, 15.0], [15.0, 15.0], [25.0, 15.2], [15.0, 20.0]])
    hull = convex_hull_2d(degen)
    scan = scanline_fill(hull, 0, W, 0, H)
    ref = reference_fill(hull, 0, W, 0, H)
    print(f"  Scanline: {len(scan)} pixels, Reference: {len(ref)} pixels")
    scan_set = set(scan)
    ref_set = set(ref)
    missed = ref_set - scan_set
    extra = scan_set - ref_set
    print(f"  Missed: {len(missed)}, Extra: {len(extra)}")
    fig = visualize_comparison(hull, scan, ref, W, H, "Nearly degenerate")
    fig.savefig("tests/scanline_degen.png", dpi=100)
    plt.close()


def test_projected_tets():
    """Test with actual projected tetrahedra."""
    W, H = 64, 64
    np.random.seed(42)

    cam_pos = np.array([0.0, 0.0, 2.0])
    vp = make_vp_matrix(cam_pos, fov=1.0, width=W, height=H)

    # Random points
    pts = np.random.randn(8, 3).astype(np.float32) * 0.3
    tri = Delaunay(pts)
    tets = tri.simplices

    print(f"\n=== Projected tets: {len(tets)} tets, {W}x{H} image ===")

    total_scan = 0
    total_ref = 0
    total_missed = 0
    total_extra = 0

    # Composite image
    img_all = np.zeros((H, W, 3), dtype=np.float32)

    for ti, tet in enumerate(tets):
        verts_3d = pts[tet]
        proj_2d = project_tet(verts_3d, vp, W, H)
        if proj_2d is None:
            print(f"  Tet {ti}: behind camera, skipping")
            continue

        hull = convex_hull_2d(proj_2d)
        if len(hull) < 3:
            print(f"  Tet {ti}: degenerate hull ({len(hull)} verts)")
            continue

        scan = scanline_fill(hull, 0, W, 0, H)
        ref = reference_fill(hull, 0, W, 0, H)

        scan_set = set(scan)
        ref_set = set(ref)
        missed = ref_set - scan_set
        extra = scan_set - ref_set

        total_scan += len(scan)
        total_ref += len(ref)
        total_missed += len(missed)
        total_extra += len(extra)

        # Color this tet
        color = np.random.rand(3) * 0.3 + 0.2
        for x, y in scan:
            if 0 <= x < W and 0 <= y < H:
                img_all[y, x] += color

        if missed or extra:
            print(f"  Tet {ti}: scan={len(scan)}, ref={len(ref)}, missed={len(missed)}, extra={len(extra)}")

    print(f"\n  TOTAL: scan={total_scan}, ref={total_ref}, missed={total_missed}, extra={total_extra}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    img_all = np.clip(img_all, 0, 1)
    ax.imshow(img_all, origin='upper')
    ax.set_title(f"All tets scanline-filled ({len(tets)} tets)")

    # Draw hull outlines
    for ti, tet in enumerate(tets):
        verts_3d = pts[tet]
        proj_2d = project_tet(verts_3d, vp, W, H)
        if proj_2d is None:
            continue
        hull = convex_hull_2d(proj_2d)
        if len(hull) >= 3:
            closed = np.vstack([hull, hull[0]])
            ax.plot(closed[:, 0], closed[:, 1], 'w-', linewidth=0.5, alpha=0.5)

    fig.savefig("tests/scanline_projected_tets.png", dpi=150)
    plt.close()


def test_tile_level_fill():
    """Test scanline fill at tile level (16x16 pixel tiles).

    This is what forward_compute.wgsl needs: count how many tiles a tet touches.
    """
    W, H = 256, 256
    TILE = 16
    tiles_x = W // TILE
    tiles_y = H // TILE

    np.random.seed(42)
    cam_pos = np.array([0.0, 0.0, 2.0])
    vp = make_vp_matrix(cam_pos, fov=1.0, width=W, height=H)

    pts = np.random.randn(8, 3).astype(np.float32) * 0.3
    tri = Delaunay(pts)
    tets = tri.simplices

    print(f"\n=== Tile-level fill: {len(tets)} tets, {W}x{H} image, {TILE}x{TILE} tiles ===")

    for ti, tet in enumerate(tets):
        verts_3d = pts[tet]
        proj_2d = project_tet(verts_3d, vp, W, H)
        if proj_2d is None:
            continue

        hull = convex_hull_2d(proj_2d)
        if len(hull) < 3:
            continue

        # Tile-level: scale hull to tile coords
        hull_tile = hull / TILE
        hull_tile_hull = convex_hull_2d(hull_tile)  # already convex, but re-sort

        # Scanline at tile granularity
        tile_pixels = scanline_fill(hull_tile_hull, 0, tiles_x, 0, tiles_y)

        # AABB count (what current shader does)
        pmin = np.floor(hull.min(axis=0) / TILE).astype(int)
        pmax = np.floor(hull.max(axis=0) / TILE).astype(int)
        pmin = np.clip(pmin, 0, [tiles_x - 1, tiles_y - 1])
        pmax = np.clip(pmax, 0, [tiles_x - 1, tiles_y - 1])
        aabb_count = (pmax[0] - pmin[0] + 1) * (pmax[1] - pmin[1] + 1)

        # Current hull overlap test count (reference)
        hull_overlap_count = 0
        for ty in range(pmin[1], pmax[1] + 1):
            for tx in range(pmin[0], pmax[0] + 1):
                rect_left = tx * TILE
                rect_top = ty * TILE
                rect_right = rect_left + TILE
                rect_bottom = rect_top + TILE
                # Check center in hull
                cx, cy = (rect_left + rect_right) / 2, (rect_top + rect_bottom) / 2
                path = Path(np.vstack([hull, hull[0]]))
                if path.contains_point((cx, cy)):
                    hull_overlap_count += 1
                    continue
                # Check any hull vertex in rect
                for v in hull:
                    if rect_left <= v[0] <= rect_right and rect_top <= v[1] <= rect_bottom:
                        hull_overlap_count += 1
                        break

        print(f"  Tet {ti}: scanline_tiles={len(tile_pixels)}, aabb={aabb_count}, hull_overlap={hull_overlap_count}")


def test_within_tile_fill():
    """Test pixel-level fill WITHIN a 16x16 tile.

    This is what the forward_tiled_compute.wgsl shader needs:
    given a tet assigned to a specific tile, find which of the 256 pixels
    in that tile are covered.
    """
    TILE = 16
    W, H = 256, 256

    np.random.seed(42)
    cam_pos = np.array([0.0, 0.0, 2.0])
    vp = make_vp_matrix(cam_pos, fov=1.0, width=W, height=H)

    pts = np.random.randn(8, 3).astype(np.float32) * 0.3
    tri = Delaunay(pts)
    tets = tri.simplices

    print(f"\n=== Within-tile pixel fill ===")

    # Pick a tet and a tile it covers
    for ti, tet in enumerate(tets):
        verts_3d = pts[tet]
        proj_2d = project_tet(verts_3d, vp, W, H)
        if proj_2d is None:
            continue

        hull = convex_hull_2d(proj_2d)
        if len(hull) < 3:
            continue

        # Find a tile this tet covers
        center = hull.mean(axis=0)
        tile_x = int(center[0] / TILE)
        tile_y = int(center[1] / TILE)

        # Pixel range for this tile
        px_min = tile_x * TILE
        py_min = tile_y * TILE
        px_max = px_min + TILE
        py_max = py_min + TILE

        # Transform hull to tile-local coords
        hull_local = hull.copy()
        hull_local[:, 0] -= px_min
        hull_local[:, 1] -= py_min

        # Scanline fill within the tile
        local_pixels = scanline_fill(hull_local, 0, TILE, 0, TILE)
        ref_pixels = reference_fill(hull_local, 0, TILE, 0, TILE)

        scan_set = set(local_pixels)
        ref_set = set(ref_pixels)
        missed = ref_set - scan_set
        extra = scan_set - ref_set

        print(f"  Tet {ti}, tile ({tile_x},{tile_y}): "
              f"scan={len(local_pixels)}, ref={len(ref_pixels)}, "
              f"missed={len(missed)}, extra={len(extra)}")

        if ti == 0:  # Visualize just the first one
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            img_scan = np.zeros((TILE, TILE), dtype=np.float32)
            for x, y in local_pixels:
                img_scan[y, x] = 1.0

            img_ref = np.zeros((TILE, TILE), dtype=np.float32)
            for x, y in ref_pixels:
                img_ref[y, x] = 1.0

            axes[0].imshow(img_scan, cmap='gray', origin='upper', vmin=0, vmax=1)
            axes[0].set_title(f"Scanline ({len(local_pixels)} px)")
            closed = np.vstack([hull_local, hull_local[0]])
            axes[0].plot(closed[:, 0], closed[:, 1], 'r-', linewidth=1)
            for v in hull_local:
                axes[0].plot(v[0], v[1], 'ro', markersize=4)

            axes[1].imshow(img_ref, cmap='gray', origin='upper', vmin=0, vmax=1)
            axes[1].set_title(f"Reference ({len(ref_pixels)} px)")
            axes[1].plot(closed[:, 0], closed[:, 1], 'r-', linewidth=1)
            for v in hull_local:
                axes[1].plot(v[0], v[1], 'ro', markersize=4)

            fig.suptitle(f"Tet {ti} in tile ({tile_x},{tile_y})")
            fig.savefig("tests/scanline_within_tile.png", dpi=150)
            plt.close()

        if ti >= 5:
            break


def scanline_fill_4pt(pts4, x_min, x_max, y_min, y_max):
    """Scanline fill for exactly 4 points — no ConvexHull needed.

    This is the WGSL-equivalent algorithm: intersect all 6 edges of the 4 points
    with each horizontal scanline, take min/max x. Works because interior-point
    edges only produce intersections WITHIN the convex hull boundary.

    pts4: (4, 2) array of 2D points.
    Returns list of (px, py) covered pixels.
    """
    # All 6 edges of 4 points
    EDGES = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

    pixels = []
    for y in range(y_min, y_max):
        yc = y + 0.5

        xl = 1e30
        xr = -1e30
        n_intersect = 0

        for ei, ej in EDGES:
            y0, y1 = pts4[ei, 1], pts4[ej, 1]
            x0, x1 = pts4[ei, 0], pts4[ej, 0]

            if (y0 <= yc and y1 > yc) or (y1 <= yc and y0 > yc):
                t = (yc - y0) / (y1 - y0)
                x_at_y = x0 + t * (x1 - x0)
                xl = min(xl, x_at_y)
                xr = max(xr, x_at_y)
                n_intersect += 1

        if n_intersect < 2:
            continue

        px_start = max(int(np.floor(xl)), x_min)
        px_end = min(int(np.floor(xr)), x_max - 1)

        for x in range(px_start, px_end + 1):
            xc = x + 0.5
            if xc >= xl and xc <= xr:
                pixels.append((x, y))

    return pixels


def test_4pt_vs_hull():
    """Compare 4-point scanline (no hull) vs ConvexHull-based scanline."""
    W, H = 64, 64
    np.random.seed(42)

    cam_pos = np.array([0.0, 0.0, 2.0])
    vp = make_vp_matrix(cam_pos, fov=1.0, width=W, height=H)

    pts = np.random.randn(12, 3).astype(np.float32) * 0.3
    tri = Delaunay(pts)
    tets = tri.simplices

    print(f"\n=== 4-point scanline vs ConvexHull scanline: {len(tets)} tets ===")

    total_match = 0
    total_mismatch = 0

    for ti, tet in enumerate(tets):
        verts_3d = pts[tet]
        proj_2d = project_tet(verts_3d, vp, W, H)
        if proj_2d is None:
            continue

        # ConvexHull-based
        hull = convex_hull_2d(proj_2d)
        if len(hull) < 3:
            continue
        scan_hull = set(scanline_fill(hull, 0, W, 0, H))

        # 4-point (no hull)
        scan_4pt = set(scanline_fill_4pt(proj_2d, 0, W, 0, H))

        # Reference
        ref = set(reference_fill(hull, 0, W, 0, H))

        hull_match = scan_hull == ref
        pt4_match = scan_4pt == ref

        if not pt4_match:
            missed = ref - scan_4pt
            extra = scan_4pt - ref
            print(f"  Tet {ti}: 4pt MISMATCH! hull_ok={hull_match}, "
                  f"4pt={len(scan_4pt)} ref={len(ref)} missed={len(missed)} extra={len(extra)}")
            total_mismatch += 1
        else:
            total_match += 1

    print(f"  Results: {total_match} match, {total_mismatch} mismatch")


def test_degenerate_hulls():
    """Test cases where the 4 projected points form a triangle (one interior)."""
    W, H = 32, 32

    print("\n=== Degenerate hulls (triangle from 4 points) ===")

    # Case 1: point D clearly inside triangle ABC
    pts = np.array([
        [5.0, 5.0],
        [25.0, 5.0],
        [15.0, 28.0],
        [15.0, 12.0],  # interior point
    ])
    hull = convex_hull_2d(pts)
    print(f"  Hull has {len(hull)} vertices (expected 3)")

    scan_hull = set(scanline_fill(hull, 0, W, 0, H))
    scan_4pt = set(scanline_fill_4pt(pts, 0, W, 0, H))
    ref = set(reference_fill(hull, 0, W, 0, H))

    print(f"  hull_scan={len(scan_hull)}, 4pt_scan={len(scan_4pt)}, ref={len(ref)}")
    print(f"  hull matches ref: {scan_hull == ref}")
    print(f"  4pt matches ref: {scan_4pt == ref}")

    if scan_4pt != ref:
        missed = ref - scan_4pt
        extra = scan_4pt - ref
        print(f"  4pt missed={len(missed)}, extra={len(extra)}")

    # Case 2: point on edge of triangle
    pts2 = np.array([
        [5.0, 5.0],
        [25.0, 5.0],
        [15.0, 28.0],
        [15.0, 5.0],  # on edge AB
    ])
    hull2 = convex_hull_2d(pts2)
    print(f"\n  Edge case: hull has {len(hull2)} vertices")
    scan_4pt2 = set(scanline_fill_4pt(pts2, 0, W, 0, H))
    ref2 = set(reference_fill(hull2, 0, W, 0, H))
    print(f"  4pt={len(scan_4pt2)}, ref={len(ref2)}, match={scan_4pt2 == ref2}")

    # Case 3: two coincident points
    pts3 = np.array([
        [5.0, 5.0],
        [25.0, 5.0],
        [15.0, 28.0],
        [15.0, 28.0],  # same as vertex 2
    ])
    hull3 = convex_hull_2d(pts3[:3])  # only 3 unique
    scan_4pt3 = set(scanline_fill_4pt(pts3, 0, W, 0, H))
    ref3 = set(reference_fill(hull3, 0, W, 0, H))
    print(f"\n  Coincident case: 4pt={len(scan_4pt3)}, ref={len(ref3)}, match={scan_4pt3 == ref3}")


def test_tile_counting_accuracy():
    """For tile allocation, we need tiles_touched >= actual pairs written.

    Compare three approaches:
    1. AABB (current forward_compute.wgsl) - very conservative
    2. Scanline at tile granularity (tests tile centers) - might undercount
    3. Exact: pixel-level scanline, then count unique tiles - ground truth
    """
    W, H = 256, 256
    TILE = 16
    tiles_x = W // TILE
    tiles_y = H // TILE

    np.random.seed(42)
    cam_pos = np.array([0.0, 0.0, 2.0])
    vp = make_vp_matrix(cam_pos, fov=1.0, width=W, height=H)

    pts = np.random.randn(12, 3).astype(np.float32) * 0.3
    tri = Delaunay(pts)
    tets = tri.simplices

    print(f"\n=== Tile counting accuracy: {len(tets)} tets ===")
    print(f"  {'Tet':>4} {'AABB':>6} {'ScanTile':>8} {'Exact':>6} {'ScanOK':>7}")

    for ti, tet in enumerate(tets):
        verts_3d = pts[tet]
        proj_2d = project_tet(verts_3d, vp, W, H)
        if proj_2d is None:
            continue

        hull = convex_hull_2d(proj_2d)
        if len(hull) < 3:
            continue

        # 1. AABB count
        pmin = np.floor(hull.min(axis=0) / TILE).astype(int)
        pmax = np.floor(hull.max(axis=0) / TILE).astype(int)
        pmin = np.clip(pmin, 0, [tiles_x - 1, tiles_y - 1])
        pmax = np.clip(pmax, 0, [tiles_x - 1, tiles_y - 1])
        aabb_count = (pmax[0] - pmin[0] + 1) * (pmax[1] - pmin[1] + 1)

        # 2. Scanline at tile granularity
        hull_tile = hull / TILE
        hull_tile = convex_hull_2d(hull_tile)
        tile_scan = scanline_fill(hull_tile, 0, tiles_x, 0, tiles_y)
        tile_scan_count = len(tile_scan)

        # 3. Exact: pixel scanline → unique tiles
        pixel_scan = scanline_fill_4pt(proj_2d, 0, W, 0, H)
        exact_tiles = set()
        for px, py in pixel_scan:
            tx = px // TILE
            ty = py // TILE
            exact_tiles.add((tx, ty))
        exact_count = len(exact_tiles)

        scan_ok = tile_scan_count >= exact_count
        print(f"  {ti:4d} {aabb_count:6d} {tile_scan_count:8d} {exact_count:6d} {'OK' if scan_ok else 'UNDER!':>7}")

        if not scan_ok:
            # Find which tiles are missed
            tile_scan_set = set(tile_scan)
            for tx, ty in exact_tiles:
                if (tx, ty) not in tile_scan_set:
                    print(f"         Missing tile ({tx},{ty})")


if __name__ == "__main__":
    test_basic_shapes()
    test_projected_tets()
    test_4pt_vs_hull()
    test_degenerate_hulls()
    test_tile_counting_accuracy()
    test_tile_level_fill()
    test_within_tile_fill()
    print("\nDone! Check tests/scanline_*.png")
