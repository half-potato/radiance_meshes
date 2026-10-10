import utils;
import safe_math;
import camera;
import interp_version;

// Per-fan-vertex data layout (16 floats):
//   [0,1]    = pixel_xy
//   [2]      = z_front (camera-space depth of front surface)
//   [3]      = z_back  (camera-space depth of back surface)
//   [4,5,6]  = normal_front (interpolated vertex normal at front)
//   [7,8,9]  = normal_back  (interpolated vertex normal at back)
//   [10..12] = color_front (c(p) = dot(grd, p - v0) + base_color at front)
//   [13..15] = color_back

static const uint INTERVAL_VERT_DIM = 16;

struct IntervalVertex : IDifferentiable {
    float2 pixel_xy;
    float z_front;
    float z_back;
    float3 normal_front;
    float3 normal_back;
    float3 color_front;
    float3 color_back;
};

[Differentiable]
IntervalVertex load_interval_vertex(uint tet_idx, uint vert_idx, DiffTensorView interval_verts) {
    IntervalVertex v;
    v.pixel_xy = float2(
        interval_verts[uint3(tet_idx, vert_idx, 0)],
        interval_verts[uint3(tet_idx, vert_idx, 1)]);
    v.z_front = interval_verts[uint3(tet_idx, vert_idx, 2)];
    v.z_back  = interval_verts[uint3(tet_idx, vert_idx, 3)];
    v.normal_front = float3(
        interval_verts[uint3(tet_idx, vert_idx, 4)],
        interval_verts[uint3(tet_idx, vert_idx, 5)],
        interval_verts[uint3(tet_idx, vert_idx, 6)]);
    v.normal_back = float3(
        interval_verts[uint3(tet_idx, vert_idx, 7)],
        interval_verts[uint3(tet_idx, vert_idx, 8)],
        interval_verts[uint3(tet_idx, vert_idx, 9)]);
    v.color_front = float3(
        interval_verts[uint3(tet_idx, vert_idx, 10)],
        interval_verts[uint3(tet_idx, vert_idx, 11)],
        interval_verts[uint3(tet_idx, vert_idx, 12)]);
    v.color_back = float3(
        interval_verts[uint3(tet_idx, vert_idx, 13)],
        interval_verts[uint3(tet_idx, vert_idx, 14)],
        interval_verts[uint3(tet_idx, vert_idx, 15)]);
    return v;
}

// Per-tet data layout (1 + AUX_DIM floats):
//   [0]   = density
//   [1..] = aux attributes
// (base color / gradient now live in the fan vertices as baked colors)

struct IntervalTetData : IDifferentiable {
    float density;
    float aux[AUX_DIM];
};

[Differentiable]
IntervalTetData load_interval_tet_data(uint tet_idx, DiffTensorView interval_tet_data) {
    IntervalTetData d;
    d.density = max(float(interval_tet_data[uint2(tet_idx, 0)]), 0.f);
    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        d.aux[i] = float(interval_tet_data[uint2(tet_idx, 1 + i)]);
    }
    return d;
}

// 2D point-in-triangle test. Returns true if p is inside triangle (v0,v1,v2)
// and outputs barycentric coordinates (b0,b1,b2) where p = b0*v0 + b1*v1 + b2*v2.
[Differentiable]
bool point_in_triangle_2d(
    float2 p, float2 v0, float2 v1, float2 v2,
    out float b0, out float b1, out float b2)
{
    float2 e0 = v1 - v0;
    float2 e1 = v2 - v0;
    float2 v = p - v0;

    float dot00 = dot(e0, e0);
    float dot01 = dot(e0, e1);
    float dot02 = dot(e0, v);
    float dot11 = dot(e1, e1);
    float dot12 = dot(e1, v);

    float denom = dot00 * dot11 - dot01 * dot01;
    if (abs(denom) < 1e-10f) {
        b0 = 0; b1 = 0; b2 = 0;
        return false;
    }

    float inv_denom = 1.0f / denom;
    b1 = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    b2 = (dot00 * dot12 - dot01 * dot02) * inv_denom;
    b0 = 1.0f - b1 - b2;

    // Epsilon tolerance for edge pixels
    float eps = -1e-4f;
    return (b0 >= eps && b1 >= eps && b2 >= eps);
}

// Non-differentiable hit search over the fan triangles. Only reads the
// 2-float pixel_xy of each fan vertex, so the forward pass doesn't pull all
// 16 floats per candidate triangle and — more importantly — the autodiff
// backward only scatters gradient atomics for the single hit triangle
// instead of every candidate.
int find_hit_triangle(
    float2 pixel_pos,
    uint tet_idx,
    uint num_sil_verts,
    DiffTensorView interval_verts)
{
    float2 center_xy = float2(
        interval_verts[uint3(tet_idx, 0, 0)],
        interval_verts[uint3(tet_idx, 0, 1)]);

    for (uint tri = 0; tri < 4; ++tri) {
        if (tri >= num_sil_verts) break;

        uint idx_a = 1 + tri;
        uint idx_b = 1 + ((tri + 1) % num_sil_verts);
        float2 a_xy = float2(
            interval_verts[uint3(tet_idx, idx_a, 0)],
            interval_verts[uint3(tet_idx, idx_a, 1)]);
        float2 b_xy = float2(
            interval_verts[uint3(tet_idx, idx_b, 0)],
            interval_verts[uint3(tet_idx, idx_b, 1)]);

        float b0, b1, b2;
        if (point_in_triangle_2d(pixel_pos, center_xy, a_xy, b_xy, b0, b1, b2)) {
            return int(tri);
        }
    }
    return -1;
}

// Core interval evaluation from already-loaded fan vertices. Pure function
// of the three IntervalVertex structs, so the backward pass can hand-manage
// the vertex differentials (wave-reduced scatter) instead of letting
// autodiff issue per-thread gradient atomics through the tensor loads.
//
// ray_scale converts a camera-space z extent into distance along the pixel
// ray: ray_scale = 1/cos(angle to principal axis) = sqrt(1 + dx^2 + dy^2).
// t_offset (= cam.min_t) shifts ray distances to match the interp renderer's
// parameterization, whose rays start at origin + min_t * dir.
[Differentiable]
bool evaluate_interval_verts(
    no_diff float2 pixel_pos,
    IntervalVertex center,
    IntervalVertex va,
    IntervalVertex vb,
    IntervalTetData tet_data,
    no_diff float ray_scale,
    no_diff float t_offset,
    out CtrlPt ctrl_pt)
{
    ctrl_pt = no_diff CtrlPt.zero(tet_data.density);

    // Differentiable barycentric coordinates of the pixel in the hit
    // triangle (gradient w.r.t. the fan vertex pixel positions moves the
    // silhouette). The inside test already passed in find_hit_triangle.
    float b0, b1, b2;
    if (!point_in_triangle_2d(pixel_pos, center.pixel_xy, va.pixel_xy, vb.pixel_xy, b0, b1, b2))
        return false;

    // Barycentric interpolation of z_front and z_back
    float z_front = b0 * center.z_front + b1 * va.z_front + b2 * vb.z_front;
    float z_back  = b0 * center.z_back  + b1 * va.z_back  + b2 * vb.z_back;

    // Interpolate normals and colors
    float3 n_front = b0 * center.normal_front + b1 * va.normal_front + b2 * vb.normal_front;
    float3 n_back  = b0 * center.normal_back  + b1 * va.normal_back  + b2 * vb.normal_back;
    float3 c_front = b0 * center.color_front + b1 * va.color_front + b2 * vb.color_front;
    float3 c_back  = b0 * center.color_back  + b1 * va.color_back  + b2 * vb.color_back;

    // Ensure z_front <= z_back
    float zf = min(z_front, z_back);
    float zb = max(z_front, z_back);

    // Ray-distance parameterization (matches interp's t along the ray)
    float t_enter = zf * ray_scale - t_offset;
    float t_exit  = zb * ray_scale - t_offset;
    float dt = max(t_exit - t_enter, 0.0f);
    float ddt = dt * tet_data.density;
    float alpha = max(-safe_expm1(-ddt), 0.f);

    // Density-weighted color integral, matching interp's
    // compute_integral(c_exit, c_enter, ...) with pointwise clamping.
    float3 c_enter = max(c_front, 0.f);
    float3 c_exit = max(c_back, 0.f);
    float3 final_color = compute_integral(c_exit, c_enter, ddt);

    // Normal contribution (front = enter, back = exit)
    float3 normal_contrib = compute_integral(n_back, n_front, ddt);

    // Expected ray distance (interp divides by ray.ortho ~ 1)
    float depth = compute_integral1D(t_exit, t_enter, ddt);

    ctrl_pt.rgba = float4(final_color.x, final_color.y, final_color.z, ddt);
    ctrl_pt.dist = float2(t_enter, t_exit);
    ctrl_pt.density = tet_data.density;
    ctrl_pt.xyzd = float4(normal_contrib, depth);

    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        ctrl_pt.aux[i] = tet_data.aux[i] * alpha;
    }

    return true;
}

// Evaluate a pixel against the interval representation of a tet.
// Fan triangles are formed from the center vertex (index 0) and the
// silhouette vertices (indices 1..num_sil_verts).
//   3-vert fan (Case 1): triangles (0,1,2), (0,2,3), (0,3,1)
//   4-vert fan (Case 2): triangles (0,1,2), (0,2,3), (0,3,4), (0,4,1)
[Differentiable]
bool evaluate_tetra_interval(
    no_diff float2 pixel_pos,
    no_diff uint tet_idx,
    no_diff uint num_sil_verts,
    DiffTensorView interval_verts,
    IntervalTetData tet_data,
    no_diff float ray_scale,
    no_diff float t_offset,
    out CtrlPt ctrl_pt)
{
    ctrl_pt = no_diff CtrlPt.zero(tet_data.density);

    if (num_sil_verts == 0) return false;

    int hit = no_diff find_hit_triangle(pixel_pos, tet_idx, num_sil_verts, interval_verts);
    if (hit < 0) return false;

    uint idx_a = 1 + uint(hit);
    uint idx_b = 1 + ((uint(hit) + 1) % num_sil_verts);

    IntervalVertex center = load_interval_vertex(tet_idx, 0, interval_verts);
    IntervalVertex va = load_interval_vertex(tet_idx, idx_a, interval_verts);
    IntervalVertex vb = load_interval_vertex(tet_idx, idx_b, interval_verts);

    return evaluate_interval_verts(pixel_pos, center, va, vb, tet_data,
                                   ray_scale, t_offset, ctrl_pt);
}

// Note: a wave-reduced gradient scatter (WaveActiveSum across the warp, one
// atomic per warp; gotcha: pick the writer lane with thread_rank % 32 == 0,
// not WaveIsFirstLane, whose subgroup size may differ from the warp size)
// was tried for the backward pass and measured ~2.3x SLOWER than plain
// per-thread gradient atomics here: reducing 5 fan verts x 16 floats costs
// ~90 wave reductions per contribution and the register pressure spills.
// It only pays off for small gradient payloads (~20 floats). Keep the simple
// autodiff scatter in alphablend_shader_interval.slang.
