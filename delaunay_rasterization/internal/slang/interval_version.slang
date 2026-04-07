import utils;
import safe_math;
import camera;
import interp_version;

// Per-fan-vertex data layout (10 floats):
//   [0,1]   = pixel_xy
//   [2]     = z_front (camera-space depth of front surface)
//   [3]     = z_back  (camera-space depth of back surface)
//   [4,5,6] = normal_front (interpolated vertex normal at front)
//   [7,8,9] = normal_back  (interpolated vertex normal at back)

struct IntervalVertex : IDifferentiable {
    float2 pixel_xy;
    float z_front;
    float z_back;
    float3 normal_front;
    float3 normal_back;
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
    return v;
}

IntervalVertex load_interval_vertex(uint tet_idx, uint vert_idx, TensorView<float> interval_verts) {
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
    return v;
}

// Per-tet data layout (7 + AUX_DIM floats):
//   [0]     = density
//   [1,2,3] = base_color rgb
//   [4,5,6] = gradient (grd)
//   [7..]   = aux attributes

struct IntervalTetData : IDifferentiable {
    float density;
    float3 base_color;
    float3 grd;
    float aux[AUX_DIM];
};

[Differentiable]
IntervalTetData load_interval_tet_data(uint tet_idx, DiffTensorView interval_tet_data) {
    IntervalTetData d;
    d.density    = max(float(interval_tet_data[uint2(tet_idx, 0)]), 0.f);
    d.base_color = float3(
        interval_tet_data[uint2(tet_idx, 1)],
        interval_tet_data[uint2(tet_idx, 2)],
        interval_tet_data[uint2(tet_idx, 3)]);
    d.grd = float3(
        interval_tet_data[uint2(tet_idx, 4)],
        interval_tet_data[uint2(tet_idx, 5)],
        interval_tet_data[uint2(tet_idx, 6)]);
    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        d.aux[i] = float(interval_tet_data[uint2(tet_idx, 7 + i)]);
    }
    return d;
}

IntervalTetData load_interval_tet_data(uint tet_idx, TensorView<float> interval_tet_data) {
    IntervalTetData d;
    d.density    = max(float(interval_tet_data[uint2(tet_idx, 0)]), 0.f);
    d.base_color = float3(
        interval_tet_data[uint2(tet_idx, 1)],
        interval_tet_data[uint2(tet_idx, 2)],
        interval_tet_data[uint2(tet_idx, 3)]);
    d.grd = float3(
        interval_tet_data[uint2(tet_idx, 4)],
        interval_tet_data[uint2(tet_idx, 5)],
        interval_tet_data[uint2(tet_idx, 6)]);
    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        d.aux[i] = float(interval_tet_data[uint2(tet_idx, 7 + i)]);
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

// Evaluate a pixel against the interval representation of a tet.
// Fan triangles are formed from center vertex (index depends on silhouette case)
// and silhouette vertices. The fan covers the tet's screen-space footprint.
//
// Vertex layout in interval_verts[tet_idx, 0..4]:
//   For 3-triangle fan (Case 1): center=0, silhouette=1,2,3
//     Triangles: (0,1,2), (0,2,3), (0,3,1)
//   For 4-triangle fan (Case 2): center_a=0, center_b=1, silhouette=2,3,4  (but we use 4 verts for the quad)
//     Actually: center=0, silhouette=1,2,3,4
//     Triangles: (0,1,2), (0,2,3), (0,3,4), (0,4,1)
[Differentiable]
bool evaluate_tetra_interval(
    no_diff float2 pixel_pos,
    no_diff uint tet_idx,
    no_diff uint num_sil_verts,
    DiffTensorView interval_verts,
    IntervalTetData tet_data,
    no_diff float ray_scale,
    out CtrlPt ctrl_pt)
{
    ctrl_pt = no_diff CtrlPt.zero(tet_data.density);

    if (num_sil_verts == 0) return false;

    // Load center vertex (always index 0)
    IntervalVertex center = load_interval_vertex(tet_idx, 0, interval_verts);

    // Try each fan triangle
    [ForceUnroll]
    for (uint tri = 0; tri < 4; ++tri) {
        if (tri >= num_sil_verts) break;

        uint idx_a = 1 + tri;
        uint idx_b = 1 + ((tri + 1) % num_sil_verts);

        IntervalVertex va = load_interval_vertex(tet_idx, idx_a, interval_verts);
        IntervalVertex vb = load_interval_vertex(tet_idx, idx_b, interval_verts);

        float b0, b1, b2;
        if (point_in_triangle_2d(pixel_pos, center.pixel_xy, va.pixel_xy, vb.pixel_xy, b0, b1, b2)) {
            // Barycentric interpolation of z_front and z_back
            float z_front = b0 * center.z_front + b1 * va.z_front + b2 * vb.z_front;
            float z_back  = b0 * center.z_back  + b1 * va.z_back  + b2 * vb.z_back;

            // Interpolate normals
            float3 n_front = b0 * center.normal_front + b1 * va.normal_front + b2 * vb.normal_front;
            float3 n_back  = b0 * center.normal_back  + b1 * va.normal_back  + b2 * vb.normal_back;

            // Ensure z_front <= z_back
            float zf = min(z_front, z_back);
            float zb = max(z_front, z_back);

            float dt = max((zb - zf) * ray_scale, 0.0f);
            float alpha = max(-safe_expm1(-tet_data.density * dt), 0.f);

            // For color: use base_color directly (gradient is spatial,
            // but we don't have a world-space offset here, so use base_color)
            float3 c0 = max(tet_data.base_color, 0.f);
            float3 final_color = c0 * alpha;

            // Normal contribution
            float3 normal_contrib = compute_integral(n_back, n_front, dt * tet_data.density);

            // Depth: average of front and back, scaled by ray_scale
            float depth = safe_div(compute_integral1D(zb * ray_scale, zf * ray_scale, dt * tet_data.density), 1.0f);

            ctrl_pt.rgba = float4(final_color.x, final_color.y, final_color.z, tet_data.density * dt);
            ctrl_pt.dist = float2(zf * ray_scale, zb * ray_scale);
            ctrl_pt.density = tet_data.density;
            ctrl_pt.xyzd = float4(normal_contrib, depth);

            [ForceUnroll]
            for (int i = 0; i < AUX_DIM; ++i) {
                ctrl_pt.aux[i] = tet_data.aux[i] * alpha;
            }

            return true;
        }
    }

    return false;
}
