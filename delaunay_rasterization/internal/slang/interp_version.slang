static const uint AUX_DIM = PYTHON_AUX_DIM;

import intersect;
import utils;
import safe_math;
import camera;

struct CtrlPt: IDifferentiable {
    float4 rgba;
    float2 dist;
    float density;
    float4 xyzd;
    float aux[AUX_DIM];
    float3 entry_contrib; // unused (reserved)
    float3 exit_contrib;  // exit face normal n_exit (for end-of-ray correction)

    static CtrlPt zero(float density) {
        CtrlPt cp;
        cp.rgba = float4(0);
        cp.dist = float2(0);
        cp.density = density;
        cp.xyzd = float4(0);
        cp.entry_contrib = float3(0);
        cp.exit_contrib = float3(0);
        [ForceUnroll]
        for (int i = 0; i < AUX_DIM; ++i) cp.aux[i] = 0;
        return cp;
    }
};

struct SplatTetrahedra_interp : IDifferentiable
{
    uint4 vtet;
    Tetrahedra tet;
    float3 base_color;
    float3 grd;
    float density;
    float aux[AUX_DIM]; // <--- Added
};

void print_verts(float4x3 verts) {
    printf("%f, %f, %f, %f\n", verts[0].x, verts[1].x, verts[2].x, verts[3].x);
}
void print_vec(float3 vec) {
    printf("(%f, %f, %f),", vec.x, vec.y, vec.z);
}

[Differentiable]
inline float phi(float x) {
    if (abs(x) < 1e-6) {
        return 1.0f - x / 2.0f;
    }
    // return (1.0f - exp(-x)) / x;
    return -(safe_expm1(-x)) / x;
}

[Differentiable]
inline float compute_integral1D(float c0, float c1, float ddt) {
    float alpha = safe_exp(-ddt);
    float phi_val = phi(ddt);
    float w0 = phi_val - alpha;
    float w1 = 1.0f - phi_val;
    return w0 * c0 + w1 * c1;
}
[Differentiable]
inline float3 compute_integral(float3 c0, float3 c1, float ddt) {
    float alpha = safe_exp(-ddt);
    float phi_val = phi(ddt);
    float w0 = phi_val - alpha;
    float w1 = 1.0f - phi_val;
    return w0 * c0 + w1 * c1;
}

[Differentiable]
float3[4] load_vert_normals(no_diff uint4 vtet, DiffTensorView vertex_normals) {
    float3[4] vn;
    vn[0] = read_t3_float3(vtet.x, vertex_normals);
    vn[1] = read_t3_float3(vtet.y, vertex_normals);
    vn[2] = read_t3_float3(vtet.z, vertex_normals);
    vn[3] = read_t3_float3(vtet.w, vertex_normals);
    return vn;
}

float3[4] load_vert_normals(uint4 vtet, TensorView<float> vertex_normals) {
    float3[4] vn;
    vn[0] = read_t3_float3(vtet.x, vertex_normals);
    vn[1] = read_t3_float3(vtet.y, vertex_normals);
    vn[2] = read_t3_float3(vtet.z, vertex_normals);
    vn[3] = read_t3_float3(vtet.w, vertex_normals);
    return vn;
}

[Differentiable]
inline CtrlPt integrate_tetra_segment(
    in SplatTetrahedra_interp g,
    in Ray ray,
    in float enter,
    in float exit,
    in float3 enter_norm,
    in float3 exit_norm,
    in float3[4] vert_normals,
)
{
    float dt = max(exit - enter, 0.0f);

    // v_enter/v_exit = offset from vertex 0, reused for both color and normal bary coords
    float3 v_enter = enter * ray.d + ray.o - g.tet.verts[0];
    float3 v_exit = exit * ray.d + ray.o - g.tet.verts[0];
    float3 c_enter = max(dot(g.grd, v_enter) + g.base_color, 0.f);
    float3 c_exit = max(dot(g.grd, v_exit) + g.base_color, 0.f);

    float alpha = max(-safe_expm1(-g.density * dt), 0.f);
    float3 final_color = compute_integral(c_exit, c_enter, dt * g.density);

    // Barycentric interpolation of vertex normals via Cramer's rule.
    // Compute cofactors once; reuse for both enter and exit points.
    float3 e1 = g.tet.verts[1] - g.tet.verts[0];
    float3 e2 = g.tet.verts[2] - g.tet.verts[0];
    float3 e3 = g.tet.verts[3] - g.tet.verts[0];
    float3 c23 = cross(e2, e3);
    float3 c31 = cross(e3, e1);
    float3 c12 = cross(e1, e2);
    float inv_det = safe_div(1.0f, dot(e1, c23));

    // Enter point bary coords (reuse v_enter = p_enter - v0)
    float b1_en = dot(v_enter, c23) * inv_det;
    float b2_en = dot(v_enter, c31) * inv_det;
    float b3_en = dot(v_enter, c12) * inv_det;
    float b0_en = 1.0 - b1_en - b2_en - b3_en;
    float3 n_enter = b0_en * vert_normals[0] + b1_en * vert_normals[1] + b2_en * vert_normals[2] + b3_en * vert_normals[3];

    // Exit point bary coords (reuse v_exit = p_exit - v0)
    float b1_ex = dot(v_exit, c23) * inv_det;
    float b2_ex = dot(v_exit, c31) * inv_det;
    float b3_ex = dot(v_exit, c12) * inv_det;
    float b0_ex = 1.0 - b1_ex - b2_ex - b3_ex;
    float3 n_exit = b0_ex * vert_normals[0] + b1_ex * vert_normals[1] + b2_ex * vert_normals[2] + b3_ex * vert_normals[3];

    // Flip interpolated normals to face camera (same convention as old
    // density-diff approach).  Without this, front-facing and back-facing
    // tet contributions cancel during alpha blending → black normals.
    // The sign is no_diff so gradient correctly pushes the rendered normal
    // toward the target; the underlying parameter stores the axis.

    float3 normal_contrib = compute_integral(n_exit, n_enter, dt * g.density);
    float depth = safe_div(compute_integral1D(exit, enter, dt * g.density), ray.ortho);

    CtrlPt result;
    result.rgba = { final_color.x, final_color.y, final_color.z, g.density * dt };
    result.dist = { enter, exit };
    result.density = g.density;
    result.xyzd = float4(normal_contrib, depth);

    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        result.aux[i] = g.aux[i] * alpha;
    }

    result.entry_contrib = float3(0);
    result.exit_contrib = float3(0);

    return result;
}

[Differentiable]
inline bool evaluate_tetra_interp(
    in SplatTetrahedra_interp g,
    in Ray ray,
    no_diff in float t_min,
    in float3[4] vert_normals,
    out CtrlPt ctrl_pt)
{
    float2 dist;

    float4x3 verts = float4x3(g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]);

    float3 enter_norm, exit_norm;
    bool hit = ray_tetrahedron_intersect_fused(
        ray.o, ray.d, verts, dist, enter_norm, exit_norm);

    float enter = dist.x;
    float exit = max(dist.y, enter);

    if (!hit) {
        ctrl_pt = no_diff CtrlPt.zero(g.density);
        return false;
    } else {
        ctrl_pt = integrate_tetra_segment(g, ray, enter, exit, enter_norm, exit_norm, vert_normals);
        return true;
    }
}

[Differentiable]
inline SplatTetrahedra_interp load_tet_alphablend_interp(
    int32_t        g_idx,
    DiffTensorView vertices,
    TensorView<int32_t> indices,
    DiffTensorView cell_values)
{
    no_diff uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);

    float density = max(float(cell_values[g_idx, 0]), 0.f);
    float3 base_color = {
        float(cell_values[g_idx, 1]),
        float(cell_values[g_idx, 2]),
        float(cell_values[g_idx, 3]),
    };
    float3 grd = {
        float(cell_values[g_idx, 4]),
        float(cell_values[g_idx, 5]),
        float(cell_values[g_idx, 6]),
    };

    SplatTetrahedra_interp result = {
        virtual_tet, tet, base_color, grd, density
    };

    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        // 1 (density) + 3 (color) + 3 (grd) = 7
        result.aux[i] = float(cell_values[g_idx, 7 + i]);
    }

    return result;
}

inline SplatTetrahedra_interp load_tet_alphablend_interp(
    int32_t        g_idx,
    TensorView<float> vertices,
    TensorView<int32_t> indices,
    TensorView<float>  cell_values)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);

    float density = max(float(cell_values[g_idx, 0]), 0.f);
    SplatTetrahedra_interp result;
    result.vtet = virtual_tet;
    result.tet       = tet;
    result.density   = density;
    result.base_color = {
        float(cell_values[g_idx, 1]),
        float(cell_values[g_idx, 2]),
        float(cell_values[g_idx, 3]),
    };
    result.grd = {
        float(cell_values[g_idx, 4]),
        float(cell_values[g_idx, 5]),
        float(cell_values[g_idx, 6]),
    };
    for (int i = 0; i < AUX_DIM; ++i) {
        // 1 (density) + 3 (color) + 3 (grd) = 7
        result.aux[i] = float(cell_values[g_idx, 7 + i]);
    }
    return result;
}
