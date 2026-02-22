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
    float aux[AUX_DIM]; // <--- Added
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
inline CtrlPt integrate_tetra_segment(
    in SplatTetrahedra_interp g, 
    in Ray ray,
    in float enter,
    in float exit,
    in float3 enter_norm,
    in float3 exit_norm,
)
{
    float dt = max(exit - enter, 0.0f);

    float3 v_enter = enter * ray.d + ray.o - g.tet.verts[0];
    float3 v_exit = exit * ray.d + ray.o - g.tet.verts[0];
    float3 c_enter = max(dot(g.grd, v_enter) + g.base_color, 0.f);
    float3 c_exit = max(dot(g.grd, v_exit) + g.base_color, 0.f);
    // float3 c_enter = dot(g.grd, v_enter) + g.base_color;
    // float3 c_exit = dot(g.grd, v_exit) + g.base_color;

    // float alpha = safe_clip(1.0f - safe_exp(-g.density * dt), 0.f, 1.f);
    float alpha = max(-safe_expm1(-g.density * dt), 0.f);
    float3 final_color = compute_integral(c_exit, c_enter, dt * g.density);

    float3 rendered_norm = alpha * l2_normalize(enter_norm);
    float depth = safe_div(compute_integral1D(enter, exit, dt * g.density), ray.ortho);

    float4 rgba = { final_color.x,
                    final_color.y,
                    final_color.z,
                    g.density * dt };

    CtrlPt result;
    result.rgba = { final_color.x, final_color.y, final_color.z, g.density * dt };
    result.dist = { enter, exit };
    result.density = g.density;
    result.xyzd = float4(rendered_norm, depth);

    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        result.aux[i] = g.aux[i] * detach(alpha);
    }
    
    return result;
}

[Differentiable]
inline bool evaluate_tetra_interp(
    in SplatTetrahedra_interp g, 
    in Ray ray,
    no_diff in float t_min,
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
        ctrl_pt = { (float4)(0.f, 0.f, 0.f, 0.f), {0.f, 0.f}, g.density };
        return false;
    } else {
        ctrl_pt = integrate_tetra_segment(g, ray, enter, exit, enter_norm, exit_norm);
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
