import intersect;
import utils;
import safe_math;
import camera;

struct SplatTetrahedra_interp : IDifferentiable
{
    uint4 vtet;
    Tetrahedra tet;
    float3 base_color;
    float3 grd;
    float density;
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
    in float3 ray_o,
    in float3 ray_d,
    in float enter,
    in float exit,
)
{
    float dt = max(exit - enter, 0.0f);

    float3 v_enter = enter * ray_d + ray_o - g.tet.verts[0];
    float3 v_exit = exit * ray_d + ray_o - g.tet.verts[0];
    // float3 c_enter = max(dot(g.grd, v_enter) + g.base_color, 0.f);
    // float3 c_exit = max(dot(g.grd, v_exit) + g.base_color, 0.f);
    float3 c_enter = dot(g.grd, v_enter) + g.base_color;
    float3 c_exit = dot(g.grd, v_exit) + g.base_color;

    // float alpha = safe_clip(1.0f - safe_exp(-g.density * dt), 0.f, 1.f);
    float alpha = max(-safe_expm1(-g.density * dt), 0.f);
    float3 final_color = compute_integral(c_exit, c_enter, dt*g.density);

    float4 rgba = {final_color.x,
                    final_color.y,
                    final_color.z,
                    g.density * dt};
    return { rgba, {enter, exit}, g.density };
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
        ctrl_pt = integrate_tetra_segment(g, ray.o, ray.d, enter, exit);
        return true;
    }
}

[Differentiable]
inline SplatTetrahedra_interp load_tet_alphablend_interp(
    int32_t        g_idx,
    DiffTensorView vertices,
    TensorView<int32_t> indices,
    DiffTensorView tet_density)
{
    no_diff uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);

    float density = max(float(tet_density[g_idx, 0]), 0.f);
    float3 base_color = {
        float(tet_density[g_idx, 1]),
        float(tet_density[g_idx, 2]),
        float(tet_density[g_idx, 3]),
    };
    float3 grd = {
        float(tet_density[g_idx, 4]),
        float(tet_density[g_idx, 5]),
        float(tet_density[g_idx, 6]),
    };

    SplatTetrahedra_interp result = {
        virtual_tet, tet, base_color, grd, density
    };
    return result;
}

inline SplatTetrahedra_interp load_tet_alphablend_interp(
    int32_t        g_idx,
    TensorView<float> vertices,
    TensorView<int32_t> indices,
    TensorView<float>  tet_density)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);

    float density = max(float(tet_density[g_idx, 0]), 0.f);
    SplatTetrahedra_interp result;
    result.vtet = virtual_tet;
    result.tet       = tet;
    result.density   = density;
    result.base_color = {
        float(tet_density[g_idx, 1]),
        float(tet_density[g_idx, 2]),
        float(tet_density[g_idx, 3]),
    };
    result.grd = {
        float(tet_density[g_idx, 4]),
        float(tet_density[g_idx, 5]),
        float(tet_density[g_idx, 6]),
    };
    return result;
}
