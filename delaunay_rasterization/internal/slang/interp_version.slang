import intersect;
import utils;
import safe_math;

struct SplatTetrahedra_interp : IDifferentiable
{
    Tetrahedra tet;
    // float4x3 vertex_color; // Per-vertex color to be barycentrically interpolated
    float3 base_color;
    float3 grd;
    float density;          // Single scalar density for the entire tetrahedron
};

// Given three vertex-colors (c0, c1, c2) and the barycentric coords (u,v)
//   w0 = u, w1 = v, w2 = 1 - u - v
// returns the final interpolated color:
[Differentiable]
float3 interpolate_triangle_color(
    float3 c0, float3 c1, float3 c2,
    float u, float v)
{
    float w0 = u;
    float w1 = v;
    float w2 = 1.0f - w0 - w1;
    return c0 * w0 + c1 * w1 + c2 * w2;
}

[Differentiable]
float4 tet_barycentric(float3 p, float4x3 vs)
{
    // Construct matrix T from tetrahedron vertices
    float3x3 T = float3x3(vs[0] - vs[3], vs[1] - vs[3], vs[2] - vs[3]);

    // Solve for (u, v, w)
    float3 uvw = SolveLinearSystem(T, p - vs[3]);

    // Compute t
    float t = 1.0 - uvw.x - uvw.y - uvw.z;

    return float4(uvw, t);
}

void print_verts(float4x3 verts) {
    printf("%f, %f, %f, %f\n", verts[0].x, verts[1].x, verts[2].x, verts[3].x);
}
void print_vec(float3 vec) {
    printf("(%f, %f, %f),", vec.x, vec.y, vec.z);
}

[Differentiable]
float3 compute_integral(float3 c0, float3 c1, float dt, float d) {
    float alpha = safe_exp(-d*dt);
    float X = (-d*dt*alpha + 1 - alpha);
    float Y = (d*dt-1) + alpha;
    return safe_div(X*c0+Y*c1, d*dt);
}

[Differentiable]
float compute_integral_1D(float c0, float c1, float d_dt) {
    float alpha = safe_exp(-d_dt);
    float X = (-d_dt*alpha + 1 - alpha);
    float Y = (d_dt-1) + alpha;
    return safe_div(X*c0+Y*c1, d_dt);
}

[Differentiable]
inline float integrate_channel(
    float t_n, float t_f,
    float c_at_t0, float dc_dt, float density)
{
    // Find zero-crossing and adjust integration bounds to the positive segment
    float t_zero = - safe_div(c_at_t0, dc_dt);
    bool change_within = (t_n < t_zero) && (t_zero < t_f);
    float c0 = max(c_at_t0 + dc_dt * t_n, 0.f);
    float c1 = max(c_at_t0 + dc_dt * t_f, 0.f);
    if (change_within) {
        float cm = max(c_at_t0 + dc_dt * t_zero, 0.f);
        float dt1 = max(t_zero - t_n, 0);
        float dt2 = max(t_f - t_zero, 0);
        float onemalpha = safe_clip(safe_exp(-density * dt1), 0.f, 1.f);
        if (dc_dt < 0.f) {
            return compute_integral_1D(cm, c0, dt1 * density);
        }
        if (dc_dt > 0.f) {
            //return compute_integral_1D(0, 0, dt1, density) + onemalpha * compute_integral_1D(c1, cm, dt2, density);
            return onemalpha * compute_integral_1D(c1, cm, dt2 * density);
        }
    }
    float dt = max(t_f - t_n, 0.f);
    return compute_integral_1D(c1, c0, dt * density);
}

/*
[Differentiable]
inline float integrate_channel(
    float t_n, float t_f,
    float c_at_t0, float dc_dt, float density)
{
    // Find where the linear color function C(t) would cross zero.
    float t_zero = clamp(-safe_div(c_at_t0, dc_dt), t_n, t_f);

    // Determine the start and end of the segment where C(t) > 0.
    // This replaces the main "if (change_within)" and nested branches.
    float t_start = (dc_dt > 0.0f) ? t_zero : t_n;
    float t_end   = (dc_dt < 0.0f) ? t_zero : t_f;

    // Clamp this "positive" segment to the actual integration bounds [t_n, t_f].
    float dt_pos_segment = t_end - t_start;
    float d_dt = dt_pos_segment * density;
    if (d_dt < 1e-3) {
        return 0.f;
    }

    // Calculate transmittance through the initial "zero-color" segment [t_n, t_start].
    float dt_zero_segment = t_start - t_n;
    float T_zero_segment = safe_exp(-density * dt_zero_segment);

    // Calculate the integral over the clamped positive segment [t_start, t_end].
    float c_start = c_at_t0 + dc_dt * t_start;
    float c_end   = c_at_t0 + dc_dt * t_end;

    // The final result is the attenuated integral over the positive part.
    return T_zero_segment * compute_integral_1D(c_end, c_start, d_dt);
}
*/

[Differentiable]
bool evaluate_tetra_interp(
    in SplatTetrahedra_interp g, 
    in float3 ray_o,
    in float3 ray_d,
    no_diff in float t_min,
    out CtrlPt ctrl_pt)
{
    float2 dist;

    float4x3 verts = float4x3(g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]);

    bool hit = ray_tetrahedron_intersect_fused(
        ray_o, ray_d, verts, dist);

    float enter = dist.x;
    float exit = max(dist.y, enter);
    float dt = max(dist.y - enter, 0.0f);
    float intersection_t = 0.5*(exit+enter);

    if (!hit || dt < 1e-5f) {
        ctrl_pt = { (float4)(0.f, 0.f, 0.f, 0.f), {0.f, 0.f}, g.density };
        return false;
    } else {
        float3 v_enter = enter * ray_d + ray_o - verts[0];
        float3 v_exit = exit * ray_d + ray_o - verts[0];

        float alpha = safe_clip(-safe_expm1(-g.density * dt), 0.f, 1.f);
        float dc_dt = dot(g.grd, ray_d);
        float offset = dot(ray_o - verts[0], g.grd);
        float3 final_color = {
            integrate_channel(enter, exit, g.base_color.r + offset, dc_dt, g.density),
            integrate_channel(enter, exit, g.base_color.g + offset, dc_dt, g.density),
            integrate_channel(enter, exit, g.base_color.b + offset, dc_dt, g.density)
        };
        //*/

        float4 rgba = {final_color.x,
                       final_color.y,
                       final_color.z,
                       alpha};
        ctrl_pt = { rgba, {enter, exit}, g.density };
        return true;
    }
}

////////////////////////////////////////////////////////////////////////////////
// 5) Loading functions for separate density & vertex colors
//    We add "_interp" suffix to keep the old ones intact.
////////////////////////////////////////////////////////////////////////////////

[Differentiable]
SplatTetrahedra_interp load_tet_alphablend_interp(
    int32_t        g_idx,
    DiffTensorView vertices,
    TensorView<int32_t> indices,
    DiffTensorView tet_density)
{
    // 4 ints
    no_diff uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet); // 12 floats

    // Here we assume each tetrahedron has a single scalar density in tet_density:
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
    // 7 floats

    SplatTetrahedra_interp result = {
        tet, base_color, grd, density
    };
    return result;
}

// Non-diff version (if needed)
SplatTetrahedra_interp load_tet_alphablend_interp(
    int32_t        g_idx,
    TensorView<float> vertices,
    TensorView<int32_t> indices,
    TensorView<float>  tet_density)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);

    float density = max(float(tet_density[g_idx, 0]), 0.f);
    SplatTetrahedra_interp result;
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
