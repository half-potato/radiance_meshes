import intersect;
import utils;
import safe_math;

struct SplatTetrahedra : IDifferentiable
{
    Tetrahedra tet;
    float4x4 vertex_rgbs; // Per-vertex color to be barycentrically interpolated
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
inline float4 mix_color(float4x4 vertex_color, float4 bary) {
    return vertex_color[0] * bary.x + vertex_color[1] * bary.y + vertex_color[2] * bary.z + vertex_color[3] * bary.w;
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

[Differentiable]
bool evaluate_tetra_interp(
    in SplatTetrahedra g, 
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
        float3x3 T = transpose(float3x3(verts[0] - verts[3], verts[1] - verts[3], verts[2] - verts[3]));
        float3 uvw_enter = SolveLinearSystem(T, enter * ray_d + ray_o - verts[3]);
        float3 uvw_exit = SolveLinearSystem(T, dist.y * ray_d + ray_o - verts[3]);

        float w_enter = safe_clip(1.0 - uvw_enter.x - uvw_enter.y - uvw_enter.z, 0, 1);
        float w_exit = safe_clip(1.0 - uvw_exit.x - uvw_exit.y - uvw_exit.z, 0, 1);

        float4 b_enter = float4(uvw_enter, w_enter);
        float4 b_exit = float4(uvw_exit, w_exit);

        float4 c_enter = mix_color(g.vertex_color, b_enter);
        float4 c_exit = mix_color(g.vertex_color, b_exit);

        float alpha = safe_clip(1.0f - safe_exp(-g.density * dt), 0.f, 1.f);
        float3 final_color = compute_integral(c_exit, c_enter, dt, g.density);

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
SplatTetrahedra load_tet_alphablend_interp(
    int32_t        g_idx,
    DiffTensorView vertices,
    TensorView<int32_t> indices,
    DiffTensorView vert_attr)
{
    // 4 ints
    no_diff uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet); // 12 floats
    float4x4 vert_rgbs = float4x4(
        read_t4_float4(virtual_tet.x, vert_attr),
        read_t4_float4(virtual_tet.y, vert_attr),
        read_t4_float4(virtual_tet.z, vert_attr),
        read_t4_float4(virtual_tet.w, vert_attr)
    );
    SplatTetrahedra result = {
        tet, vert_rgbs
    };
    return result;
}

// Non-diff version (if needed)
SplatTetrahedra load_tet_alphablend_interp(
    int32_t        g_idx,
    TensorView<float> vertices,
    TensorView<int32_t> indices,
    TensorView<float>  vert_attr)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);
    float4x4 vert_rgbs = float4x4(
        read_t4_float4(virtual_tet.x, vert_attr),
        read_t4_float4(virtual_tet.y, vert_attr),
        read_t4_float4(virtual_tet.z, vert_attr),
        read_t4_float4(virtual_tet.w, vert_attr)
    );
    SplatTetrahedra result = {
        tet, vert_rgbs
    };
    return result;
}
