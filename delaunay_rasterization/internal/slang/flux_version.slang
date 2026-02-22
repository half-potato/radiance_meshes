static const uint AUX_DIM = PYTHON_AUX_DIM;

import intersect;
import utils;
import safe_math;
import camera;

const static float DENSITY_OFFSET = -2;
const static float COLOR_BETA = 5;

// ==========================================
// 1. Two-Sample Quadrature Helpers
// ==========================================

struct IntegrationWeights : IDifferentiable {
    float w0; // Weight for entrance value
    float w1; // Weight for exit value
    float alpha; // Total segment opacity
};

[Differentiable]
inline IntegrationWeights compute_two_sample_weights(float d_in, float d_out, float dt) {
    // Two-sample (trapezoidal) quadrature:
    //   Split the segment into two equal half-segments.
    //   Sample 0 uses density d_in  over [0, dt/2]
    //   Sample 1 uses density d_out over [dt/2, dt]
    //
    // Each sub-segment has constant density and constant color.
    // tau_0 = d_in  * (dt / 2)
    // tau_1 = d_out * (dt / 2)
    //
    // Front-to-back compositing of the two sub-segments:
    //   T_0     = exp(-tau_0)
    //   alpha_0 = 1 - T_0
    //   T_1     = exp(-tau_1)
    //   alpha_1 = 1 - T_1
    //
    //   Combined color  = alpha_0 * C_in  +  T_0 * alpha_1 * C_out
    //   Combined alpha  = 1 - T_0 * T_1  =  1 - exp(-(tau_0 + tau_1))

    float half_dt = 0.5f * dt;

    float tau0 = max(d_in, 0.0f) * half_dt;
    float tau1 = max(d_out, 0.0f) * half_dt;

    float T0 = safe_exp(-tau0);
    float T1 = safe_exp(-tau1);

    float alpha0 = 1.0f - T0;
    float alpha1 = 1.0f - T1;

    IntegrationWeights w;
    w.w0 = alpha0;           // Weight for entrance color (first sub-segment)
    w.w1 = T0 * alpha1;     // Weight for exit color (second sub-segment, attenuated by first)
    w.alpha = 1.0f - T0 * T1; // Total segment opacity
    return w;
}

// ==========================================
// 2. Data Structures
// ==========================================

struct LinearField : IDifferentiable {
    float3 grad; 
    float  div;
};

struct FluxTetrahedra : IDifferentiable
{
    uint4 vtet;
    Tetrahedra tet;
    
    LinearField density;
    LinearField r;
    LinearField g;
    LinearField b;
    LinearField aux[AUX_DIM]; 
};

struct CtrlPt : IDifferentiable {
    float4 rgba;
    float2 dist;
    float density;
    float4 xyzd;
    float aux[AUX_DIM]; 
};

// ==========================================
// 3. Field Evaluation Helpers
// ==========================================

[Differentiable]
inline LinearField compute_linear_field(
    float4 face_fluxes, 
    float3 e1, float3 e2, float3 e3, 
    float inv_3V
)
{
    // Divergence (Slope)
    float div = (face_fluxes.x + face_fluxes.y + face_fluxes.z + face_fluxes.w) * inv_3V;

    // Gradient (Base vector)
    float3 weighted_sum = (face_fluxes.y * e1) + (face_fluxes.z * e2) + (face_fluxes.w * e3);
    float3 grad = weighted_sum * -inv_3V;

    LinearField result;
    result.grad = grad;
    result.div  = div;
    return result;
}

[Differentiable]
inline float evaluate_linear_field_at_proj(
    LinearField f, 
    float3 ray_d, 
    float p_proj
) 
{
    // u(x) . d = (a . d) + b * (x . d)
    return dot(f.grad, ray_d) + f.div * p_proj;
}

// ==========================================
// 4. Main Integration Logic
// ==========================================

[Differentiable]
inline CtrlPt integrate_tetra_segment(
    in FluxTetrahedra g, 
    in Ray ray,
    in float enter,
    in float exit,
    in float3 enter_norm,
    in float3 exit_norm
)
{
    float dt = max(exit - enter, 0.0f);
    
    // 1. Geometry Prep
    float3 v0 = g.tet.verts[0];
    
    // Project ray positions relative to v0 onto ray direction
    float3 p_enter_vec = (ray.o + enter * ray.d) - v0;
    float3 p_exit_vec  = (ray.o + exit * ray.d) - v0;
    
    float p_enter = dot(p_enter_vec, ray.d);
    float p_exit  = dot(p_exit_vec,  ray.d);

    // 2. Evaluate Density at Enter/Exit
    float d_in  = safe_exp(evaluate_linear_field_at_proj(g.density, ray.d, p_enter) + DENSITY_OFFSET);
    float d_out = safe_exp(evaluate_linear_field_at_proj(g.density, ray.d, p_exit) + DENSITY_OFFSET);
    
    // 3. Compute Two-Sample Quadrature Weights
    //    Each half-segment uses one sample point (entrance or exit).
    //    Density is clamped inside the weight computation.
    IntegrationWeights weights = compute_two_sample_weights(d_in, d_out, dt);
    
    // Average density (for diagnostic / output only)
    float density_avg = 0.5f * (max(d_in, 0.0f) + max(d_out, 0.0f));

    // 4. Integrate Color Channels
    //    w0 * C_enter + w1 * C_exit  (two-sample front-to-back composite)
    float r_in  = softplus(evaluate_linear_field_at_proj(g.r, ray.d, p_enter), COLOR_BETA);
    float r_out = softplus(evaluate_linear_field_at_proj(g.r, ray.d, p_exit), COLOR_BETA);
    float r_int = weights.w0 * max(r_in, 0.f) + weights.w1 * max(r_out, 0.f);
    
    float g_in  = softplus(evaluate_linear_field_at_proj(g.g, ray.d, p_enter), COLOR_BETA);
    float g_out = softplus(evaluate_linear_field_at_proj(g.g, ray.d, p_exit), COLOR_BETA);
    float g_int = weights.w0 * max(g_in, 0.f) + weights.w1 * max(g_out, 0.f);

    float b_in  = softplus(evaluate_linear_field_at_proj(g.b, ray.d, p_enter), COLOR_BETA);
    float b_out = softplus(evaluate_linear_field_at_proj(g.b, ray.d, p_exit), COLOR_BETA);
    float b_int = weights.w0 * max(b_in, 0.f) + weights.w1 * max(b_out, 0.f);

    // 5. Integrate Aux
    float aux_integrated[AUX_DIM];
    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        float a_in  = evaluate_linear_field_at_proj(g.aux[i], ray.d, p_enter);
        float a_out = evaluate_linear_field_at_proj(g.aux[i], ray.d, p_exit);
        // Aux may be negative, so no clamping
        aux_integrated[i] = weights.w0 * a_in + weights.w1 * a_out;
    }

    // 6. Pack Result
    //    rgba.rgb = pre-integrated emitted radiance for this segment
    //    rgba.a   = segment opacity (alpha)
    //
    //    With two-sample quadrature the weights w0/w1 are dimensionless
    //    alpha values (incorporating dt via the sub-segment optical depths),
    //    so the products w0*C + w1*C already have the correct scale — no
    //    extra dt multiplication is needed, matching the original contract.
    
    CtrlPt result;
    result.rgba    = float4(r_int, g_int, b_int, weights.alpha);
    result.dist    = float2(enter, exit);
    result.density = density_avg;
    result.xyzd    = float4(enter_norm, weights.alpha);

    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        result.aux[i] = aux_integrated[i];
    }
    
    return result;
}

[Differentiable]
inline bool evaluate_tetra_interp(
    in FluxTetrahedra g, 
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
        ctrl_pt = { (float4)(0.f), {0.f, 0.f}, 0.f };
        return false;
    } else {
        ctrl_pt = integrate_tetra_segment(g, ray, enter, exit, enter_norm, exit_norm);
        return true;
    }
}

// ==========================================
// 5. Loading Logic
// ==========================================

[Differentiable]
inline FluxTetrahedra load_tet_alphablend_interp(
    int32_t g_idx,
    DiffTensorView vertices,
    TensorView<int32_t> indices,
    TensorView<int32_t> tet_face_ids,
    DiffTensorView face_values
)
{
    // 1. Load Geometry
    no_diff uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet = load_tetrahedra(vertices, virtual_tet);

    // 2. Precompute Geometry Constants
    float3 v0 = tet.verts[0];
    float3 e1 = tet.verts[1] - v0;
    float3 e2 = tet.verts[2] - v0;
    float3 e3 = tet.verts[3] - v0;

    float vol6 = abs(dot(e1, cross(e2, e3)));
    float inv_3V = 1;//detach(safe_div(2.0f, vol6+1e-2));

    FluxTetrahedra result;
    result.vtet = virtual_tet;
    result.tet = tet;

    // 3. Unpack Face Indices and Signs
    int4 raw_ids = int4(
        tet_face_ids[g_idx, 0],
        tet_face_ids[g_idx, 1],
        tet_face_ids[g_idx, 2],
        tet_face_ids[g_idx, 3]
    );

    int4 f_ids;
    f_ids.x = (raw_ids.x < 0) ? ~raw_ids.x : raw_ids.x;
    f_ids.y = (raw_ids.y < 0) ? ~raw_ids.y : raw_ids.y;
    f_ids.z = (raw_ids.z < 0) ? ~raw_ids.z : raw_ids.z;
    f_ids.w = (raw_ids.w < 0) ? ~raw_ids.w : raw_ids.w;

    float4 signs;
    signs.x = (raw_ids.x < 0) ? -1.0f : 1.0f;
    signs.y = (raw_ids.y < 0) ? -1.0f : 1.0f;
    signs.z = (raw_ids.z < 0) ? -1.0f : 1.0f;
    signs.w = (raw_ids.w < 0) ? -1.0f : 1.0f;

    // 4. Gather & Compute Fields

    // -- Density --
    float4 f_density = float4(
                           face_values[f_ids.x, 0], face_values[f_ids.y, 0],
                           face_values[f_ids.z, 0], face_values[f_ids.w, 0]
    ) * signs;
    result.density = compute_linear_field(f_density, e1, e2, e3, inv_3V);

    // -- R, G, B --
    float4 f_r = float4(
                     face_values[f_ids.x, 1], face_values[f_ids.y, 1],
                     face_values[f_ids.z, 1], face_values[f_ids.w, 1]
    ) * signs;
    result.r = compute_linear_field(f_r, e1, e2, e3, inv_3V);

    float4 f_g = float4(
                     face_values[f_ids.x, 2], face_values[f_ids.y, 2],
                     face_values[f_ids.z, 2], face_values[f_ids.w, 2]
    ) * signs;
    result.g = compute_linear_field(f_g, e1, e2, e3, inv_3V);

    float4 f_b = float4(
                     face_values[f_ids.x, 3], face_values[f_ids.y, 3],
                     face_values[f_ids.z, 3], face_values[f_ids.w, 3]
    ) * signs;
    result.b = compute_linear_field(f_b, e1, e2, e3, inv_3V);

    // -- Aux --
    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        int ch = 4 + i;
        float4 f_aux = float4(
                           face_values[f_ids.x, ch], face_values[f_ids.y, ch],
                           face_values[f_ids.z, ch], face_values[f_ids.w, ch]
        ) * signs;
        result.aux[i] = compute_linear_field(f_aux, e1, e2, e3, inv_3V);
    }

    return result;
}
