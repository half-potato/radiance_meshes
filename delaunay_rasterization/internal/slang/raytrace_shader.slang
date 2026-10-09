import utils;
import intersect;
import safe_math;

#ifndef PYTHON_AUX_DIM
#define PYTHON_AUX_DIM 0
#endif

static const int AUX_DIM = PYTHON_AUX_DIM;
// cell_values columns: [0]=density, [1:4]=base_rgb_v0, [4:7]=color_grad,
// [7:7+AUX_DIM]=aux. AUX_DIM > 0 forwards an alpha-weighted blend of those
// channels into aux_img.
static const int CELL_AUX_OFFSET = 7;

#if PYTHON_AUX_DIM > 0
#define AUX_BUF_SIZE PYTHON_AUX_DIM
#else
#define AUX_BUF_SIZE 1
#endif

static const float LOG_T_THRESHOLD = -5.54f; // log(1/255)
static const float MAX_OD = 88.0f;
static const float MIN_OD = 1e-8f;

// Face j is opposite vertex j. xyz are the triangle indices, w is the
// opposite vertex (used to flip the face normal inward). Matches build_adj
// and the WGSL TET_FACES constant.
static const uint4 kTetFaces[4] = {
    uint4(0u, 2u, 1u, 3u),
    uint4(1u, 2u, 3u, 0u),
    uint4(0u, 3u, 2u, 1u),
    uint4(3u, 0u, 1u, 2u),
};

float phi_fn(float od)
{
    // (1 - exp(-od)) / od. od is clamped to [MIN_OD, MAX_OD] before this is
    // called, so the divide is safe.
    return safe_div(1.0f - exp(-od), od);
}

struct RayTetEval
{
    float3 c_premul;
    float od;
    float t_enter;
    float t_exit;
    int entry_face;
    int exit_face;
    float3 entry_normal;
};

RayTetEval eval_tet_rt(float3 cam,
                       float3 ray_dir,
                       float t_near,
                       float t_far,
                       float density,
                       float3 colors_v0,
                       float3 grad,
                       Tetrahedra tet)
{
    RayTetEval res;
    res.c_premul = float3(0.0f, 0.0f, 0.0f);
    res.od = -1.0f;
    res.t_enter = t_near;
    res.t_exit = t_near;
    res.entry_face = 4;
    res.exit_face = 4;
    res.entry_normal = float3(0.0f, 0.0f, 0.0f);

    float t_min_val = -1e30f;
    float t_max_val =  1e30f;
    int entry_face_idx = 4;
    int exit_face_idx = 4;
    float3 face_normals[4];

    [unroll]
    for (int fi = 0; fi < 4; ++fi) {
        uint4 f = kTetFaces[fi];
        float3 va = tet.verts[f.x];
        float3 vb = tet.verts[f.y];
        float3 vc = tet.verts[f.z];
        float3 v_opp = tet.verts[f.w];
        float3 n = cross(vc - va, vb - va);
        if (dot(n, v_opp - va) < 0.0f) n = -n;
        face_normals[fi] = n;

        float num = dot(n, va - cam);
        float den = dot(n, ray_dir);

        // Length-scaled parallel check: `n` is unnormalized, so `den` scales
        // with |n|. An absolute eps like 1e-20 is far too small for typical
        // face areas and only catches numerical zeros, while small-but-real
        // dens still produce wildly large t = num/den that destroy t_min/
        // t_max. Scale the tolerance with |n| so the check is geometric --
        // equivalent to checking the angle between unit ray_dir and unit
        // face normal against ~1e-6.
        float n_len = length(n);
        if (abs(den) < n_len * 1e-6f) {
            // Ray is (numerically) parallel to this face plane. It cannot
            // be a meaningful entry or exit; skip without updating
            // t_min/t_max. We deliberately do NOT mark the tet invalid
            // based on which side of the plane the camera is on -- for
            // thin/sliver tets that test fires spuriously while the ray
            // does still pass through the volume via the OTHER faces.
            continue;
        }
        float t = num / den;
        if (den > 0.0f) {
            if (t > t_min_val) { t_min_val = t; entry_face_idx = fi; }
        } else {
            if (t < t_max_val) { t_max_val = t; exit_face_idx = fi; }
        }
    }

    t_min_val = max(t_min_val, t_near);
    // Clip the segment at t_far. If the original tet exit was past t_far,
    // we no longer flag a valid exit face -- the ray is supposed to stop
    // here, not continue into a neighbour.
    bool stopped_at_far = false;
    if (t_max_val > t_far) {
        t_max_val = t_far;
        stopped_at_far = true;
    }
    // Recovery for thin-tet ambiguity: when no face produced a real exit
    // (all four were judged near-parallel, or only entry faces were
    // found) AND we're not legitimately stopping at t_far, pick a
    // fallback exit face -- the one whose inward normal is most
    // aligned with -ray_dir, i.e. the face the ray is most "headed
    // toward" -- and bump the segment length by a small amount so the
    // walk advances. Without this fallback, ~0.5% of pixels see the
    // walk terminate immediately, producing scattered hot pixels in
    // shadow rays (alpha=0 → visibility=1 → spurious bright dots).
    // Recovery for thin-tet ambiguity (force-recompile marker).
    if (exit_face_idx >= 4 && !stopped_at_far) {
        float best_dot_v = 1e30f;
        int best_fi_v = 0;
        [unroll]
        for (int fi2 = 0; fi2 < 4; ++fi2) {
            float dv = dot(face_normals[fi2], ray_dir);
            if (dv < best_dot_v) { best_dot_v = dv; best_fi_v = fi2; }
        }
        exit_face_idx = best_fi_v;
        t_max_val = max(t_max_val, t_min_val + 1e-3f);
        t_max_val = min(t_max_val, t_far);
    }
    bool clipped_by_far = stopped_at_far;
    if (t_min_val >= t_max_val) {
        return res;
    }

    if (entry_face_idx < 4) {
        res.entry_normal = l2_normalize(-face_normals[entry_face_idx]);
    }
    res.entry_face = entry_face_idx;
    // If t_far cut the segment, signal end-of-walk by reporting no real
    // exit face (the kernel uses exit_face >= 4 as the loop break flag).
    res.exit_face = clipped_by_far ? 4 : exit_face_idx;
    res.t_enter = t_min_val;
    res.t_exit = t_max_val;

    // Affine color along the ray segment, same model as the rasterizer.
    float base_offset = dot(grad, cam - tet.verts[0]);
    float3 base_color = colors_v0 + float3(base_offset, base_offset, base_offset);
    float dc_dt = dot(grad, ray_dir);
    float3 c_start = max(base_color + float3(dc_dt * t_min_val, dc_dt * t_min_val, dc_dt * t_min_val), float3(0.0f));
    float3 c_end   = max(base_color + float3(dc_dt * t_max_val, dc_dt * t_max_val, dc_dt * t_max_val), float3(0.0f));

    float dist = t_max_val - t_min_val;
    float od = safe_clip(density * dist, MIN_OD, MAX_OD);
    float alpha_t = exp(-od);
    float phi_val = phi_fn(od);
    float w0 = phi_val - alpha_t;
    float w1 = 1.0f - phi_val;
    res.c_premul = safe_clip(c_end * w0 + c_start * w1, 0.0f, MAX_VAL);
    res.od = od;
    return res;
}

// Fast-path evaluation for tets pre-flagged as slivers/thin.
//
// `eval_tet_rt`'s 4-face plane-intersection scheme is fragile on slivers:
// multiple faces lie near-parallel to typical rays, the parallel-test guard
// fires inconsistently, and the wrong t_min / t_max gets latched. The
// recovery branch (`exit_face_idx >= 4 && !stopped_at_far`) only kicks in
// when ALL four faces fail -- a sliver that erroneously latches an exit on
// one bogus face slips through and corrupts the walk.
//
// Bypass it: pick the single face whose inward normal is most anti-parallel
// to `ray_dir` -- the face the ray is most "headed toward" -- and intersect
// with that one plane. Single-plane intersection is well-conditioned
// because we chose the face with the steepest geometric contact. Returned
// `od = 0` means the outer walker accumulates nothing from this tet.
RayTetEval eval_thin_tet(float3 cam,
                         float3 ray_dir,
                         float t_near,
                         float t_far,
                         Tetrahedra tet)
{
    RayTetEval res;
    res.c_premul = float3(0.0f, 0.0f, 0.0f);
    res.od = 0.0f;
    res.t_enter = t_near;
    res.t_exit = t_near;
    res.entry_face = 4;
    res.exit_face = 4;
    res.entry_normal = float3(0.0f, 0.0f, 0.0f);

    float best_dot = 1e30f;
    int best_fi = 0;
    float3 best_n = float3(0.0f, 0.0f, 0.0f);
    float3 best_va = float3(0.0f, 0.0f, 0.0f);
    [unroll]
    for (int fi = 0; fi < 4; ++fi) {
        uint4 f = kTetFaces[fi];
        float3 va = tet.verts[f.x];
        float3 vb = tet.verts[f.y];
        float3 vc = tet.verts[f.z];
        float3 v_opp = tet.verts[f.w];
        float3 n = cross(vc - va, vb - va);
        if (dot(n, v_opp - va) < 0.0f) n = -n;
        // n points inward (toward v_opp). Most negative dot(n, ray_dir)
        // = face the ray is heading INTO from the tet interior.
        float dv = dot(n, ray_dir);
        if (dv < best_dot) {
            best_dot = dv;
            best_fi = fi;
            best_n = n;
            best_va = va;
        }
    }

    // Pathological case: even the best-aligned face is near-parallel. Step
    // forward by a tiny epsilon along the chosen face and rely on the
    // neighbour to recover. Same constant as the eval_tet_rt fallback.
    float n_len = length(best_n);
    if (abs(best_dot) < n_len * 1e-6f) {
        res.t_exit = min(t_near + 1e-3f, t_far);
        res.exit_face = (res.t_exit > t_near) ? best_fi : 4;
        return res;
    }

    // Plane: n·(x - va) = 0 with n inward. Ray inside the tet at t_near has
    // n·(p_near - va) > 0; the ray exits when this crosses zero.
    //   t_exit = n·(va - cam) / (n · ray_dir)
    float t = dot(best_n, best_va - cam) / best_dot;
    t = max(t, t_near + 1e-6f);
    if (t >= t_far) {
        res.t_exit = t_far;
        res.exit_face = 4;   // hit far cap -- end-of-walk sentinel
        return res;
    }

    res.t_enter = t_near;
    res.t_exit = t;
    res.exit_face = best_fi;
    return res;
}

[AutoPyBindCUDA]
[CUDAKernel]
void ray_trace(TensorView<float> rays,           // [N, 6] origin+dir
               TensorView<int32_t> indices,      // [T, 4]
               TensorView<float> vertices,       // [V, 3]
               TensorView<float> cell_values,    // [T, 7+AUX_DIM]
               TensorView<int32_t> tet_adj,      // [T, 4] -1 for boundary
               TensorView<int32_t> start_tet,    // [N]
               TensorView<float> t_far,          // [N] per-ray far cap
               TensorView<int32_t> thin_mask,    // [T] nonzero ⇒ skip fragile face-intersection, use eval_thin_tet
               TensorView<float> output_img,     // [N, 4] rgb+alpha
               TensorView<float> aux_img,        // [N, 8+AUX_DIM]
               TensorView<int32_t> n_steps,      // [N]
               TensorView<float> t_saturate,     // [N] t at which alpha first crossed sat_threshold; -1 if never
               float min_t,
               int max_iters,
               float saturate_log_t)            // log_t threshold; e.g. -3 ⇒ alpha 0.95
{
    uint pixel_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    uint N = uint(rays.size(0));
    if (pixel_idx >= N) return;
    float ray_t_far = t_far[pixel_idx];

    float3 ray_o = float3(rays[uint2(pixel_idx, 0u)],
                          rays[uint2(pixel_idx, 1u)],
                          rays[uint2(pixel_idx, 2u)]);
    float3 ray_d_raw = float3(rays[uint2(pixel_idx, 3u)],
                              rays[uint2(pixel_idx, 4u)],
                              rays[uint2(pixel_idx, 5u)]);
    float3 ray_d = l2_normalize(ray_d_raw);
    ray_o = ray_o + ray_d * min_t;

    int current_tet = start_tet[pixel_idx];
    float t_cursor = 0.0f;

    float3 color_accum  = float3(0.0f, 0.0f, 0.0f);
    float3 normal_accum = float3(0.0f, 0.0f, 0.0f);
    float depth_accum   = 0.0f;
    // Volumetric VSM second moment: alpha-premultiplied front-to-back
    // composite of E[z^2] over ray segments. Combined with depth_accum
    // (E[z]) and the final alpha, lets a downstream pass recover variance
    // and apply a Chebyshev bound for soft shadows. See evaluate_transmittance
    // in rmesh-render's deferred_shade_frag.wgsl for the consumer.
    float depth2_accum  = 0.0f;
    float4 entropy      = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float aux_accum[AUX_BUF_SIZE];
    [unroll]
    for (int ai0 = 0; ai0 < AUX_BUF_SIZE; ++ai0) aux_accum[ai0] = 0.0f;

    float log_t = 0.0f;
    int iter = 0;
    int n_walked = 0;
    // First-saturation t: -1 means the ray never reached alpha=sat_threshold
    // before terminating. Otherwise, the t (along the ray, in world units) at
    // which log_t first crossed `saturate_log_t`. Used to detect real
    // occluders vs the surface itself: if t_sat << dist_to_surface, there's
    // an actual shadow caster between the light and the receiver.
    float t_sat = -1.0f;

    [loop]
    while (current_tet >= 0 && log_t > LOG_T_THRESHOLD && iter < max_iters) {
        uint4 vt = load_virtual_tetrahedra(current_tet, indices);
        Tetrahedra tet = load_tetrahedra(vertices, vt);

        // Thin-tet branch: skip the 4-face intersection (numerically unstable
        // on slivers) and use a single-plane fast path. Returns od = 0 so
        // the accumulators below treat this tet as transparent.
        bool is_thin = thin_mask[uint(current_tet)] != 0;
        RayTetEval r;
        if (is_thin) {
            r = eval_thin_tet(ray_o, ray_d, t_cursor, ray_t_far, tet);
        } else {
            float density = cell_values[uint2(uint(current_tet), 0u)];
            float3 colors_v0 = float3(cell_values[uint2(uint(current_tet), 1u)],
                                       cell_values[uint2(uint(current_tet), 2u)],
                                       cell_values[uint2(uint(current_tet), 3u)]);
            float3 grad = float3(cell_values[uint2(uint(current_tet), 4u)],
                                  cell_values[uint2(uint(current_tet), 5u)],
                                  cell_values[uint2(uint(current_tet), 6u)]);
            r = eval_tet_rt(ray_o, ray_d, t_cursor, ray_t_far,
                            density, colors_v0, grad, tet);
        }
        n_walked++;

        if (r.od > 0.0f) {
            float T_j = safe_exp(log_t);
            color_accum += r.c_premul * T_j;
            float alpha = 1.0f - exp(-r.od);
            normal_accum += T_j * alpha * r.entry_normal;

            float phi_val = phi_fn(r.od);
            float alpha_t = exp(-r.od);
            float w0 = phi_val - alpha_t;
            float w1 = 1.0f - phi_val;
            // Exact within-segment α·E[z²] for the exponential termination
            // distribution on [t_enter, t_exit] (PDF f(z) = σ_eff·exp(-σ_eff
            // (z - t_enter))):
            //   α·E[z²] = w0·t_exit² + w1·t_enter² − L²·correction(od)
            // where correction(od) = w0 + α_t − 2·w0/od ≥ 0. For small od the
            // direct form has catastrophic cancellation; Taylor expansion:
            //   correction(od) ≈ od/6 − od²/12 + od³/40
            // Matches interval_fragment.wgsl. The earlier 2-point upper-bound
            // (w0·t_exit² + w1·t_enter² with no correction) over-estimated
            // variance and softened shadows unnecessarily.
            //
            // Thin-segment filter: skip moment writes when the segment carries
            // effectively no absorption — for very small od (e.g. zero-density
            // empty tets clamped to MIN_OD), the identity w0 + w1 = alpha
            // breaks down in fp32 and would poison the running accumulators.
            // Other accumulators are already alpha-weighted and self-attenuate.
            if (alpha > 0.002f) {
                float depth_premul = w0 * r.t_exit + w1 * r.t_enter;
                depth_accum += T_j * depth_premul;

                float correction;
                if (abs(r.od) < 0.02f) {
                    correction = r.od * ((1.0f / 6.0f)
                                       - r.od * ((1.0f / 12.0f)
                                               - r.od * (1.0f / 40.0f)));
                } else {
                    correction = w0 + alpha_t - 2.0f * w0 / r.od;
                }
                float L = r.t_exit - r.t_enter;
                float depth2_premul = w0 * r.t_exit * r.t_exit
                                    + w1 * r.t_enter * r.t_enter
                                    - L * L * correction;
                depth2_accum += T_j * depth2_premul;
            }

            float wt = alpha * T_j;
            float c_mid = (r.t_enter + r.t_exit) * 0.5f;
            entropy[0] += wt;
            entropy[1] += wt > 1e-20f ? wt * log(wt) : 0.0f;
            entropy[2] += wt * c_mid;
            float wc = wt * c_mid;
            entropy[3] += wc > 1e-20f ? wc * log(wc) : 0.0f;

            #if PYTHON_AUX_DIM > 0
            [unroll]
            for (int ai = 0; ai < PYTHON_AUX_DIM; ++ai) {
                aux_accum[ai] += T_j * alpha *
                    cell_values[uint2(uint(current_tet), uint(CELL_AUX_OFFSET + ai))];
            }
            #endif

            // Detect first time alpha crosses the saturation threshold.
            // log_t went from log_t_before to log_t_before - od; if it
            // crosses the threshold inside this tet, interpolate the t
            // linearly along the segment for an accurate boundary.
            if (t_sat < 0.0f) {
                float log_t_after = log_t - r.od;
                if (log_t_after < saturate_log_t) {
                    float frac = (log_t - saturate_log_t) / max(r.od, 1e-12f);
                    frac = clamp(frac, 0.0f, 1.0f);
                    t_sat = r.t_enter + frac * (r.t_exit - r.t_enter);
                }
            }
            log_t -= r.od;
        }

        if (r.exit_face >= 4) break;
        int neighbor = tet_adj[uint2(uint(current_tet), uint(r.exit_face))];
        current_tet = neighbor;
        t_cursor = r.t_exit;
        iter++;
    }

    float T_final = safe_exp(log_t);
    output_img[uint2(pixel_idx, 0u)] = color_accum.x;
    output_img[uint2(pixel_idx, 1u)] = color_accum.y;
    output_img[uint2(pixel_idx, 2u)] = color_accum.z;
    output_img[uint2(pixel_idx, 3u)] = 1.0f - T_final;

    aux_img[uint2(pixel_idx, 0u)] = normal_accum.x;
    aux_img[uint2(pixel_idx, 1u)] = normal_accum.y;
    aux_img[uint2(pixel_idx, 2u)] = normal_accum.z;
    aux_img[uint2(pixel_idx, 3u)] = depth_accum;
    aux_img[uint2(pixel_idx, 4u)] = entropy.x;
    aux_img[uint2(pixel_idx, 5u)] = entropy.y;
    aux_img[uint2(pixel_idx, 6u)] = entropy.z;
    aux_img[uint2(pixel_idx, 7u)] = entropy.w;
    aux_img[uint2(pixel_idx, 8u)] = depth2_accum;
    #if PYTHON_AUX_DIM > 0
    [unroll]
    for (int ai2 = 0; ai2 < PYTHON_AUX_DIM; ++ai2) {
        aux_img[uint2(pixel_idx, uint(9 + ai2))] = aux_accum[ai2];
    }
    #endif
    n_steps[pixel_idx] = n_walked;
    t_saturate[pixel_idx] = t_sat;
}
