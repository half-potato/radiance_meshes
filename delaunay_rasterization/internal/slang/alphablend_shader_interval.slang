// Tile-based alpha blending kernel using interval evaluation.
// Replaces per-pixel ray-tet intersection with barycentric fan triangle lookup.

import utils;
import camera;
import safe_math;
import interp_version;
import interval_version;

static const uint TILE_HEIGHT = PYTHON_TILE_HEIGHT;
static const uint TILE_WIDTH = PYTHON_TILE_WIDTH;

// Shared memory: compact per-tet data. Fan vertex data stays in global memory
// (L1 cache coherent since all tile threads read the same tet).
struct IntervalSMData {
    float density;
    float3 base_color;
    float3 grd;
    uint num_sil_verts;
    uint tet_idx;
    float aux[AUX_DIM];
};

groupshared IntervalSMData collected_interval[TILE_HEIGHT * TILE_WIDTH];
groupshared uint32_t collected_idx[TILE_HEIGHT * TILE_WIDTH];

struct DistortionState5 : IDifferentiable {
    float x;
    float y;
    float z;
    float w;
    float v;
};

[Differentiable]
float tukey_power_ladder(float x, float p) {
    if (p == 1) {
        return x;
    }
    let xp = abs(x);
    let xs = xp / max(TINY_VAL, abs(p - 1));
    let y = sign(x) * abs(p - 1) / p * (pow(xs + 1, p) - 1);
    return y;
}

[Differentiable]
float safe_exp4alpha(float x) {
    return safe_clip(safe_exp(x), 0, 1);
}

// Exact moments for piecewise-constant density w(t) = σ·T·exp(-σ(t-a)).
// M0 = T·α = T·(1-exp(-u)),  where u = σ·(b-a)
// M1 = T·(a·α + (1-(1+u)·exp(-u))/σ)
// Self = T²·(1 - 2u·exp(-u) - exp(-2u))/σ
// These converge to the Barron point-mass approximation as u→0.

[Differentiable]
float moment_0_int(float a, float b, float density, float T) {
    float dt = max(b - a, 0);
    float alpha = max(1.0f - safe_exp(-density * dt), 0);
    return T * alpha;
}
[Differentiable]
float moment_1_int(float a, float b, float density, float T) {
    float dt = max(b - a, 0);
    float u = density * dt;
    float exp_neg_u = safe_exp(-u);
    float alpha = max(1.0f - exp_neg_u, 0);
    return T * (a * alpha + safe_div(1 - (1 + u) * exp_neg_u, density));
}

[Differentiable]
float scale_dist(float x) {
    return tukey_power_ladder(160 * x, 0.1);
}


[Differentiable]
float self_dist(float a, float b, float density, float T) {
    float dt = max(b - a, 0);
    float u = density * dt;
    float exp_neg_u = safe_exp(-u);
    return T * T * safe_div(1 - 2 * u * exp_neg_u - exp_neg_u * exp_neg_u, density);
}

[Differentiable]
DistortionState5 update_distortion_state(
    DistortionState5 old_state,
    float T,
    float density,
    float2 dist)
{
    float start = scale_dist(dist.x);
    float end = scale_dist(max(dist.x, dist.y));

    float m0 = moment_0_int(start, end, density, T);
    float m1 = moment_1_int(start, end, density, T);

    DistortionState5 new_state;
    new_state.x = old_state.x + 2.0f * m1 * old_state.z;
    new_state.y = old_state.y + 2.0f * m0 * old_state.w;
    new_state.z = old_state.z + m0;
    new_state.w = old_state.w + m1;
    new_state.v = old_state.v + self_dist(start, end, density, T);

    return new_state;
}

DistortionState5 undo_distortion_state(
    DistortionState5 new_state,
    float T,
    float density,
    float2 dist)
{
    float start = scale_dist(dist.x);
    float end = scale_dist(max(dist.x, dist.y));

    float m0 = moment_0_int(start, end, density, T);
    float m1 = moment_1_int(start, end, density, T);

    DistortionState5 old_state;
    old_state.z = new_state.z - m0;
    old_state.w = new_state.w - m1;
    old_state.x = new_state.x - 2.0f * m1 * old_state.z;
    old_state.y = new_state.y - 2.0f * m0 * old_state.w;
    old_state.v = new_state.v - self_dist(start, end, density, T);
    return old_state;
}

#define LOG_EPS 1e-6f

[Differentiable]
float4 update_pixel_state(float4 pixel_state_t_nm1, float4 gauss_rgba_t_n)
{
    float transmittance_t_nm1 = safe_exp(pixel_state_t_nm1.a);
    float3 color_t_n = pixel_state_t_nm1.rgb + gauss_rgba_t_n.rgb * transmittance_t_nm1;
    float log_transmittance_t_n = pixel_state_t_nm1.a - gauss_rgba_t_n.a;

    return float4(color_t_n, log_transmittance_t_n);
}

float4 undo_pixel_state(float4 pixel_state_t_n, float4 gauss_rgba_t_n)
{
    float log_transmittance_t_nm1 = pixel_state_t_n.a + gauss_rgba_t_n.a;
    float transmittance_t_nm1 = safe_exp(log_transmittance_t_nm1);
    float3 color_t_nm1 = pixel_state_t_n.rgb - gauss_rgba_t_n.rgb * transmittance_t_nm1;
    return float4(color_t_nm1, log_transmittance_t_nm1);
}

[Differentiable]
float4 update_xyzd_state(in float4 prev_xyzd, in float4 ctrl_xyzd, in float T) {
    return prev_xyzd + T * ctrl_xyzd;
}

float4 undo_xyzd_state(in float4 next_xyzd, in float4 ctrl_xyzd, in float T) {
    return next_xyzd - T * ctrl_xyzd;
}


struct AlphaOut: IDifferentiable {
    float4 rgba;
    DistortionState5 distortion_state;
    float4 xyzd;
    float aux[AUX_DIM];
};

// Helper to create a zeroed struct
AlphaOut get_empty_alpha_out() {
    AlphaOut ao;
    ao.rgba = float4(0.f);
    ao.distortion_state = {0.f, 0.f, 0.f, 0.f, 0.f};
    ao.xyzd = float4(0.f);
    
    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        ao.aux[i] = 0.f;
    }
    return ao;
}

[Differentiable]
AlphaOut update_alpha_out(AlphaOut prev, CtrlPt ctrl) {
    float T = safe_exp(prev.rgba.a);

    AlphaOut res;
    res.rgba = update_pixel_state(prev.rgba, ctrl.rgba);
    res.distortion_state = update_distortion_state(prev.distortion_state, T, ctrl.density, ctrl.dist);

    res.xyzd = update_xyzd_state(prev.xyzd, ctrl.xyzd, T);

    // 2. Accumulate Aux (Standard Alpha Blending)
    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        res.aux[i] = prev.aux[i] + ctrl.aux[i] * T;
    }

    return res;
}

AlphaOut undo_alpha_out(AlphaOut curr, CtrlPt ctrl) {
    float4 pix_state = undo_pixel_state(curr.rgba, ctrl.rgba);
    float T = safe_exp(pix_state.a);
    AlphaOut res;
    res.rgba = pix_state;
    res.distortion_state = undo_distortion_state(curr.distortion_state, T, ctrl.density, ctrl.dist);
    res.xyzd = undo_xyzd_state(curr.xyzd, ctrl.xyzd, T);

    // 3. Undo Aux
    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        res.aux[i] = curr.aux[i] - T * ctrl.aux[i];
    }

    return res;
}

[BackwardDerivative(bwd_alpha_blend_interval)]
AlphaOut alpha_blend_interval(
    TensorView<int32_t> sorted_gauss_idx,
    TensorView<int32_t> indices,
    DiffTensorView interval_verts,
    DiffTensorView interval_tet_data,
    TensorView<int32_t> interval_meta,
    DiffTensorView final_pixel_state,
    DiffTensorView final_distortion_state,
    DiffTensorView final_xyzd_state,
    TensorView<int32_t> n_contributors,
    no_diff Camera cam,
    no_diff float ray_jitter_x,
    no_diff float ray_jitter_y,
    uint32_t2 pix_coord,
    uint32_t tile_idx_start,
    uint32_t tile_idx_end)
{
    float2 center_pix_coord = {pix_coord.x + ray_jitter_x, pix_coord.y + ray_jitter_y};
    AlphaOut ao = get_empty_alpha_out();
    [ForceUnroll]
    for (int k = 0; k < AUX_DIM; ++k) {
        ao.aux[k] = 0.0f;
    }
    uint32_t block_size = cam.tile_height * cam.tile_width;
    bool is_inside = (pix_coord.x < cam.W && pix_coord.y < cam.H);
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start) + block_size - 1) / block_size;
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

    // Compute ray_scale: the ratio to convert camera-space z difference to ray distance.
    // For perspective cameras: ray_scale = 1/cos(angle) where angle is from principal axis.
    // This matches what the interp variant computes via ray.ortho.
    float fx = cam.K[0][0];
    float fy = cam.K[1][1];
    float cx = cam.K[0][2];
    float cy = cam.K[1][2];
    float dx = (center_pix_coord.x - cx) / fx;
    float dy = (center_pix_coord.y - cy) / fy;
    float ray_scale = 1.0f / sqrt(1.0f + dx * dx + dy * dy);

    int32_t local_n_contrib = 0;
    int splats_left_to_process = tile_idx_end - tile_idx_start;

    for (int i = 0; i < shared_memory_rounds; i++) {
        AllMemoryBarrierWithGroupSync();
        int splat_pointer_offset = i * block_size + thread_rank;
        if (tile_idx_start + splat_pointer_offset < tile_idx_end) {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            bool valid = coll_id < indices.size(0);
            collected_idx[thread_rank] = valid ? coll_id : 0;

            IntervalSMData sm;
            if (valid) {
                uint meta = interval_meta[coll_id];
                sm.num_sil_verts = meta;
                sm.tet_idx = coll_id;
                sm.density = max(float(interval_tet_data[uint2(coll_id, 0)]), 0.f);
                sm.base_color = float3(
                    interval_tet_data[uint2(coll_id, 1)],
                    interval_tet_data[uint2(coll_id, 2)],
                    interval_tet_data[uint2(coll_id, 3)]);
                sm.grd = float3(
                    interval_tet_data[uint2(coll_id, 4)],
                    interval_tet_data[uint2(coll_id, 5)],
                    interval_tet_data[uint2(coll_id, 6)]);
                [ForceUnroll]
                for (int a = 0; a < AUX_DIM; ++a) {
                    sm.aux[a] = float(interval_tet_data[uint2(coll_id, 7 + a)]);
                }
            } else {
                sm.density = 0;
                sm.num_sil_verts = 0;
                sm.tet_idx = 0;
                sm.base_color = float3(0);
                sm.grd = float3(0);
                [ForceUnroll]
                for (int a = 0; a < AUX_DIM; ++a) sm.aux[a] = 0;
            }
            collected_interval[thread_rank] = sm;
        }
        AllMemoryBarrierWithGroupSync();

        if (thread_active) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++) {
                local_n_contrib++;
                IntervalSMData sm = collected_interval[j];

                if (sm.num_sil_verts == 0 || sm.density <= 0) continue;

                IntervalTetData td;
                td.density = sm.density;
                td.base_color = sm.base_color;
                td.grd = sm.grd;
                [ForceUnroll]
                for (int a = 0; a < AUX_DIM; ++a) td.aux[a] = sm.aux[a];

                CtrlPt tetra_ctrl;
                if (evaluate_tetra_interval(center_pix_coord, sm.tet_idx, sm.num_sil_verts,
                                            interval_verts, td, ray_scale, tetra_ctrl)
                    && isfinite(tetra_ctrl.rgba.a) && tetra_ctrl.rgba.a > 0) {
                    tetra_ctrl.dist.y = max(tetra_ctrl.dist.x, tetra_ctrl.dist.y);

                    AlphaOut new_ao = update_alpha_out(ao, tetra_ctrl);
                    ao = new_ao;
                    if (new_ao.rgba.a < -5.5) {
                        thread_active = false;
                        break;
                    }
                }
            }
        }
        splats_left_to_process -= block_size;
    }

    if (is_inside)
        n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = local_n_contrib;

    return ao;
}

void bwd_alpha_blend_interval(
    TensorView<int32_t> sorted_gauss_idx,
    TensorView<int32_t> indices,
    DiffTensorView interval_verts,
    DiffTensorView interval_tet_data,
    TensorView<int32_t> interval_meta,
    DiffTensorView final_pixel_state,
    DiffTensorView final_distortion_state,
    DiffTensorView final_xyzd_state,
    TensorView<int32_t> n_contributors,
    no_diff Camera cam,
    no_diff float ray_jitter_x,
    no_diff float ray_jitter_y,
    uint32_t2 pix_coord,
    uint32_t tile_idx_start,
    uint32_t tile_idx_end,
    AlphaOut d_ao)
{
    float2 center_pix_coord = {pix_coord.x + ray_jitter_x, pix_coord.y + ray_jitter_y};
    bool is_inside = (pix_coord.x < cam.W && pix_coord.y < cam.H);
    uint32_t block_size = cam.tile_height * cam.tile_width;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);

    int splats_left_to_process = tile_idx_end - tile_idx_start;
    int64_t current_splat_offset = tile_idx_end - tile_idx_start;

    float fx = cam.K[0][0];
    float fy = cam.K[1][1];
    float cx = cam.K[0][2];
    float cy = cam.K[1][2];
    float dx = (center_pix_coord.x - cx) / fx;
    float dy = (center_pix_coord.y - cy) / fy;
    float ray_scale = 1.0f / sqrt(1.0f + dx * dx + dy * dy);

    int32_t n_contrib_fwd;
    AlphaOut ao = get_empty_alpha_out();

    if (is_inside) {
        ao.distortion_state = {
            final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
            final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
            final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
            final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)],
            final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4)]
        };
        ao.rgba = float4(
            final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
            final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
            final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
            final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)]);

        [ForceUnroll]
        for (int i = 0; i < AUX_DIM; ++i) {
            ao.aux[i] = final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4 + i)];
        }

        ao.xyzd = float4(
            final_xyzd_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
            final_xyzd_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
            final_xyzd_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
            final_xyzd_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)]);
        n_contrib_fwd = n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)];
    }

    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    for (int i = 0; i < rounds; i++) {
        AllMemoryBarrierWithGroupSync();
        int progress = i * block_size + thread_rank;
        if (tile_idx_start + progress < tile_idx_end) {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_end - progress - 1]);
            bool valid = coll_id < indices.size(0);
            collected_idx[thread_rank] = valid ? coll_id : 0;

            IntervalSMData sm;
            if (valid) {
                uint meta = interval_meta[coll_id];
                sm.num_sil_verts = meta;
                sm.tet_idx = coll_id;
                sm.density = max(float(interval_tet_data[uint2(coll_id, 0)]), 0.f);
                sm.base_color = float3(
                    interval_tet_data[uint2(coll_id, 1)],
                    interval_tet_data[uint2(coll_id, 2)],
                    interval_tet_data[uint2(coll_id, 3)]);
                sm.grd = float3(
                    interval_tet_data[uint2(coll_id, 4)],
                    interval_tet_data[uint2(coll_id, 5)],
                    interval_tet_data[uint2(coll_id, 6)]);
                [ForceUnroll]
                for (int a = 0; a < AUX_DIM; ++a) {
                    sm.aux[a] = float(interval_tet_data[uint2(coll_id, 7 + a)]);
                }
            } else {
                sm.density = 0;
                sm.num_sil_verts = 0;
                sm.tet_idx = 0;
                sm.base_color = float3(0);
                sm.grd = float3(0);
                [ForceUnroll]
                for (int a = 0; a < AUX_DIM; ++a) sm.aux[a] = 0;
            }
            collected_interval[thread_rank] = sm;
        }
        AllMemoryBarrierWithGroupSync();

        if (is_inside) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++) {
                current_splat_offset--;
                if (current_splat_offset >= n_contrib_fwd || current_splat_offset < 0)
                    continue;

                IntervalSMData sm = collected_interval[j];
                if (sm.num_sil_verts == 0 || sm.density <= 0) continue;

                IntervalTetData td;
                td.density = sm.density;
                td.base_color = sm.base_color;
                td.grd = sm.grd;
                [ForceUnroll]
                for (int a = 0; a < AUX_DIM; ++a) td.aux[a] = sm.aux[a];

                CtrlPt tetra_ctrl;
                if (evaluate_tetra_interval(center_pix_coord, sm.tet_idx, sm.num_sil_verts,
                                            interval_verts, td, ray_scale, tetra_ctrl)
                    && isfinite(tetra_ctrl.rgba.a) && tetra_ctrl.rgba.a > 0) {
                    tetra_ctrl.dist.y = max(tetra_ctrl.dist.x, tetra_ctrl.dist.y);

                    ao = undo_alpha_out(ao, tetra_ctrl);

                    // Autodiff through update_alpha_out
                    DifferentialPair<AlphaOut> dp_ao = diffPair(ao);
                    DifferentialPair<CtrlPt> dp_ctrl = diffPair(tetra_ctrl);
                    bwd_diff(update_alpha_out)(dp_ao, dp_ctrl, d_ao);
                    d_ao = dp_ao.getDifferential();

                    // Autodiff through evaluate_tetra_interval
                    DifferentialPair<IntervalTetData> dp_td = diffPair(td);
                    bwd_diff(evaluate_tetra_interval)(
                        center_pix_coord, sm.tet_idx, sm.num_sil_verts,
                        interval_verts, dp_td, ray_scale, dp_ctrl.d);

                    // Propagate gradients to interval_tet_data
                    IntervalTetData d_td = dp_td.getDifferential();
                    // density gradient
                    interval_tet_data.storeOnce(uint2(sm.tet_idx, 0), d_td.density);
                    // base_color gradient
                    interval_tet_data.storeOnce(uint2(sm.tet_idx, 1), d_td.base_color.x);
                    interval_tet_data.storeOnce(uint2(sm.tet_idx, 2), d_td.base_color.y);
                    interval_tet_data.storeOnce(uint2(sm.tet_idx, 3), d_td.base_color.z);
                    // grd gradient
                    interval_tet_data.storeOnce(uint2(sm.tet_idx, 4), d_td.grd.x);
                    interval_tet_data.storeOnce(uint2(sm.tet_idx, 5), d_td.grd.y);
                    interval_tet_data.storeOnce(uint2(sm.tet_idx, 6), d_td.grd.z);
                    [ForceUnroll]
                    for (int a = 0; a < AUX_DIM; ++a) {
                        interval_tet_data.storeOnce(uint2(sm.tet_idx, 7 + a), d_td.aux[a]);
                    }
                }
            }
        }
        splats_left_to_process -= block_size;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void splat_tiled(TensorView<int32_t> sorted_gauss_idx,
                 TensorView<int32_t> tile_ranges,
                 TensorView<int32_t> indices,
                 DiffTensorView interval_verts,
                 DiffTensorView interval_tet_data,
                 TensorView<int32_t> interval_meta,
                 DiffTensorView output_img,
                 DiffTensorView xyzd_img,
                 DiffTensorView distortion_img,
                 TensorView<int32_t> n_contributors,
                 TensorView<float> ray_jitter,
                 TensorCamera tcam)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    uint32_t2 pix_coord = globalIdx.xy;
    Camera cam = no_diff load_tensor_camera(tcam);

    uint32_t tile_idx = cudaBlockIdx().y * cam.grid_width + cudaBlockIdx().x;
    if (tile_idx >= tile_ranges.size(0)) {
        return;
    }
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));

    float rjx = 0;
    float rjy = 0;
    if (is_inside) {
        rjx = no_diff ray_jitter[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), uint32_t(0))];
        rjy = no_diff ray_jitter[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), uint32_t(1))];
    }

    AlphaOut pixel_states = alpha_blend_interval(
        sorted_gauss_idx,
        indices,
        interval_verts,
        interval_tet_data,
        interval_meta,
        output_img,
        distortion_img,
        xyzd_img,
        n_contributors,
        cam,
        rjx,
        rjy,
        pix_coord,
        tile_idx_start,
        tile_idx_end);

    if (is_inside) {
        output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0), pixel_states.rgba.r);
        output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1), pixel_states.rgba.g);
        output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2), pixel_states.rgba.b);
        output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3), pixel_states.rgba.a);
        [ForceUnroll]
        for (int i = 0; i < AUX_DIM; ++i) {
            output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4 + i), pixel_states.aux[i]);
        }

        xyzd_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0), pixel_states.xyzd.x);
        xyzd_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1), pixel_states.xyzd.y);
        xyzd_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2), pixel_states.xyzd.z);
        xyzd_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3), pixel_states.xyzd.a);

        distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0), pixel_states.distortion_state.x);
        distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1), pixel_states.distortion_state.y);
        distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2), pixel_states.distortion_state.z);
        distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3), pixel_states.distortion_state.w);
        distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4), pixel_states.distortion_state.v);
    }
}
