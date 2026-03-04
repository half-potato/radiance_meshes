// Copyright 2025 Alexander Mai
// Fast backward pass variant: uses WaveActiveSum to reduce atomic contention
// during gradient scatter. Forward is identical to alphablend_shader_interp.

import intersect;
import utils;
import camera;
import safe_math;
import interp_version;

static const uint TILE_HEIGHT = PYTHON_TILE_HEIGHT;
static const uint TILE_WIDTH = PYTHON_TILE_WIDTH;

groupshared SplatTetrahedra_interp collected_splats[TILE_HEIGHT * TILE_WIDTH];
groupshared uint32_t collected_idx[TILE_HEIGHT * TILE_WIDTH];

struct DistortionState5 {
    float x;
    float y;
    float z;
    float w;
    float v;
};

struct AlphaOut_nd {
    float4 rgba;
    DistortionState5 distortion_state;
    float4 xyzd;
    float aux[AUX_DIM];
};

// ---- Non-differentiable helpers (same math as interp version) ----

float nd_safe_exp(float x) {
    return exp(clamp(x, -80.f, 80.f));
}

float4 nd_update_pixel_state(float4 pixel_state_t_nm1, float4 gauss_rgba_t_n)
{
    float transmittance_t_nm1 = nd_safe_exp(pixel_state_t_nm1.a);
    float3 color_t_n = pixel_state_t_nm1.rgb + gauss_rgba_t_n.rgb * transmittance_t_nm1;
    float log_transmittance_t_n = pixel_state_t_nm1.a - gauss_rgba_t_n.a;
    return float4(color_t_n, log_transmittance_t_n);
}

float4 nd_undo_pixel_state(float4 pixel_state_t_n, float4 gauss_rgba_t_n)
{
    float log_transmittance_t_nm1 = pixel_state_t_n.a + gauss_rgba_t_n.a;
    float transmittance_t_nm1 = nd_safe_exp(log_transmittance_t_nm1);
    float3 color_t_nm1 = pixel_state_t_n.rgb - gauss_rgba_t_n.rgb * transmittance_t_nm1;
    return float4(color_t_nm1, log_transmittance_t_nm1);
}

float nd_moment_0_int(float a, float b, float s1, float T) { return s1; }
float nd_moment_1_int(float a, float b, float s1, float T) { return s1 * (b+a)/2; }
float nd_self_dist(float a, float b, float s, float T) {
    float dt = max(b - a, 0);
    return s*s*dt/3;
}

DistortionState5 nd_update_distortion_state(
    DistortionState5 old_state, float T, float density, float2 dist)
{
    float start = dist.x;
    float end = max(dist.x, dist.y);
    float d = density;
    float m0 = nd_moment_0_int(start, end, d, T);
    float m1 = nd_moment_1_int(start, end, d, T);

    DistortionState5 new_state;
    new_state.x = old_state.x + 2.0f * m1 * old_state.z;
    new_state.y = old_state.y + 2.0f * m0 * old_state.w;
    new_state.z = old_state.z + m0;
    new_state.w = old_state.w + m1;
    new_state.v = old_state.v + nd_self_dist(start, end, d, T);
    return new_state;
}

DistortionState5 nd_undo_distortion_state(
    DistortionState5 new_state, float T, float density, float2 dist)
{
    float start = dist.x;
    float end = max(dist.x, dist.y);
    float d = density;
    float m0 = nd_moment_0_int(start, end, d, T);
    float m1 = nd_moment_1_int(start, end, d, T);

    DistortionState5 old_state;
    old_state.z = new_state.z - m0;
    old_state.w = new_state.w - m1;
    old_state.x = new_state.x - 2.0f * m1 * old_state.z;
    old_state.y = new_state.y - 2.0f * m0 * old_state.w;
    old_state.v = new_state.v - nd_self_dist(start, end, d, T);
    return old_state;
}

float4 nd_update_xyzd_state(in float4 prev_xyzd, in float4 ctrl_xyzd, in float T) {
    return prev_xyzd + T * ctrl_xyzd;
}

float4 nd_undo_xyzd_state(in float4 next_xyzd, in float4 ctrl_xyzd, in float T) {
    return next_xyzd - T * ctrl_xyzd;
}

AlphaOut_nd nd_get_empty_alpha_out() {
    AlphaOut_nd ao;
    ao.rgba = float4(0.f);
    ao.distortion_state = {0.f, 0.f, 0.f, 0.f, 0.f};
    ao.xyzd = float4(0.f);
    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        ao.aux[i] = 0.f;
    }
    return ao;
}

AlphaOut_nd nd_update_alpha_out(AlphaOut_nd prev, CtrlPt ctrl) {
    float T = nd_safe_exp(prev.rgba.a);

    AlphaOut_nd res;
    res.rgba = nd_update_pixel_state(prev.rgba, ctrl.rgba);
    res.distortion_state = nd_update_distortion_state(prev.distortion_state, T, ctrl.density, ctrl.dist);
    res.xyzd = nd_update_xyzd_state(prev.xyzd, ctrl.xyzd, T);

    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        res.aux[i] = prev.aux[i] + ctrl.aux[i] * T;
    }
    return res;
}

AlphaOut_nd nd_undo_alpha_out(AlphaOut_nd curr, CtrlPt ctrl) {
    float4 pix_state = nd_undo_pixel_state(curr.rgba, ctrl.rgba);
    float T = nd_safe_exp(pix_state.a);

    AlphaOut_nd res;
    res.rgba = pix_state;
    res.distortion_state = nd_undo_distortion_state(curr.distortion_state, T, ctrl.density, ctrl.dist);
    res.xyzd = nd_undo_xyzd_state(curr.xyzd, ctrl.xyzd, T);

    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        res.aux[i] = curr.aux[i] - T * ctrl.aux[i];
    }
    return res;
}

// Non-differentiable tet loader using plain TensorView<float>
SplatTetrahedra_interp nd_load_tet(
    int32_t g_idx,
    TensorView<float> vertices,
    TensorView<int32_t> indices,
    TensorView<float> cell_values)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet = load_tetrahedra(vertices, virtual_tet);

    float density = max(float(cell_values[g_idx, 0]), 0.f);
    SplatTetrahedra_interp result;
    result.vtet = virtual_tet;
    result.tet = tet;
    result.density = density;
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
    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        result.aux[i] = float(cell_values[g_idx, 7 + i]);
    }
    return result;
}

// =====================================================================
// FORWARD KERNEL — identical logic to splat_tiled in interp version
// =====================================================================
[AutoPyBindCUDA]
[CUDAKernel]
void splat_tiled_fast_fwd(
    TensorView<int32_t> sorted_gauss_idx,
    TensorView<int32_t> tile_ranges,
    TensorView<int32_t> indices,
    TensorView<float> vertices,
    TensorView<float> cell_values,
    TensorView<float> output_img,
    TensorView<float> xyzd_img,
    TensorView<float> distortion_img,
    TensorView<int32_t> n_contributors,
    TensorView<bool> tet_alive,
    TensorView<float> ray_jitter,
    TensorCamera tcam)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    uint32_t2 pix_coord = globalIdx.xy;
    Camera cam = load_tensor_camera(tcam);

    uint32_t tile_idx = cudaBlockIdx().y * cam.grid_width + cudaBlockIdx().x;
    if (tile_idx >= tile_ranges.size(0)) return;

    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < uint32_t(cam.W) && pix_coord.y < uint32_t(cam.H));

    float rjx = 0;
    float rjy = 0;
    if (is_inside) {
        rjx = ray_jitter[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), uint32_t(0))];
        rjy = ray_jitter[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), uint32_t(1))];
    }

    float2 center_pix_coord = {pix_coord.x + rjx, pix_coord.y + rjy};
    AlphaOut_nd ao = nd_get_empty_alpha_out();
    uint32_t block_size = cam.tile_height * cam.tile_width;
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start) + block_size - 1) / block_size;
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

    Ray ray = get_ray(cam, center_pix_coord);
    int32_t local_n_contrib = 0;
    int splats_left_to_process = tile_idx_end - tile_idx_start;
    for (int i = 0; i < shared_memory_rounds; i++)
    {
        AllMemoryBarrierWithGroupSync();
        int splat_pointer_offset = i * block_size + thread_rank;
        if (tile_idx_start + splat_pointer_offset < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            bool valid = coll_id < indices.size(0);
            collected_idx[thread_rank] = valid ? coll_id : 0;
            SplatTetrahedra_interp g = nd_load_tet(collected_idx[thread_rank], vertices, indices, cell_values);
            if (!valid) g.density = 0;
            collected_splats[thread_rank] = g;
        }
        AllMemoryBarrierWithGroupSync();
        if (thread_active) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                local_n_contrib++;
                SplatTetrahedra_interp g = collected_splats[j];
                CtrlPt tetra_ctrl;
                if (evaluate_tetra_interp(g, ray, cam.min_t, tetra_ctrl) && isfinite(tetra_ctrl.rgba.a) && tetra_ctrl.rgba.a > 0) {
                    tetra_ctrl.dist.y = max(tetra_ctrl.dist.x, tetra_ctrl.dist.y);
                    AlphaOut_nd new_ao = nd_update_alpha_out(ao, tetra_ctrl);
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

    if (is_inside) {
        n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = local_n_contrib;
        output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = ao.rgba.r;
        output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)] = ao.rgba.g;
        output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)] = ao.rgba.b;
        output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)] = ao.rgba.a;
        [ForceUnroll]
        for (int i = 0; i < AUX_DIM; ++i) {
            output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4 + i)] = ao.aux[i];
        }
        xyzd_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = ao.xyzd.x;
        xyzd_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)] = ao.xyzd.y;
        xyzd_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)] = ao.xyzd.z;
        xyzd_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)] = ao.xyzd.a;
        distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = ao.distortion_state.x;
        distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)] = ao.distortion_state.y;
        distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)] = ao.distortion_state.z;
        distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)] = ao.distortion_state.w;
        distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4)] = ao.distortion_state.v;
    }
}


// =====================================================================
// BACKWARD KERNEL — wave-reduced gradient scatter
// =====================================================================

// We still use Slang's autodiff for per-pixel gradient math (update_alpha_out,
// evaluate_tetra_interp). The differentiable versions live in interp_version.slang
// and alphablend_shader_interp.slang. We import them via the existing imports.

// Differentiable wrappers needed for bwd_diff calls
struct DistortionState5_diff : IDifferentiable {
    float x; float y; float z; float w; float v;
};

[Differentiable]
float diff_moment_0_int(float a, float b, float s1, float T) { return s1; }
[Differentiable]
float diff_moment_1_int(float a, float b, float s1, float T) { return s1 * (b+a)/2; }
[Differentiable]
float diff_self_dist(float a, float b, float s, float T) {
    float dt = max(b - a, 0);
    return s*s*dt/3;
}

[Differentiable]
DistortionState5_diff diff_update_distortion_state(
    DistortionState5_diff old_state, float T, float density, float2 dist)
{
    float start = dist.x;
    float end = max(dist.x, dist.y);
    float d = density;
    float m0 = diff_moment_0_int(start, end, d, T);
    float m1 = diff_moment_1_int(start, end, d, T);

    DistortionState5_diff new_state;
    new_state.x = old_state.x + 2.0f * m1 * old_state.z;
    new_state.y = old_state.y + 2.0f * m0 * old_state.w;
    new_state.z = old_state.z + m0;
    new_state.w = old_state.w + m1;
    new_state.v = old_state.v + diff_self_dist(start, end, d, T);
    return new_state;
}

struct AlphaOut_diff: IDifferentiable {
    float4 rgba;
    DistortionState5_diff distortion_state;
    float4 xyzd;
    float aux[AUX_DIM];
};

[Differentiable]
float4 diff_update_pixel_state(float4 pixel_state_t_nm1, float4 gauss_rgba_t_n)
{
    float transmittance_t_nm1 = safe_exp(pixel_state_t_nm1.a);
    float3 color_t_n = pixel_state_t_nm1.rgb + gauss_rgba_t_n.rgb * transmittance_t_nm1;
    float log_transmittance_t_n = pixel_state_t_nm1.a - gauss_rgba_t_n.a;
    return float4(color_t_n, log_transmittance_t_n);
}

[Differentiable]
float4 diff_update_xyzd_state(in float4 prev_xyzd, in float4 ctrl_xyzd, in float T) {
    return prev_xyzd + T * ctrl_xyzd;
}

[Differentiable]
AlphaOut_diff diff_update_alpha_out(AlphaOut_diff prev, CtrlPt ctrl) {
    float T = safe_exp(prev.rgba.a);

    AlphaOut_diff res;
    res.rgba = diff_update_pixel_state(prev.rgba, ctrl.rgba);
    res.distortion_state = diff_update_distortion_state(prev.distortion_state, T, ctrl.density, ctrl.dist);
    res.xyzd = diff_update_xyzd_state(prev.xyzd, ctrl.xyzd, T);

    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        res.aux[i] = prev.aux[i] + ctrl.aux[i] * detach(T);
    }
    return res;
}

[AutoPyBindCUDA]
[CUDAKernel]
void splat_tiled_fast_bwd(
    TensorView<int32_t> sorted_gauss_idx,
    TensorView<int32_t> tile_ranges,
    TensorView<int32_t> indices,
    TensorView<float> vertices,
    TensorView<float> cell_values,
    TensorView<float> output_img,
    TensorView<float> xyzd_img,
    TensorView<float> distortion_img,
    TensorView<int32_t> n_contributors,
    TensorView<bool> tet_alive,
    TensorView<float> ray_jitter,
    TensorView<float> grad_output_img,
    TensorView<float> grad_xyzd_img,
    TensorView<float> grad_distortion_img,
    TensorView<float> vertices_grad,
    TensorView<float> cell_values_grad,
    TensorCamera tcam)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    uint32_t2 pix_coord = globalIdx.xy;
    Camera cam = load_tensor_camera(tcam);

    uint32_t tile_idx = cudaBlockIdx().y * cam.grid_width + cudaBlockIdx().x;
    if (tile_idx >= tile_ranges.size(0)) return;

    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < uint32_t(cam.W) && pix_coord.y < uint32_t(cam.H));
    uint32_t block_size = cam.tile_height * cam.tile_width;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);

    int splats_left_to_process = tile_idx_end - tile_idx_start;
    int64_t current_splat_offset = tile_idx_end - tile_idx_start;

    int32_t n_contrib_fwd = 0;

    // Load final state from forward pass
    AlphaOut_diff ao;
    ao.rgba = float4(0.f);
    ao.distortion_state = {0.f, 0.f, 0.f, 0.f, 0.f};
    ao.xyzd = float4(0.f);
    [ForceUnroll]
    for (int k = 0; k < AUX_DIM; ++k) ao.aux[k] = 0.f;

    // Load incoming gradient (d_loss/d_output)
    AlphaOut_diff.Differential d_ao;
    d_ao.rgba = float4(0.f);
    d_ao.distortion_state = {0.f, 0.f, 0.f, 0.f, 0.f};
    d_ao.xyzd = float4(0.f);
    [ForceUnroll]
    for (int k = 0; k < AUX_DIM; ++k) d_ao.aux[k] = 0.f;

    float rjx = 0;
    float rjy = 0;

    if (is_inside) {
        rjx = ray_jitter[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), uint32_t(0))];
        rjy = ray_jitter[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), uint32_t(1))];

        ao.distortion_state = {
            distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
            distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
            distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
            distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)],
            distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4)]
        };
        ao.rgba = float4(
            output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
            output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
            output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
            output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)]
        );
        [ForceUnroll]
        for (int i = 0; i < AUX_DIM; ++i) {
            ao.aux[i] = output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4 + i)];
        }
        ao.xyzd = float4(
            xyzd_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
            xyzd_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
            xyzd_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
            xyzd_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)]
        );
        n_contrib_fwd = n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)];

        // Incoming gradient
        d_ao.rgba = float4(
            grad_output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
            grad_output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
            grad_output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
            grad_output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)]
        );
        [ForceUnroll]
        for (int i = 0; i < AUX_DIM; ++i) {
            d_ao.aux[i] = grad_output_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4 + i)];
        }
        d_ao.xyzd = float4(
            grad_xyzd_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
            grad_xyzd_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
            grad_xyzd_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
            grad_xyzd_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)]
        );
        d_ao.distortion_state = {
            grad_distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
            grad_distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
            grad_distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
            grad_distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)],
            grad_distortion_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4)]
        };
    }

    float2 center_pix_coord = {pix_coord.x + rjx, pix_coord.y + rjy};
    Ray ray = get_ray(cam, center_pix_coord);
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

    for (int i = 0; i < rounds; i++)
    {
        AllMemoryBarrierWithGroupSync();
        int progress = i * block_size + thread_rank;
        if (tile_idx_start + progress < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_end - progress - 1]);
            bool valid = coll_id < indices.size(0);
            collected_idx[thread_rank] = valid ? coll_id : 0;
            SplatTetrahedra_interp splat = nd_load_tet(collected_idx[thread_rank], vertices, indices, cell_values);
            if (!valid) splat.density = 0;
            collected_splats[thread_rank] = splat;
        }
        AllMemoryBarrierWithGroupSync();

        for (int j = 0; j < min(block_size, splats_left_to_process); j++)
        {
            current_splat_offset--;

            // All threads participate in the loop for wave convergence.
            // Threads that skip still contribute 0 to the wave sum.
            bool this_thread_active = is_inside &&
                current_splat_offset < n_contrib_fwd &&
                current_splat_offset >= 0;

            uint32_t g_idx = collected_idx[j];
            SplatTetrahedra_interp g = collected_splats[j];

            // Per-thread gradient accumulators for this tet (zero by default)
            // Vertex grads: 4 vertices x 3 components = 12 floats
            float v_grad[12];
            [ForceUnroll] for (int k = 0; k < 12; ++k) v_grad[k] = 0.f;
            // Cell value grads: 7 + AUX_DIM floats
            float cv_grad[7 + AUX_DIM];
            [ForceUnroll] for (int k = 0; k < 7 + AUX_DIM; ++k) cv_grad[k] = 0.f;

            if (this_thread_active) {
                CtrlPt tetra_ctrl;
                if (evaluate_tetra_interp(g, ray, cam.min_t, tetra_ctrl) && isfinite(tetra_ctrl.rgba.a) && tetra_ctrl.rgba.a > 0) {
                    tetra_ctrl.dist.y = max(tetra_ctrl.dist.x, tetra_ctrl.dist.y);

                    // Undo forward state to get prev state
                    // Use the non-diff undo on the diff struct by converting
                    float4 pix_state = nd_undo_pixel_state(float4(ao.rgba), float4(tetra_ctrl.rgba));
                    float T_prev = nd_safe_exp(pix_state.a);

                    AlphaOut_diff prev_ao;
                    prev_ao.rgba = pix_state;
                    prev_ao.distortion_state.x = ao.distortion_state.x;
                    prev_ao.distortion_state.y = ao.distortion_state.y;
                    prev_ao.distortion_state.z = ao.distortion_state.z;
                    prev_ao.distortion_state.w = ao.distortion_state.w;
                    prev_ao.distortion_state.v = ao.distortion_state.v;
                    {
                        float start = tetra_ctrl.dist.x;
                        float end = max(tetra_ctrl.dist.x, tetra_ctrl.dist.y);
                        float d = tetra_ctrl.density;
                        float m0 = d;
                        float m1 = d * (end+start)/2;
                        prev_ao.distortion_state.z = ao.distortion_state.z - m0;
                        prev_ao.distortion_state.w = ao.distortion_state.w - m1;
                        prev_ao.distortion_state.x = ao.distortion_state.x - 2.0f * m1 * prev_ao.distortion_state.z;
                        prev_ao.distortion_state.y = ao.distortion_state.y - 2.0f * m0 * prev_ao.distortion_state.w;
                        float dt_d = max(end - start, 0);
                        prev_ao.distortion_state.v = ao.distortion_state.v - d*d*dt_d/3;
                    }
                    prev_ao.xyzd = ao.xyzd - T_prev * tetra_ctrl.xyzd;
                    [ForceUnroll]
                    for (int k = 0; k < AUX_DIM; ++k) {
                        prev_ao.aux[k] = ao.aux[k] - T_prev * tetra_ctrl.aux[k];
                    }

                    // Use autodiff for update_alpha_out backward
                    DifferentialPair<AlphaOut_diff> dp_ao = diffPair(prev_ao);
                    DifferentialPair<CtrlPt> dp_tetra_ctrl = diffPair(tetra_ctrl);

                    bwd_diff(diff_update_alpha_out)(dp_ao, dp_tetra_ctrl, d_ao);
                    d_ao = dp_ao.getDifferential();

                    // Use autodiff for evaluate_tetra_interp backward
                    CtrlPt.Differential d_tetra_scale = {
                        dp_tetra_ctrl.d.rgba,
                        dp_tetra_ctrl.d.dist,
                        dp_tetra_ctrl.d.density,
                        dp_tetra_ctrl.d.xyzd,
                        dp_tetra_ctrl.d.aux
                    };

                    DifferentialPair<SplatTetrahedra_interp> dp_g = diffPair(g);
                    DifferentialPair<Ray> dp_ray = diffPair(ray);
                    bwd_diff(evaluate_tetra_interp)(dp_g, dp_ray, cam.min_t, d_tetra_scale);

                    // Extract gradients from dp_g.d into flat arrays
                    SplatTetrahedra_interp.Differential dg = dp_g.d;

                    // Vertex gradients (4 verts x 3 components)
                    v_grad[0]  = dg.tet.verts[0].x;
                    v_grad[1]  = dg.tet.verts[0].y;
                    v_grad[2]  = dg.tet.verts[0].z;
                    v_grad[3]  = dg.tet.verts[1].x;
                    v_grad[4]  = dg.tet.verts[1].y;
                    v_grad[5]  = dg.tet.verts[1].z;
                    v_grad[6]  = dg.tet.verts[2].x;
                    v_grad[7]  = dg.tet.verts[2].y;
                    v_grad[8]  = dg.tet.verts[2].z;
                    v_grad[9]  = dg.tet.verts[3].x;
                    v_grad[10] = dg.tet.verts[3].y;
                    v_grad[11] = dg.tet.verts[3].z;

                    // Cell value gradients
                    // density (index 0) — handle max(density, 0) clamp
                    float raw_density = cell_values[g_idx, 0];
                    cv_grad[0] = (raw_density > 0.f) ? dg.density : 0.f;
                    // base_color (1-3)
                    cv_grad[1] = dg.base_color.x;
                    cv_grad[2] = dg.base_color.y;
                    cv_grad[3] = dg.base_color.z;
                    // grd (4-6)
                    cv_grad[4] = dg.grd.x;
                    cv_grad[5] = dg.grd.y;
                    cv_grad[6] = dg.grd.z;
                    // aux (7+)
                    [ForceUnroll]
                    for (int k = 0; k < AUX_DIM; ++k) {
                        cv_grad[7 + k] = dg.aux[k];
                    }

                    // Update ao to prev state for next iteration
                    ao = prev_ao;
                }
            }

            // =========================================================
            // WAVE-REDUCED SCATTER: WaveActiveSum reduces within
            // each warp (32 threads on NVIDIA). One thread per warp
            // writes the reduced gradient. WaveIsFirstLane() is NOT
            // used because Slang's subgroup size may differ from the
            // warp size, causing multiple "first lanes" per warp.
            // =========================================================
            static const uint WARP_SIZE = 32;

            // Reduce vertex gradients within each warp
            [ForceUnroll]
            for (int k = 0; k < 12; ++k) {
                v_grad[k] = WaveActiveSum(v_grad[k]);
            }
            // Reduce cell value gradients within each warp
            [ForceUnroll]
            for (int k = 0; k < 7 + AUX_DIM; ++k) {
                cv_grad[k] = WaveActiveSum(cv_grad[k]);
            }

            // First thread in each warp writes the reduced gradient
            if (thread_rank % WARP_SIZE == 0) {
                float old_val;
                uint4 vtet = g.vtet;

                // Scatter vertex grads
                [ForceUnroll]
                for (int vi = 0; vi < 4; ++vi) {
                    uint vidx = vtet[vi];
                    vertices_grad.InterlockedAdd(uint2(vidx, 0), v_grad[vi*3 + 0], old_val);
                    vertices_grad.InterlockedAdd(uint2(vidx, 1), v_grad[vi*3 + 1], old_val);
                    vertices_grad.InterlockedAdd(uint2(vidx, 2), v_grad[vi*3 + 2], old_val);
                }

                // Scatter cell value grads
                [ForceUnroll]
                for (int ci = 0; ci < 7 + AUX_DIM; ++ci) {
                    cell_values_grad.InterlockedAdd(uint2(g_idx, ci), cv_grad[ci], old_val);
                }
            }
        }
        splats_left_to_process -= block_size;
    }
}
