// Copyright 2025 Alexander Mai
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import random;
import utils;
import safe_math;
import interp_version;

static const uint TILE_HEIGHT = PYTHON_TILE_HEIGHT;
static const uint TILE_WIDTH = PYTHON_TILE_WIDTH;

groupshared SplatTetrahedra_interp collected_splats[TILE_HEIGHT * TILE_WIDTH];
groupshared uint32_t collected_idx[TILE_HEIGHT * TILE_WIDTH];

struct DistortionState5 : IDifferentiable {
    float x;  // old_state.x
    float y;  // old_state.y
    float z;  // old_state.z
    float w;  // old_state.w
    float v;  // new 5th term, e.g., for w_i^2 * interval / 3, or anything else
};

[Differentiable]
float tukey_power_ladder(float x, float p) {
  // Compute sign(x) * |p - 1|/p * ((|x|/|p-1| + 1)^p - 1)
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

// [Differentiable]
// float moment_0_int(float a, float b, float s1, float T) {
//     return T * safe_div((-safe_expm1(s1*(a - b))),s1);
// }
// [Differentiable]
// float moment_1_int(float a, float b, float s1, float T) {
//     float s12 = s1*s1;
//     if (s12 < 1e-6) {
//         return T * 0.5 * max(b * b - a * a, 0);
//     } else {
//         // return (a*s1 - (b*s1 + 1)*safe_exp4alpha(s1*(a - b)) + 1) / (s1*s1);
//         return T * max(
//             - safe_div(safe_expm1(s1*(a - b)), s12)
//             + safe_div(a - b*safe_exp(s1*(a - b)), s1), 0);
//     }
// }
// [Differentiable]
// float self_dist(float a, float b, float s, float T) {
//     float dt = (a - b);
//     float ds = s * dt;
//     float s3 = s*s*s;
//     if (abs(ds) < 1e-4 || s3 < 1e-5) {
//         return T * T * max(-1 / 3 * pow(dt, 3), 0);
//     } else {
//         return T * T * max(
//             safe_div(
//                 ( 2 * s * (a - b) * exp(ds) - safe_expm1(2 * ds)),
//                 s3),
//             0);
//     }
// }

[Differentiable]
float moment_0_int(float a, float b, float s1, float T) {
    return s1;
}
[Differentiable]
float moment_1_int(float a, float b, float s1, float T) {
    return s1 * (b+a)/2;
}
[Differentiable]
float self_dist(float a, float b, float s, float T) {
    float dt = max(b - a, 0);
    return s*s*dt/3;
}

[Differentiable]
DistortionState5 update_distortion_state(
    DistortionState5 old_state,
    float T,
    float density,
    float2 dist)
{
    no_diff float start = dist.x;//tukey_power_ladder(dist.x, ladder_p);
    no_diff float end = dist.y;  // tukey_power_ladder(dist.y, ladder_p);
    float scaling = 1;//no_diff scale_dist((dist.x+dist.y)*0.5);
    float d = density;

    float m0 = moment_0_int(start, end, d, T); // just weight
    float m1 = moment_1_int(start, end, d, T);

    DistortionState5 new_state;
    new_state.x = old_state.x + 2.0f * m1 * old_state.z;
    new_state.y = old_state.y + 2.0f * m0 * old_state.w;
    new_state.z = old_state.z + m0;
    new_state.w = old_state.w + m1;
    // new_state.v = old_state.v + T * T * self_dist(start, end, d);
    new_state.v = old_state.v + self_dist(start, end, d, T);
    // printf("%f - %f, d: %f, v: %f T: %f, m0: %f, m1: %f\n", start, end, d, new_state.v, T, m0, m1);

    return new_state;
}

DistortionState5 undo_distortion_state(
    DistortionState5 new_state,
    float T,
    float density,
    float2 dist)
{
    float start = dist.x;
    float end = dist.y;
    float scaling = 1;
    float d = density;

    float m0 = moment_0_int(start, end, d, T);
    float m1 = moment_1_int(start, end, d, T);

    DistortionState5 old_state;
    old_state.z = new_state.z - m0;
    old_state.w = new_state.w - m1;
    old_state.x = new_state.x - 2.0f * m1 * old_state.z;
    old_state.y = new_state.y - 2.0f * m0 * old_state.w;
    old_state.v = new_state.v - self_dist(start, end, d, T);
    return old_state;
}

[Differentiable]
float4 update_pixel_state(float4 pixel_state_t_nm1, float4 gauss_rgba_t_n)
{
    float3 color_t_n = pixel_state_t_nm1.rgb + gauss_rgba_t_n.rgb * pixel_state_t_nm1.a;
    float transmittance_t_n = pixel_state_t_nm1.a * (1 - gauss_rgba_t_n.a);
    return float4(color_t_n, transmittance_t_n);
}

float4 undo_pixel_state(float4 pixel_state_t_n, float4 gauss_rgba_t_n)
{
    float transmittance_t_nm1;
    if (1 - gauss_rgba_t_n.a < 1e-3) {
        transmittance_t_nm1 = 1;
    } else {
        transmittance_t_nm1 = safe_div(pixel_state_t_n.a, (1 - gauss_rgba_t_n.a));
    }
    float3 color_t_nm1 = pixel_state_t_n.rgb - gauss_rgba_t_n.rgb * transmittance_t_nm1;
    return float4(color_t_nm1, transmittance_t_nm1);
}

struct AlphaOut: IDifferentiable {
    float4 rgba;
    DistortionState5 distortion_state;
};

[Differentiable]
AlphaOut update_alpha_out(AlphaOut prev, CtrlPt ctrl) {
    float T = prev.rgba.a;
    return {
        update_pixel_state(prev.rgba, ctrl.rgba),
        update_distortion_state(prev.distortion_state, T, ctrl.density, ctrl.dist)
    };
}

AlphaOut undo_alpha_out(AlphaOut curr, CtrlPt ctrl) {
    float4 pix_state = undo_pixel_state(curr.rgba, ctrl.rgba);
    float T = pix_state.a;
    return {
        pix_state,
        undo_distortion_state(curr.distortion_state, T, ctrl.density, ctrl.dist)
    };
}

[BackwardDerivative(bwd_alpha_blend)] // Use a custom derivative so that we can hand-write the structure of the reverse loop
AlphaOut alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                   TensorView<int32_t> indices,
                   DiffTensorView vertices,
                   DiffTensorView tet_density,
                   DiffTensorView final_pixel_state,
                   DiffTensorView final_distortion_state,
                   TensorView<int32_t> n_contributors,
                   TensorView<bool> tet_alive,
                   no_diff Camera cam,
                   no_diff float ray_jitter_x,
                   no_diff float ray_jitter_y,
                   uint32_t2 pix_coord,
                   uint32_t tile_idx_start,
                   uint32_t tile_idx_end)
{
    float2 center_pix_coord = {pix_coord.x+ray_jitter_x, pix_coord.y+ray_jitter_y};
    float4 curr_pixel_state = float4(0.f, 0.f, 0.f, 1.f);
    DistortionState5 curr_distortion_state = {0.f, 0.f, 0.f, 0.f, 0.f};
    AlphaOut ao = {
        curr_pixel_state, curr_distortion_state
    };
    uint32_t block_size = cam.tile_height * cam.tile_width;
    bool is_inside = (pix_coord.x < cam.W && pix_coord.y < cam.H);
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start) + block_size - 1) / block_size;
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    //bool print = thread_rank == 0;
    bool print = false;


    Ray ray = get_ray(cam, center_pix_coord);
    // if (print) {
    //     printf("ray_o: %f, %f, %f, ray_d: %f, %f, %f\n", ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z);
    // }

    int32_t local_n_contrib = 0;
    // int32_t low = 0;
    int countdown = 1;
    int splats_left_to_process = tile_idx_end - tile_idx_start;
    for (int i = 0; i < shared_memory_rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int splat_pointer_offset = i * block_size + thread_rank;
        if (tile_idx_start + splat_pointer_offset < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_tet_alphablend_interp(coll_id, vertices, indices, tet_density);
        }
        AllMemoryBarrierWithGroupSync();
        if (thread_active) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                local_n_contrib++;
                SplatTetrahedra_interp g = collected_splats[j];
                CtrlPt tetra_ctrl;
                if (evaluate_tetra_interp(g, ray.o, ray.d, cam.min_t, tetra_ctrl) && tetra_ctrl.rgba.a > 0) {
                    AlphaOut new_ao = update_alpha_out(ao, tetra_ctrl);
                    ao = new_ao;
                    // tet_alive[collected_idx[j]] = true;
                    // if (pix_coord.x == 3 && pix_coord.y == 0) {
                    //     printf("%f-%f (%f), (%f, %f, %f, %f, %f)\n", tetra_ctrl.dist.x, tetra_ctrl.dist.y, tetra_ctrl.density,
                    //     new_ao.distortion_state.x, new_ao.distortion_state.y, new_ao.distortion_state.z, new_ao.distortion_state.w, new_ao.distortion_state.v);
                    // }

                    if (new_ao.rgba.a < 1 / 255.f) {
                        // This Splat never registred so we subtract it before we break.
                        // local_n_contrib--;
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
    // printf("%i/%i,", low, local_n_contrib);

    return ao;
}

void bwd_alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                     TensorView<int32_t> indices,
                     DiffTensorView vertices,
                     DiffTensorView tet_density,
                     DiffTensorView final_pixel_state,
                     DiffTensorView final_distortion_state,
                     TensorView<int32_t> n_contributors,
                     TensorView<bool> tet_alive,
                     no_diff Camera cam,
                     no_diff float ray_jitter_x,
                     no_diff float ray_jitter_y,
                     uint32_t2 pix_coord,
                     uint32_t tile_idx_start,
                     uint32_t tile_idx_end,
                     AlphaOut d_ao)
{
    float2 center_pix_coord = {pix_coord.x+ray_jitter_x, pix_coord.y+ray_jitter_y};
    // Load the final pixel state.
    bool is_inside = (pix_coord.x < cam.W && pix_coord.y < cam.H);
    uint32_t block_size = cam.tile_height * cam.tile_width;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);

    bool print = false;//pix_coord.x == 0 && pix_coord.y == 0;

    int splats_left_to_process = tile_idx_end - tile_idx_start;
    int64_t current_splat_offset = tile_idx_end - tile_idx_start;

    float4 current_pixel_state;
    DistortionState5 curr_distortion_state;
    int32_t n_contrib_fwd;
    if (is_inside) {
        curr_distortion_state = {final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
                                 final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
                                 final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
                                 final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)],
                                 final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4)]};
        current_pixel_state = float4(final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)]);
        n_contrib_fwd = n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)];
    }
    AlphaOut ao = {current_pixel_state, curr_distortion_state};

    Ray ray = get_ray(cam, center_pix_coord);

    DifferentialPair<float2> dp_center_pix_coord = diffPair(center_pix_coord);

    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    for (int i = 0; i < rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int progress = i * block_size + thread_rank;
        if (tile_idx_start + progress < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_end - progress - 1]);
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_tet_alphablend_interp(coll_id, vertices, indices, tet_density);
        }
        AllMemoryBarrierWithGroupSync();
        if (is_inside) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                current_splat_offset--;
                if (current_splat_offset >= n_contrib_fwd || current_splat_offset < 0)
                    continue;
                uint32_t g_idx = collected_idx[j];
                SplatTetrahedra_interp g = collected_splats[j];

                CtrlPt tetra_ctrl;
                if (evaluate_tetra_interp(g, ray.o, ray.d, cam.min_t, tetra_ctrl) && tetra_ctrl.rgba.a > 0) {
                    // CtrlPt tetra_ctrl = evaluate_tetra_2D(g, cam, min_t, center_pix_coord);
                    // if (tetra_ctrl.rgba.a < 1 / 255.f) {
                    //     printf("Continue\n");
                    //     continue;
                    // }

                    // Undo pixel state
                    ao = undo_alpha_out(ao, tetra_ctrl);

                    // Back-prop automatically through blending and gaussian evaluation.
                    DifferentialPair<SplatTetrahedra_interp> dp_g = diffPair(g);
                    DifferentialPair<float3> dp_ray_o = diffPair(ray.o);
                    DifferentialPair<float3> dp_ray_d = diffPair(ray.d);
                    DifferentialPair<CtrlPt> dp_tetra_ctrl = diffPair(tetra_ctrl);
                    DifferentialPair<AlphaOut> dp_ao = diffPair(ao);

                    bwd_diff(update_alpha_out)(dp_ao, dp_tetra_ctrl, d_ao);
                    d_ao = dp_ao.getDifferential();
                    float midpt = tetra_ctrl.dist.x;
                    float grad_scale = min(1, midpt*midpt);// / scene_scaling / scene_scaling);
                    //float grad_scale = midpt*midpt / scene_scaling / scene_scaling;
                    CtrlPt.Differential d_tetra_scale = {
                        grad_scale * dp_tetra_ctrl.d.rgba,
                        grad_scale * dp_tetra_ctrl.d.dist,
                        grad_scale * dp_tetra_ctrl.d.density,
                    };
                    bwd_diff(evaluate_tetra_interp)(dp_g, dp_ray_o, dp_ray_d, cam.min_t, d_tetra_scale);
                    // bwd_diff(evaluate_tetra_2D)(dp_g, cam, min_t, dp_center_pix_coord, dp_tetra_ctrl.d);
                    bwd_diff(load_tet_alphablend_interp)(g_idx, vertices, indices, tet_density, dp_g.d);

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
                 DiffTensorView vertices,
                 DiffTensorView tet_density,
                 DiffTensorView output_img,
                 DiffTensorView distortion_img,
                 TensorView<int32_t> n_contributors,
                 TensorView<bool> tet_alive,
                 TensorView<float> ray_jitter,
                 TensorCamera tcam)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint32_t2 pix_coord = globalIdx.xy;
    Camera cam = no_diff load_tensor_camera(tcam);

    uint32_t tile_idx = cudaBlockIdx().y * cam.grid_width + cudaBlockIdx().x;
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));

    float rjx = 0;
    float rjy = 0;
    if (is_inside) {
        rjx = no_diff ray_jitter[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), uint32_t(0))];
        rjy = no_diff ray_jitter[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), uint32_t(1))];
    }

    AlphaOut pixel_states = alpha_blend(
        sorted_gauss_idx,
        indices,
        vertices,
        tet_density,
        output_img,
        distortion_img,
        n_contributors,
        tet_alive,
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
      distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0), pixel_states.distortion_state.x);
      distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1), pixel_states.distortion_state.y);
      distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2), pixel_states.distortion_state.z);
      distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3), pixel_states.distortion_state.w);
      distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4), pixel_states.distortion_state.v);
    }
}

struct VarInfo {
    float S0, S1, S2;
    float2 dist;
    int g_idx;
};

VarInfo kern_calc_tet_err(TensorView<int32_t> sorted_gauss_idx,
                  TensorView<int32_t> indices,
                  TensorView<float> vertices,
                  TensorView<float> tet_density,
                  TensorView<float> final_pixel_state,
                  TensorView<float> tet_err,
                  TensorView<int32_t> tet_count,
                  TensorView<int32_t> n_contributors,
                  Camera cam,
                  uint32_t2 pix_coord,
                  uint32_t tile_idx_start,
                  uint32_t tile_idx_end,
                  float pixel_err,
                  float3 gt_pixel)
{
    uint seed = hash_wang(pix_coord.x);
    seed = hash_wang(seed ^ pix_coord.y);
    seed = hash_wang(seed ^ tile_idx_end);
    seed = hash_wang(seed ^ tile_idx_start);
    RNG rng = {seed};

    float2 center_pix_coord = {pix_coord.x+0.5, pix_coord.y+0.5};
    float4 curr_pixel_state = float4(0.f, 0.f, 0.f, 1.f);
    DistortionState5 curr_distortion_state = {0.f, 0.f, 0.f, 0.f, 0.f};
    AlphaOut ao = {
        curr_pixel_state, curr_distortion_state
    };
    uint32_t block_size = cam.tile_height * cam.tile_width;
    bool is_inside = (pix_coord.x < cam.W && pix_coord.y < cam.H);
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start) + block_size - 1) / block_size;
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    //bool print = thread_rank == 0;
    bool print = false;


    Ray ray = get_ray(cam, center_pix_coord);
    // if (print) {
    //     printf("ray_o: %f, %f, %f, ray_d: %f, %f, %f\n", ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z);
    // }

    float max_split_alpha = 0.f;
    float max_grow_alpha = 0.f;
    VarInfo split_info = {0.f, 0.f, 0.f, -1};

    int32_t local_n_contrib = 0;
    // int32_t low = 0;
    int countdown = 1;
    int splats_left_to_process = tile_idx_end - tile_idx_start;
    for (int i = 0; i < shared_memory_rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int splat_pointer_offset = i * block_size + thread_rank;
        if (tile_idx_start + splat_pointer_offset < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_tet_alphablend_interp(coll_id, vertices, indices, tet_density);
        }
        AllMemoryBarrierWithGroupSync();
        if (thread_active) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                local_n_contrib++;
                SplatTetrahedra_interp g = collected_splats[j];
                uint32_t g_idx = collected_idx[j];
                CtrlPt tetra_ctrl;
                if (evaluate_tetra_interp(g, ray.o, ray.d, cam.min_t, tetra_ctrl)) {
                    //if (tetra_ctrl.rgba.a < 1/255.f) continue;
                    AlphaOut new_ao = update_alpha_out(ao, tetra_ctrl);

                    // calc grad scale to remove floaters
                    float grad_scale = min(1, tetra_ctrl.dist.x*tetra_ctrl.dist.x);// / scene_scaling / scene_scaling);

                    float T = grad_scale * ao.rgba.a * tetra_ctrl.rgba.a;
                    float T_p = grad_scale * ao.rgba.a;

                    // grow and split quantities
                    float err = pixel_err;
                    float3 mismatch_rgb = abs(gt_pixel - tetra_ctrl.rgba.rgb);
                    //float mismatch = grad_scale * (mismatch_rgb.r + mismatch_rgb.g + mismatch_rgb.b) / 3.f;
                    //mismatch *= pixel_err;
                    float mismatch = pixel_err;

                    float old_val;
                    int32_t old_ival;
                    tet_err.InterlockedAdd(uint2(g_idx, 3), T, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 4), 1 * mismatch, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 5), 1 * mismatch * mismatch, old_val);
                    float3 centroid = 0.5*(tetra_ctrl.dist.x+tetra_ctrl.dist.y) * ray.d + ray.o;
                    tet_err.InterlockedAdd(uint2(g_idx, 13), T * centroid.x, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 14), T * centroid.y, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 15), T * centroid.z, old_val);

                    split_info.S0 = T;
                    split_info.S1 = 1 * mismatch;
                    split_info.S2 = 1 * mismatch * mismatch;

                    tet_err.InterlockedAdd(uint2(split_info.g_idx, 0), split_info.S0, old_val);
                    tet_err.InterlockedAdd(uint2(split_info.g_idx, 1), split_info.S1, old_val);
                    tet_err.InterlockedAdd(uint2(split_info.g_idx, 2), split_info.S2, old_val);

                    split_info.g_idx = g_idx;
                    split_info.dist = tetra_ctrl.dist;
                    float w = split_info.S1;
                    float3 p1 = split_info.dist.x * ray.d + ray.o;
                    float3 p2 = split_info.dist.y * ray.d + ray.o;
                    tet_err.InterlockedAdd(uint2(split_info.g_idx, 6), w * p1.x, old_val);
                    tet_err.InterlockedAdd(uint2(split_info.g_idx, 7), w * p1.y, old_val);
                    tet_err.InterlockedAdd(uint2(split_info.g_idx, 8), w * p1.z, old_val);
                    tet_err.InterlockedAdd(uint2(split_info.g_idx, 9), w * p2.x, old_val);
                    tet_err.InterlockedAdd(uint2(split_info.g_idx, 10), w * p2.y, old_val);
                    tet_err.InterlockedAdd(uint2(split_info.g_idx, 11), w * p2.z, old_val);
                    tet_err.InterlockedAdd(uint2(split_info.g_idx, 12), w, old_val);

                    tet_count.InterlockedAdd(g_idx, 1, old_ival);

                    ao = new_ao;
                    if (new_ao.rgba.a < 1 / 255.f) {
                        // This Splat never registred so we subtract it before we break.
                        // local_n_contrib--;
                        thread_active = false;
                        break;
                    }
                }
            }
        }
        splats_left_to_process -= block_size;
    }
    return split_info;
}

[AutoPyBindCUDA]
[CUDAKernel]
void calc_tet_err(TensorView<int32_t> sorted_gauss_idx,
                 TensorView<int32_t> tile_ranges,
                 TensorView<int32_t> indices,
                 TensorView<float> vertices,
                 TensorView<float> tet_density,
                 TensorView<float> output_img,
                 TensorView<float> gt,
                 TensorView<float> pixel_err,
                 TensorView<float> tet_err,
                 TensorView<float> debug_img,
                 TensorView<int32_t> tet_count,
                 TensorView<int32_t> n_contributors,
                 TensorCamera tcam)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint32_t2 pix_coord = globalIdx.xy;

    Camera cam = load_tensor_camera(tcam);
    uint32_t tile_idx = cudaBlockIdx().y * cam.grid_width + cudaBlockIdx().x;
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));


    float this_pixel_err = 0;
    float3 this_gt = {0.f, 0.f, 0.f};
    if (is_inside) {
        this_pixel_err = pixel_err[uint2(uint32_t(pix_coord.y), uint32_t(pix_coord.x))];
        this_gt = {
            gt[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
            gt[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
            gt[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
        };
    }

    VarInfo vi = kern_calc_tet_err(
        sorted_gauss_idx,
        indices,
        vertices,
        tet_density,
        output_img,
        tet_err,
        tet_count,
        n_contributors,
        cam,
        pix_coord,
        tile_idx_start,
        tile_idx_end,
        this_pixel_err,
        this_gt);
    if (is_inside) {
      debug_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = vi.S0;
      debug_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)] = vi.S1;
      debug_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)] = vi.S2;
      debug_img[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)] = (float)vi.g_idx;
    }
}
