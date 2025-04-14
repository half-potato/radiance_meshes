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
// float moment_0_int(float a, float b, float s1) {
//     return safe_div((-safe_expm1(s1*(a - b))),s1);
// }
// [Differentiable]
// float moment_1_int(float a, float b, float s1) {
//     if (s1 < 1e-6) {
//         return 0.5 * max(b * b - a * a, 0);
//     } else {
//         // return (a*s1 - (b*s1 + 1)*safe_exp4alpha(s1*(a - b)) + 1) / (s1*s1);
//         return max(-safe_div(safe_expm1(s1*(a - b)), s1*s1) + (a - b*safe_exp(s1*(a - b))) / s1, 0);
//     }
// }
// [Differentiable]
// float self_dist(float a, float b, float s, float T) {
//     float dt = (a - b);
//     float ds = s * dt;
//     if (abs(ds) < 1e-4) {
//         return T * T * max(-1 / 3 * pow(dt, 3), 0);
//     } else {
//         return T*T*max(safe_div(( 2 * s * (a - b) * exp(ds) - safe_expm1(2 * ds)),
//                 (s * s * s)), 0);
//     }
// }

[Differentiable]
float moment_0_int(float a, float b, float s1) {
    return s1;
}
[Differentiable]
float moment_1_int(float a, float b, float s1) {
    return s1 * (b+a)/2;
}
[Differentiable]
float self_dist(float a, float b, float s, float T) {
    float dt = (a - b);
    return s*s*dt/3;

}

[Differentiable]
DistortionState5 update_distortion_state(
    DistortionState5 old_state,
    float T,
    float density,
    float2 dist,
    no_diff float pre_multi,
    no_diff float ladder_p)
{
    // float start = tukey_power_ladder(dist.x*pre_multi, ladder_p);
    // float end = tukey_power_ladder(dist.y*pre_multi, ladder_p);
    float start = dist.x;//tukey_power_ladder(dist.x*pre_multi, ladder_p);
    float end = dist.y;  // tukey_power_ladder(dist.y*pre_multi, ladder_p);
    float scaling = 1;//no_diff scale_dist((dist.x+dist.y)*0.5);
    float d = density;

    float m0 = T * moment_0_int(start, end, d);
    float m1 = T * moment_1_int(start, end, d);

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
    float2 dist,
    float pre_multi,
    float ladder_p)
{
    // float start = tukey_power_ladder(dist.x*pre_multi, ladder_p);
    // float end = tukey_power_ladder(dist.y*pre_multi, ladder_p);
    // float start = dist.x;
    // float end = dist.y;
    float start = dist.x;//tukey_power_ladder(dist.x*pre_multi, ladder_p);
    float end = dist.y;//tukey_power_ladder(dist.y*pre_multi, ladder_p);
    // float scaling = scale_dist((dist.x+dist.y)*0.5);
    float scaling = 1;//no_diff scale_dist((dist.x+dist.y)*0.5);
    float d = density;

    float m0 = T * moment_0_int(start, end, d);
    float m1 = T * moment_1_int(start, end, d);

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
AlphaOut update_alpha_out(AlphaOut prev, CtrlPt ctrl, no_diff float pre_multi, no_diff float ladder_p) {
    float T = prev.rgba.a;
    return {
        update_pixel_state(prev.rgba, ctrl.rgba),
        update_distortion_state(prev.distortion_state, T, ctrl.density, ctrl.dist, pre_multi, ladder_p)
    };
}

AlphaOut undo_alpha_out(AlphaOut curr, CtrlPt ctrl, no_diff float pre_multi, no_diff float ladder_p) {
    float4 pix_state = undo_pixel_state(curr.rgba, ctrl.rgba);
    float T = pix_state.a;
    return {
        pix_state,
        undo_distortion_state(curr.distortion_state, T, ctrl.density, ctrl.dist, pre_multi, ladder_p)
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
                   no_diff float ladder_p,
                   no_diff float pre_multi,
                   no_diff float min_t,
                   uint32_t2 pix_coord,
                   uint32_t tile_idx_start,
                   uint32_t tile_idx_end,
                   uint32_t tile_height,
                   uint32_t tile_width,
                   uint32_t H,
                   uint32_t W)
{
    float2 center_pix_coord = {pix_coord.x+0.5, pix_coord.y+0.5};
    float4 curr_pixel_state = float4(0.f, 0.f, 0.f, 1.f);
    DistortionState5 curr_distortion_state = {0.f, 0.f, 0.f, 0.f, 0.f};
    AlphaOut ao = {
        curr_pixel_state, curr_distortion_state
    };
    uint32_t block_size = tile_height * tile_width;
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start) + block_size - 1) / block_size;
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    //bool print = thread_rank == 0;
    bool print = false;

    Ray ray = get_ray(cam, center_pix_coord, min_t);
    // if (print) {
    //     printf("ray_o: %f, %f, %f, ray_d: %f, %f, %f\n", ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z);
    // }

    int32_t local_n_contrib = 0;
    // int32_t low = 0;
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
                if (evaluate_tetra_interp(g, ray.o, ray.d, min_t, tetra_ctrl)) {
                    AlphaOut new_ao = update_alpha_out(ao, tetra_ctrl, pre_multi, ladder_p);
                    ao = new_ao;
                    tet_alive[collected_idx[j]] = true;

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
                     Camera cam,
                     no_diff float ladder_p,
                     no_diff float pre_multi,
                     no_diff float min_t,
                     uint32_t2 pix_coord,
                     uint32_t tile_idx_start,
                     uint32_t tile_idx_end,
                     uint32_t tile_height,
                     uint32_t tile_width,
                     uint32_t H,
                     uint32_t W,
                     AlphaOut d_ao)
{
    float2 center_pix_coord = {pix_coord.x+0.5, pix_coord.y+0.5};
    // Load the final pixel state.
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    uint32_t block_size = tile_height * tile_width;
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

    Ray ray = get_ray(cam, center_pix_coord, min_t);

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
                if (evaluate_tetra_interp(g, ray.o, ray.d, min_t, tetra_ctrl)) {
                    // CtrlPt tetra_ctrl = evaluate_tetra_2D(g, cam, min_t, center_pix_coord);
                    // if (tetra_ctrl.rgba.a < 1 / 255.f) {
                    //     printf("Continue\n");
                    //     continue;
                    // }

                    // Undo pixel state
                    ao = undo_alpha_out(ao, tetra_ctrl, pre_multi, ladder_p);

                    // Back-prop automatically through blending and gaussian evaluation.
                    DifferentialPair<SplatTetrahedra_interp> dp_g = diffPair(g);
                    DifferentialPair<float3> dp_ray_o = diffPair(ray.o);
                    DifferentialPair<float3> dp_ray_d = diffPair(ray.d);
                    DifferentialPair<CtrlPt> dp_tetra_ctrl = diffPair(tetra_ctrl);
                    DifferentialPair<AlphaOut> dp_ao = diffPair(ao);

                    bwd_diff(update_alpha_out)(dp_ao, dp_tetra_ctrl, pre_multi, ladder_p, d_ao);
                    d_ao = dp_ao.getDifferential();
                    bwd_diff(evaluate_tetra_interp)(dp_g, dp_ray_o, dp_ray_d, min_t, dp_tetra_ctrl.d);
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
                 int grid_height,
                 int grid_width,

                 int image_height,
                 int image_width,
                 TensorView<float> world_view_transform,
                 TensorView<float> K,
                 TensorView<float> cam_pos,
                 no_diff float ladder_p,
                 no_diff float pre_multi,
                 no_diff float min_t,
                 no_diff float fovy,
                 no_diff float fovx,
                 int tile_height,
                 int tile_width)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint32_t2 pix_coord = globalIdx.xy;

    uint32_t tile_idx = cudaBlockIdx().y * grid_width + cudaBlockIdx().x;
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));

    Camera cam = no_diff load_camera(world_view_transform, K, cam_pos, fovy, fovx, image_height, image_width);

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
        ladder_p, pre_multi,
        min_t,
        pix_coord,
        tile_idx_start,
        tile_idx_end,
        tile_height,
        tile_width,
        output_img.size(0),
        output_img.size(1));
                                    
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

void kern_calc_tet_err(TensorView<int32_t> sorted_gauss_idx,
                  TensorView<int32_t> indices,
                  TensorView<float> vertices,
                  TensorView<float> tet_density,
                  TensorView<float> final_pixel_state,
                  TensorView<float> tet_err,
                  TensorView<int32_t> tet_count,
                  TensorView<int32_t> n_contributors,
                  Camera cam,
                  no_diff float min_t,
                  uint32_t2 pix_coord,
                  uint32_t tile_idx_start,
                  uint32_t tile_idx_end,
                  uint32_t tile_height,
                  uint32_t tile_width,
                  uint32_t H,
                  uint32_t W,
                  float pixel_err)
{
    float2 center_pix_coord = {pix_coord.x+0.5, pix_coord.y+0.5};
    // Load the final pixel state.
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    uint32_t block_size = tile_height * tile_width;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);

    int splats_left_to_process = tile_idx_end - tile_idx_start;
    int64_t current_splat_offset = tile_idx_end - tile_idx_start;

    float4 current_pixel_state;
    int32_t n_contrib_fwd;
    if (is_inside) {
        current_pixel_state = float4(final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)]);
        n_contrib_fwd = n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)];
    }

    Ray ray = get_ray(cam, center_pix_coord, min_t);

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
                if (evaluate_tetra_interp(g, ray.o, ray.d, min_t, tetra_ctrl)) {
                    // CtrlPt tetra_ctrl = evaluate_tetra_2D(g, cam, min_t, center_pix_coord);
                    if (tetra_ctrl.rgba.a < 1/255.f) continue;

                    // Undo pixel state
                    current_pixel_state = undo_pixel_state(current_pixel_state, tetra_ctrl.rgba);

                    float err = current_pixel_state.a * tetra_ctrl.rgba.a * pixel_err;
                    float surface = tetra_ctrl.dist.x;
                    float midpt = 0.5 * (tetra_ctrl.dist.x + tetra_ctrl.dist.y);
                    float contrib = tetra_ctrl.rgba.a * surface + (1 - tetra_ctrl.rgba.a) * midpt;
                    float3 contrib_pt = contrib * ray.d + ray.o;
                    // printf("(%i: %f, %f, %f)", g_idx, current_pixel_state.a, tetra_ctrl.rgba.a, pixel_err);
                    float old_val;
                    int32_t old_ival;
                    tet_err.InterlockedAdd(uint2(g_idx, 0), err * contrib_pt.x, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 1), err * contrib_pt.y, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 2), err * contrib_pt.z, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 3), err, old_val);
                    tet_count.InterlockedAdd(g_idx, 1, old_ival);
                }
            }
        }
        splats_left_to_process -= block_size;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void calc_tet_err(TensorView<int32_t> sorted_gauss_idx,
                 TensorView<int32_t> tile_ranges,
                 TensorView<int32_t> indices,
                 TensorView<float> vertices,
                 TensorView<float> tet_density,
                 TensorView<float> output_img,
                 TensorView<float> pixel_err,
                 TensorView<float> tet_err,
                 TensorView<int32_t> tet_count,
                 TensorView<int32_t> n_contributors,
                 int grid_height,
                 int grid_width,

                 int image_height,
                 int image_width,
                 TensorView<float> world_view_transform,
                 TensorView<float> K,
                 TensorView<float> cam_pos,
                 no_diff float min_t,
                 no_diff float fovy,
                 no_diff float fovx,
                 int tile_height,
                 int tile_width)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint32_t2 pix_coord = globalIdx.xy;

    uint32_t tile_idx = cudaBlockIdx().y * grid_width + cudaBlockIdx().x;
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));

    Camera cam = load_camera(world_view_transform, K, cam_pos, fovy, fovx, image_height, image_width);

    float this_pixel_err = 0;
    if (is_inside) {
        this_pixel_err = pixel_err[uint2(uint32_t(pix_coord.y), uint32_t(pix_coord.x))];
    }

    kern_calc_tet_err(
        sorted_gauss_idx,
        indices,
        vertices,
        tet_density,
        output_img,
        tet_err,
        tet_count,
        n_contributors,
        cam,
        min_t,
        pix_coord,
        tile_idx_start,
        tile_idx_end,
        tile_height,
        tile_width,
        output_img.size(0),
        output_img.size(1), 
        this_pixel_err);
}