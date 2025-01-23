// Copyright 2024 Google LLC
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

//TODO: Clean-up shared-memory logic, too complicated, could be simplified.

import utils;
import safe_math;

static const uint TILE_HEIGHT = PYTHON_TILE_HEIGHT;
static const uint TILE_WIDTH = PYTHON_TILE_WIDTH;

groupshared SplatTetrahedra collected_splats[TILE_HEIGHT * TILE_WIDTH];
groupshared uint32_t collected_idx[TILE_HEIGHT * TILE_WIDTH];

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

[BackwardDerivative(bwd_alpha_blend)] // Use a custom derivative so that we can hand-write the structure of the reverse loop
float4 alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                   TensorView<int32_t> indices,
                   DiffTensorView vertices,
                   DiffTensorView rgbs,
                   DiffTensorView final_pixel_state,
                   TensorView<int32_t> n_contributors,
                   no_diff Camera cam,
                   uint32_t2 pix_coord,
                   uint32_t tile_idx_start,
                   uint32_t tile_idx_end,
                   uint32_t tile_height,
                   uint32_t tile_width,
                   uint32_t H,
                   uint32_t W)
{
    float2 center_pix_coord = pix_coord;
    float4 curr_pixel_state = float4(0.f, 0.f, 0.f, 1.f);
    uint32_t block_size = tile_height * tile_width;
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start) + block_size - 1) / block_size;
    if (shared_memory_rounds > 10000) {
        printf("%i,", shared_memory_rounds);
    }
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    //bool print = thread_rank == 0;
    bool print = false;

    Ray ray = get_ray(cam, center_pix_coord);
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
            if (tile_idx_start + splat_pointer_offset >= sorted_gauss_idx.size(0)) {
                printf("WTF2");
            }
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            if (coll_id >= indices.size(0)) {
                printf("WTF");
            }
            collected_splats[thread_rank] = load_tet_alphablend(coll_id, vertices, indices, rgbs);
        }
        AllMemoryBarrierWithGroupSync();
        if (thread_active) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                local_n_contrib++;
                SplatTetrahedra g = collected_splats[j];
                float4 tetra_rgba = evaluate_tetra(g, ray.o, ray.d);
                // if (tetra_rgba.w < 1e-2) {
                //     low++;
                // }
                if (tetra_rgba.w < 1/255.f) continue;
                // if (print) {
                //     printf("tetra rgba: %f, %f, %f, %f\n", tetra_rgba.x, tetra_rgba.y, tetra_rgba.z, tetra_rgba.w);
                // }

                float4 new_pixel_state = update_pixel_state(curr_pixel_state, tetra_rgba);
                curr_pixel_state = new_pixel_state;
                if (new_pixel_state.a < 1/255.f) {
                    // This Splat never registred so we subtract it before we break.
                    // local_n_contrib--;
                    thread_active = false;
                    break;
                }
            }
        }
        splats_left_to_process -= block_size;
    }

    if (is_inside)
        n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = local_n_contrib;
    // printf("%i/%i,", low, local_n_contrib);

    return curr_pixel_state;
}

void bwd_alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                     TensorView<int32_t> indices,
                     DiffTensorView vertices,
                     DiffTensorView rgbs,
                     DiffTensorView final_pixel_state,
                     TensorView<int32_t> n_contributors,
                     Camera cam,
                     uint32_t2 pix_coord,
                     uint32_t tile_idx_start,
                     uint32_t tile_idx_end,
                     uint32_t tile_height,
                     uint32_t tile_width,
                     uint32_t H,
                     uint32_t W,
                     float4 d_current_pixel_state)
{
    // Load the final pixel state.
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    uint32_t block_size = tile_height * tile_width;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);

    bool print = false;//pix_coord.x == 0 && pix_coord.y == 0;

    int splats_left_to_process = tile_idx_end - tile_idx_start;
    uint32_t current_splat_offset = tile_idx_end - tile_idx_start;

    float4 current_pixel_state;
    int32_t n_contrib_fwd;
    if (is_inside) {
        current_pixel_state = float4(final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)]);
        n_contrib_fwd = n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)];
    }

    float2 center_pix_coord = pix_coord;

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
            collected_splats[thread_rank] = load_tet_alphablend(coll_id, vertices, indices, rgbs);
        }
        AllMemoryBarrierWithGroupSync();
        if (is_inside) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                current_splat_offset--;
                if (current_splat_offset >= n_contrib_fwd)
                    continue;
                uint32_t g_idx = collected_idx[j];
                SplatTetrahedra g = collected_splats[j];

                float4 tetra_rgba = evaluate_tetra(g, ray.o, ray.d);
                if (tetra_rgba.w < 1/255.f) continue;

                // Undo pixel state
                // if (print) {
                //     printf("1st Pixel State: %f, %f, %f, %f\n", current_pixel_state.x, current_pixel_state.y, current_pixel_state.z, current_pixel_state.w);
                // }
                current_pixel_state = undo_pixel_state(current_pixel_state, tetra_rgba);

                // Back-prop automatically through blending and gaussian evaluation.
                DifferentialPair<SplatTetrahedra> dp_g = diffPair(g);
                DifferentialPair<float3> dp_ray_o = diffPair(ray.o);
                DifferentialPair<float3> dp_ray_d = diffPair(ray.d);
                DifferentialPair<float4> dp_tetra_rgba = diffPair(tetra_rgba);
                DifferentialPair<float4> dp_current_pixel_state = diffPair(current_pixel_state);

                bwd_diff(update_pixel_state)(dp_current_pixel_state, dp_tetra_rgba, d_current_pixel_state);
                // if (print) {
                //     printf("Pixel State: %f, %f, %f, %f\n", dp_current_pixel_state.p.x, dp_current_pixel_state.p.y, dp_current_pixel_state.p.z, dp_current_pixel_state.p.w);
                //     printf("Tetra RGBA: %f, %f, %f, %f\n", tetra_rgba.x, tetra_rgba.y, tetra_rgba.z, tetra_rgba.w);
                //     printf("D Pixel State: %f, %f, %f, %f\n", dp_current_pixel_state.d.x, dp_current_pixel_state.d.y, dp_current_pixel_state.d.z, dp_current_pixel_state.d.w);
                //     printf("D RGBA: %f, %f, %f, %f\n", dp_tetra_rgba.d.x, dp_tetra_rgba.d.y, dp_tetra_rgba.d.z, dp_tetra_rgba.d.w);
                // }
                d_current_pixel_state = dp_current_pixel_state.getDifferential();
                bwd_diff(evaluate_tetra)(dp_g, dp_ray_o, dp_ray_d, dp_tetra_rgba.d);
                // if (print) {
                //     printf("D V0: %f, %f, %f\n", dp_g.d.tet.verts[0].x, dp_g.d.tet.verts[0].y, dp_g.d.tet.verts[0].z);
                //     printf("D V1: %f, %f, %f\n", dp_g.d.tet.verts[1].x, dp_g.d.tet.verts[1].y, dp_g.d.tet.verts[1].z);
                //     printf("D V2: %f, %f, %f\n", dp_g.d.tet.verts[2].x, dp_g.d.tet.verts[2].y, dp_g.d.tet.verts[2].z);
                //     printf("D V3: %f, %f, %f\n", dp_g.d.tet.verts[3].x, dp_g.d.tet.verts[3].y, dp_g.d.tet.verts[3].z);
                // }
                bwd_diff(load_tet_alphablend)(g_idx, vertices, indices, rgbs, dp_g.d);
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
                 DiffTensorView rgbs,
                 DiffTensorView output_img,
                 TensorView<int32_t> n_contributors,
                 int grid_height,
                 int grid_width,

                 int image_height,
                 int image_width,
                 TensorView<float> world_view_transform,
                 TensorView<float> K,
                 TensorView<float> cam_pos,
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

    float4 pixel_state = alpha_blend(sorted_gauss_idx,
                                     indices,
                                     vertices,
                                     rgbs,
                                     output_img,
                                     n_contributors,
                                     cam,
                                     pix_coord,
                                     tile_idx_start,
                                     tile_idx_end,
                                     tile_height,
                                     tile_width,
                                     output_img.size(0),
                                     output_img.size(1));
                                    
    if (is_inside) {
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0), pixel_state.r);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1), pixel_state.g);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2), pixel_state.b);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3), pixel_state.a);
    }
}
