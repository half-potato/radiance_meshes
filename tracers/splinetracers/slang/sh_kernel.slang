// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import safe_math;
import tensor_util;
import sh;

[AutoPyBindCUDA]
[CUDAKernel]
void sh_kernel(
    TensorView<float> means,
    TensorView<float> features,
    TensorView<float> ray_origin,
    TensorView<float> colors,
    uint sh_degree)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint prim_ind = dispatchIdx.x;
    float3 rayo = {ray_origin[0], ray_origin[1], ray_origin[2]};

    if (prim_ind >= means.size(0)) {
        return;
    }
    float3 mean = get_float3(means, prim_ind);
    // First, check
    float3 dir_pp = normalize(mean - rayo);
    Features feat = get_feats(features, prim_ind, sh_degree);
    float3 color = eval_color(dir_pp, feat, sh_degree);
    colors[prim_ind, 0] = color.x;
    colors[prim_ind, 1] = color.y;
    colors[prim_ind, 2] = color.z;
}

[AutoPyBindCUDA]
[CUDAKernel]
void bw_sh_kernel(
    TensorView<float> means,
    TensorView<float> features,
    TensorView<float> dL_dfeatures,
    TensorView<float> ray_origin,
    TensorView<float> dL_dcolors,
    uint sh_degree)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint prim_ind = dispatchIdx.x;
    float3 rayo = {ray_origin[0], ray_origin[1], ray_origin[2]};

    if (prim_ind >= means.size(0)) {
        return;
    }

    float3 deriv_color = get_float3(dL_dcolors, prim_ind);
    // skip if gradient is tiny
    if ((abs(deriv_color.x) < 1e-8) && (abs(deriv_color.y) < 1e-8) && (abs(deriv_color.z) < 1e-8)) {
        return;
    }

    float3 mean = get_float3(means, prim_ind);
    float3 dir_pp = normalize(mean - rayo);
    Features feat = get_feats(features, prim_ind, sh_degree);

    var deriv_direction = diffPair(dir_pp, {});
    var d_feat = diffPair(feat, {});
    bwd_diff(eval_color)(deriv_direction, d_feat, sh_degree, deriv_color);

    add_float3(dL_dfeatures, prim_ind, 0u, d_feat.d.f0);
    if (sh_degree > 0) {
        add_float3(dL_dfeatures, prim_ind, 1u, d_feat.d.f1);
        add_float3(dL_dfeatures, prim_ind, 2u, d_feat.d.f2);
        add_float3(dL_dfeatures, prim_ind, 3u, d_feat.d.f3);
        if (sh_degree > 1) {
            add_float3(dL_dfeatures, prim_ind, 4u, d_feat.d.f4);
            add_float3(dL_dfeatures, prim_ind, 5u, d_feat.d.f5);
            add_float3(dL_dfeatures, prim_ind, 6u, d_feat.d.f6);
            add_float3(dL_dfeatures, prim_ind, 7u, d_feat.d.f7);
            add_float3(dL_dfeatures, prim_ind, 8u, d_feat.d.f8);
            if (sh_degree > 2) {
                add_float3(dL_dfeatures, prim_ind, 9u, d_feat.d.f9);
                add_float3(dL_dfeatures, prim_ind, 10u, d_feat.d.f10);
                add_float3(dL_dfeatures, prim_ind, 11u, d_feat.d.f11);
                add_float3(dL_dfeatures, prim_ind, 12u, d_feat.d.f12);
                add_float3(dL_dfeatures, prim_ind, 13u, d_feat.d.f13);
                add_float3(dL_dfeatures, prim_ind, 14u, d_feat.d.f14);
                add_float3(dL_dfeatures, prim_ind, 15u, d_feat.d.f15);
            }
        }
    }
}
