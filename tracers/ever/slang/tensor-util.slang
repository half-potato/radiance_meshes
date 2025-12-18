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

import sh;
import sh_half;

float3 get_float3(TensorView<float> view, uint ind) {
    return {
        view[ind, 0],
        view[ind, 1],
        view[ind, 2]
    };
}

half3 get_half3(TensorView<half> view, uint ind) {
    return {
        view[ind, 0],
        view[ind, 1],
        view[ind, 2]
    };
}

half3 get_half3(TensorView<half> view, uint ind, uint feat_ind) {
    return {
        view[ind, feat_ind, 0],
        view[ind, feat_ind, 1],
        view[ind, feat_ind, 2]
    };
}

float3 get_float3(TensorView<float> view, uint ind, uint feat_ind) {
    return {
        view[ind, feat_ind, 0],
        view[ind, feat_ind, 1],
        view[ind, feat_ind, 2]
    };
}

float4 get_float4(TensorView<float> view, uint ind) {
    return {
        view[ind, 0],
        view[ind, 1],
        view[ind, 2],
        view[ind, 3]
    };
}

void atomic_add_float3(TensorView<float> view, uint ind, uint feat_ind, float3 val) {
    float temp;
    view.InterlockedAdd(uint3(ind, feat_ind, 0u), val.x, temp);
    view.InterlockedAdd(uint3(ind, feat_ind, 1u), val.y, temp);
    view.InterlockedAdd(uint3(ind, feat_ind, 2u), val.z, temp);
}

void add_half3(TensorView<half> view, uint ind, uint feat_ind, half3 val) {
    view[uint3(ind, feat_ind, 0u)] += val.x;
    view[uint3(ind, feat_ind, 1u)] += val.y;
    view[uint3(ind, feat_ind, 2u)] += val.z;
}

void add_float3(TensorView<float> view, uint ind, uint feat_ind, float3 val) {
    view[uint3(ind, feat_ind, 0u)] += val.x;
    view[uint3(ind, feat_ind, 1u)] += val.y;
    view[uint3(ind, feat_ind, 2u)] += val.z;
}

void atomic_add_float2(TensorView<float> view, uint ind, float2 val) {
    float temp;
    view.InterlockedAdd(uint2(ind, 0u), val.x, temp);
    view.InterlockedAdd(uint2(ind, 1u), val.y, temp);
}

void add_float3(TensorView<float> view, uint ind, float3 val) {
    view[uint2(ind, 0u)] += val.x;
    view[uint2(ind, 1u)] += val.y;
    view[uint2(ind, 2u)] += val.z;
}

void atomic_add_float3(TensorView<float> view, uint ind, float3 val) {
    float temp;
    view.InterlockedAdd(uint2(ind, 0u), val.x, temp);
    view.InterlockedAdd(uint2(ind, 1u), val.y, temp);
    view.InterlockedAdd(uint2(ind, 2u), val.z, temp);
}

void add_float4(TensorView<float> view, uint ind, float4 val) {
    view[uint2(ind, 0u)] += val.x;
    view[uint2(ind, 1u)] += val.y;
    view[uint2(ind, 2u)] += val.z;
    view[uint2(ind, 3u)] += val.w;
}

void atomic_add_float4(TensorView<float> view, uint ind, float4 val) {
    float temp;
    view.InterlockedAdd(uint2(ind, 0u), val.x, temp);
    view.InterlockedAdd(uint2(ind, 1u), val.y, temp);
    view.InterlockedAdd(uint2(ind, 2u), val.z, temp);
    view.InterlockedAdd(uint2(ind, 3u), val.w, temp);
}


Features get_feats(in TensorView<float> features, in uint prim_ind, in uint sh_degree) {
    Features feat;
    feat.f0 = get_float3(features, prim_ind, 0);
    if (sh_degree > 0) {
        feat.f1 = get_float3(features, prim_ind, 1);
        feat.f2 = get_float3(features, prim_ind, 2);
        feat.f3 = get_float3(features, prim_ind, 3);
        if (sh_degree > 1) {
            feat.f4 = get_float3(features, prim_ind, 4);
            feat.f5 = get_float3(features, prim_ind, 5);
            feat.f6 = get_float3(features, prim_ind, 6);
            feat.f7 = get_float3(features, prim_ind, 7);
            feat.f8 = get_float3(features, prim_ind, 8);
            if (sh_degree > 2) {
                feat.f9 = get_float3(features, prim_ind, 9);
                feat.f10 = get_float3(features, prim_ind, 10);
                feat.f11 = get_float3(features, prim_ind, 11);
                feat.f12 = get_float3(features, prim_ind, 12);
                feat.f13 = get_float3(features, prim_ind, 13);
                feat.f14 = get_float3(features, prim_ind, 14);
                feat.f15 = get_float3(features, prim_ind, 15);
            }
        }
    }
    return feat;
}

HalfFeatures half_get_feats(in TensorView<half> features, in uint prim_ind, in uint sh_degree) {
    HalfFeatures feat;
    feat.f0 = get_half3(features, prim_ind, 0);
    if (sh_degree > 0) {
        feat.f1 = get_half3(features, prim_ind, 1);
        feat.f2 = get_half3(features, prim_ind, 2);
        feat.f3 = get_half3(features, prim_ind, 3);
        if (sh_degree > 1) {
            feat.f4 = get_half3(features, prim_ind, 4);
            feat.f5 = get_half3(features, prim_ind, 5);
            feat.f6 = get_half3(features, prim_ind, 6);
            feat.f7 = get_half3(features, prim_ind, 7);
            feat.f8 = get_half3(features, prim_ind, 8);
            if (sh_degree > 2) {
                feat.f9 = get_half3(features, prim_ind, 9);
                feat.f10 = get_half3(features, prim_ind, 10);
                feat.f11 = get_half3(features, prim_ind, 11);
                feat.f12 = get_half3(features, prim_ind, 12);
                feat.f13 = get_half3(features, prim_ind, 13);
                feat.f14 = get_half3(features, prim_ind, 14);
                feat.f15 = get_half3(features, prim_ind, 15);
            }
        }
    }
    return feat;
}
