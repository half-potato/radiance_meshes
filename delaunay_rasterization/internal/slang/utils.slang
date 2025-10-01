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

import intersect;
import safe_math;

static const float FLT_MAX = 1e20;
static const float FLT_MIN = -1e20;
static const float eps = 1e-7;

struct SplatTetrahedra : IDifferentiable
{
    Tetrahedra tet;
    float4 rgbs;
};

[Differentiable]
float read_t1_float(uint32_t idx, DiffTensorView t1)
{
    return t1[uint2(idx, 0)];
}

float3 read_t3_float3(uint32_t idx, TensorView<float> t3)
{
    return float3(t3[uint2(idx, 0)],
                  t3[uint2(idx, 1)],
                  t3[uint2(idx, 2)]);
}

[Differentiable]
float3 read_t3_float3(uint32_t idx, DiffTensorView t3)
{
    return float3(t3[uint2(idx, 0)],
                  t3[uint2(idx, 1)],
                  t3[uint2(idx, 2)]);
}

[Differentiable]
float4 read_t4_float4(uint32_t idx, DiffTensorView t4)
{
    return float4(t4[uint2(idx, 0)],
                  t4[uint2(idx, 1)],
                  t4[uint2(idx, 2)],
                  t4[uint2(idx, 3)]);
}

float4 read_t4_float4(uint32_t idx, TensorView<float> t4)
{
    return float4(t4[uint2(idx, 0)],
                  t4[uint2(idx, 1)],
                  t4[uint2(idx, 2)],
                  t4[uint2(idx, 3)]);
}

[Differentiable]
float4 read_t2x2_float4(uint32_t idx, DiffTensorView t22)
{
    return float4(t22[uint3(idx, 0, 0)],
                  t22[uint3(idx, 1, 0)],
                  t22[uint3(idx, 0, 1)],
                  t22[uint3(idx, 1, 1)]);
}

[Differentiable]
float2x2 read_t2x2_float2x2(uint32_t idx, DiffTensorView t2x2)
{
    return float2x2(t2x2[uint3(idx, 0, 0)],
                    t2x2[uint3(idx, 1, 0)],
                    t2x2[uint3(idx, 0, 1)],
                    t2x2[uint3(idx, 1, 1)]);
}


float clip(float val, float min_val, float max_val)
{
    return max(min_val, min(max_val, val));
}


uint4 load_virtual_tetrahedra(int32_t g_idx,
                              TensorView<int32_t> indices)
{
    return { 
        indices[uint2(g_idx, 0)],
        indices[uint2(g_idx, 1)],
        indices[uint2(g_idx, 2)],
        indices[uint2(g_idx, 3)]
    };
}

[Differentiable]
Tetrahedra load_tetrahedra(DiffTensorView vertices,
                           uint4 virtual_tetra)
{
    float3[4] tetra = {
        read_t3_float3(virtual_tetra.x, vertices),
        read_t3_float3(virtual_tetra.y, vertices),
        read_t3_float3(virtual_tetra.z, vertices),
        read_t3_float3(virtual_tetra.w, vertices),
    };
    return { tetra };
}

Tetrahedra load_tetrahedra(TensorView<float> vertices,
                           uint4 virtual_tetra)
{
    float3[4] tetra = {
        read_t3_float3(virtual_tetra.x, vertices),
        read_t3_float3(virtual_tetra.y, vertices),
        read_t3_float3(virtual_tetra.z, vertices),
        read_t3_float3(virtual_tetra.w, vertices),
    };
    return { tetra };
}

[Differentiable]
SplatTetrahedra load_tet_alphablend(int32_t g_idx,
                                    DiffTensorView vertices,
                                    TensorView<int32_t> indices,
                                    DiffTensorView rgbs)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet = load_tetrahedra(vertices, virtual_tet);
    return {
        tet,
        read_t4_float4(g_idx, rgbs)
    };
}

SplatTetrahedra load_tet_alphablend(int32_t g_idx,
                                    TensorView<float> vertices,
                                    TensorView<int32_t> indices,
                                    TensorView<float> rgbs)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet = load_tetrahedra(vertices, virtual_tet);
    return {
        tet,
        read_t4_float4(g_idx, rgbs)
    };
}

struct CtrlPt: IDifferentiable {
    float4 rgba;
    float2 dist;
    float density;
};

[Differentiable]
CtrlPt evaluate_tetra(in SplatTetrahedra g, 
                      in float3 ray_o,
                      in float3 ray_d)
{
    // float2 dist = ray_tetrahedron_intersect(ray_o, ray_d, g.tet);
    // float2 dist = ray_tetrahedron_intersect(ray_o, ray_d, g.tet);
    float2 dist;
    // bool hit = cyrusBeckTetrahedronShearScale(
    bool hit = ray_tetrahedron_intersect_fused(
        ray_o, ray_d, float4x3(g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]), dist);
    float intersection_t = (dist.x + dist.y)/2.f;
    float dt = max(abs(dist.y - dist.x), 0.0f);
    // float2 dist = cyrusBeckTetrahedronShearScale(ray_o, ray_d, g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]);
    if (!hit) {
        return {{0.f, 0.f, 0.f, 0.f}, dist, g.rgbs.w};
    } else {
        float alpha = safe_clip(1-safe_exp(-g.rgbs.w * dt), 0, 1);
        float4 rgba = {
            alpha * g.rgbs.x, alpha * g.rgbs.y, alpha * g.rgbs.z, alpha};
        return {rgba, dist, g.rgbs.w};
    }
}