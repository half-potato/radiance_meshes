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

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <optix.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "structs.h"

extern unsigned char ptx_code_file[];
extern unsigned char ptx_code_file2[];
extern unsigned char fast_ptx_code_file[];

struct RayGenData
{
    // No data needed
};
struct MissData
{
    float3 bg_color;
};
typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;

struct HitGroupData {
};
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

struct Params
{
    StructuredBuffer<uchar4> image;
    StructuredBuffer<float4> fimage;
    StructuredBuffer<uint> iters;
    StructuredBuffer<uint> last_face;
    StructuredBuffer<uint> touch_count;
    StructuredBuffer<float4> last_dirac;
    StructuredBuffer<SplineState> last_state;
    StructuredBuffer<int> tri_collection;
    StructuredBuffer<float3> ray_origins;
    StructuredBuffer<float3> ray_directions;
    Cam camera;

    StructuredBuffer<__half> half_attribs;

    StructuredBuffer<float3> means;
    StructuredBuffer<float3> scales;
    StructuredBuffer<float4> quats;
    StructuredBuffer<float> densities;
    StructuredBuffer<float> features;

    size_t sh_degree;
    size_t max_iters;
    float tmin;
    float tmax;
    StructuredBuffer<float4> initial_drgb;
    float max_prim_size;
    OptixTraversableHandle handle;
};

class Forward {
   public:
    Forward() = default;
    Forward(
        const OptixDeviceContext &context,
        int8_t device,
        const Primitives &model,
        const bool enable_backward);
    ~Forward() noexcept(false);
    void trace_rays(const OptixTraversableHandle &handle,
                    const size_t num_rays,
                    float3 *ray_origins,
                    float3 *ray_directions,
                    void *image_out,
                    uint sh_degree,
                    float tmin,
                    float tmax,
                    float4 *initial_drgb,
                    Cam *camera=NULL,
                    const size_t max_iters=10000,
                    const float max_prim_size=3,
                    uint *iters=NULL,
                    uint *last_face=NULL,
                    uint *touch_count=NULL,
                    float4 *last_dirac=NULL,
                    SplineState *last_state=NULL,
                    int *tri_collection=NULL,
                    int *d_touch_count=NULL,
                    int *d_touch_inds=NULL);
   void reset_features(const Primitives &model);
   bool enable_backward = false;
   size_t num_prims = 0;
   private:
    Params params;
    // Context, streams, and accel structures are inherited
    OptixDeviceContext context = nullptr;
    int8_t device = -1;
    const Primitives *model;
    // Local fields used for this pipeline
    OptixModule module = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixPipeline pipeline = nullptr;
    CUdeviceptr d_param = 0;
    CUstream stream = nullptr;
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    float eps = 1e-6;
    static std::string load_ptx_data2() {
        return std::string((char *)ptx_code_file2);
    }
    static std::string load_ptx_data() {
        return std::string((char *)ptx_code_file);
    }
    static std::string load_fast_ptx_data() {
        return std::string((char *)fast_ptx_code_file);
    }
};
