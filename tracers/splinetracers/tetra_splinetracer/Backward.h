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

#include "TriangleMesh.h"
#include "structs.h"

extern unsigned char backward_ptx_code_file[];

struct BWRayGenData
{
    // No data needed
};


struct BWMissData
{
    float3 bg_color;
};

struct BWHitGroupData { };

typedef SbtRecord<BWRayGenData>     BWRayGenSbtRecord;
typedef SbtRecord<BWMissData>       BWMissSbtRecord;
typedef SbtRecord<BWHitGroupData> BWHitGroupSbtRecord;


struct BackwardParams
{
    StructuredBuffer<float4> image;
    StructuredBuffer<uint> iters;
    StructuredBuffer<uint> last_face;
    StructuredBuffer<float4> last_dirac;
    StructuredBuffer<SplineState> last_state;
    StructuredBuffer<float3> ray_origins;
    StructuredBuffer<float3> ray_directions;

    StructuredBuffer<float3> means;
    StructuredBuffer<float3> scales;
    StructuredBuffer<float4> quats;
    StructuredBuffer<float> densities;
    StructuredBuffer<float> features;

    // derivative stuff
    StructuredBuffer<float3> dL_dC;
    StructuredBuffer<float3> dL_dmean;
    StructuredBuffer<float3> dL_dscale;
    StructuredBuffer<float4> dL_dquat;
    StructuredBuffer<float> dL_dheight;
    StructuredBuffer<float> dL_dfeature;
    StructuredBuffer<float3> dL_drayo;
    StructuredBuffer<float3> dL_drayd;

    size_t feature_size;

    OptixTraversableHandle handle;
};

class Backward {
   public:
    Backward() = default;
    Backward(
        const OptixDeviceContext &context,
        int8_t device,
        const Primitives &model);
    Backward(const Backward &) = delete;
    Backward &operator=(const Backward &) = delete;
    Backward(Backward &&other) noexcept;
    Backward &operator=(Backward &&other) {
        using std::swap;
        if (this != &other) {
            Backward tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    ~Backward() noexcept(false);

    friend void swap(Backward &first, Backward &second) {
        using std::swap;
        swap(first.context, second.context);
        swap(first.device, second.device);
        swap(first.module, second.module);
        swap(first.sbt, second.sbt);
        swap(first.pipeline, second.pipeline);
        swap(first.d_param, second.d_param);
        swap(first.stream, second.stream);
        swap(first.raygen_prog_group, second.raygen_prog_group);
        swap(first.miss_prog_group, second.miss_prog_group);
        swap(first.hitgroup_prog_group, second.hitgroup_prog_group);
        swap(first.eps, second.eps);
    }

    void trace_rays(const OptixTraversableHandle &handle,
                             const size_t num_rays,
                             float3 *ray_origins,
                             float3 *ray_directions,
                             uint *iters,
                             uint *last_face,
                             float4 *last_dirac,
                             SplineState *last_state,
                             // derivative stuff
                             float3 *dL_dC,
                             float3 *dL_dmean,
                             float3 *dL_dscale,
                             float4 *dL_dquat,
                             float *dL_dheight,
                             float *dL_dfeature,
                             float3 *dL_drayo,
                             float3 *dL_drayd);
   uint32_t feature_size = 3;
   size_t num_prims = 0;
   private:
    // Context, streams, and accel structures are inherited
    BackwardParams params;
    OptixDeviceContext context = nullptr;
    int8_t device = -1;

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

    static std::string load_ptx_data() {
        return std::string((char *)backward_ptx_code_file);
    }
};


