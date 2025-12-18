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

extern unsigned char ptx_code_file[];

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
    StructuredBuffer<int>  start_tet_ids;

    StructuredBuffer<int2> side_index;
    StructuredBuffer<float> densities;
    StructuredBuffer<float> features;

    size_t max_iters;
    float tmin, tmax;
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
    Forward(const Forward &) = delete;
    Forward &operator=(const Forward &) = delete;
    Forward(Forward &&other) noexcept;
    Forward &operator=(Forward &&other) {
        using std::swap;
        if (this != &other) {
            Forward tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    ~Forward() noexcept(false);
    friend void swap(Forward &first, Forward &second) {
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
                    void *image_out,
                    float tmin,
                    float tmax,
                    const size_t max_iters=10000,
                    uint *iters=NULL,
                    uint *last_face=NULL,
                    uint *touch_count=NULL,
                    float4 *last_dirac=NULL,
                    SplineState *last_state=NULL,
                    int *tri_collection=NULL,
                    int *start_tet_ids=NULL);
   void reset_features(const Primitives &model);
   bool enable_backward = false;
   size_t num_prims = 0;
   private:
    Params params;
    // Context, streams, and accel structures are inherited
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
        return std::string((char *)ptx_code_file);
    }
};
