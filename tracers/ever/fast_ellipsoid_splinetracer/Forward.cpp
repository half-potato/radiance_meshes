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

#include <optix_stack_size.h>
#include <optix_stubs.h>

#include "cuda_util.h"
#include "Forward.h"
#include "CUDABuffer.h"
#include "initialize_density.h"

void Forward::trace_rays(
    const OptixTraversableHandle &handle,
    const size_t num_rays, float3 *ray_origins,
    float3 *ray_directions, void *image_out, uint sh_deg,
    float tmin, float tmax, float4 *initial_drgb,
    Cam *camera,
    const size_t max_iters, 
    const float max_prim_size,
    uint *iters, uint *last_face,
    uint *touch_count,
    float4 *last_dirac, SplineState *last_state,
    int *tri_collection, int *d_touch_count, int *d_touch_inds) {
  CUDA_CHECK(cudaSetDevice(device));
  {
    params.fimage.data = (float4 *)image_out;
    params.last_state.data = last_state;
    params.last_state.size = num_rays;
    params.last_dirac.data = last_dirac;
    params.last_dirac.size = num_rays;
    params.tri_collection.data = tri_collection;
    params.tri_collection.size = num_rays * max_iters;
    params.iters.data = iters;
    params.iters.size = num_rays;
    params.last_face.data = last_face;
    params.last_face.size = num_rays;
    params.touch_count.data = touch_count;
    params.sh_degree = sh_deg;
    params.max_prim_size = max_prim_size;
    params.max_iters = max_iters;
    params.ray_origins.data = ray_origins;
    params.ray_origins.size = num_rays;
    params.ray_directions.data = ray_directions;
    params.ray_directions.size = num_rays;
    if (camera != NULL) {
      params.camera = *camera;
    }
    params.tmin = tmin;
    params.tmax = tmax;

    CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(initial_drgb), 0,
                          num_rays * sizeof(float4)));
    params.initial_drgb.data = initial_drgb;
    params.initial_drgb.size = num_rays;

    initialize_density(&params, model->aabbs, d_touch_count, d_touch_inds);

    params.handle = handle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_param), &params,
                          sizeof(params), cudaMemcpyHostToDevice));
    if (camera != NULL) {
      OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt,
                              camera->width, camera->height, 1));
    } else {
      OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt,
                              num_rays, 1, 1));
    }
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

Forward::Forward(const OptixDeviceContext &context, int8_t device,
                 const Primitives &model, const bool enable_backward)
    : enable_backward(enable_backward), context(context), device(device), model(&model) {
  // Initialize fields
  OptixPipelineCompileOptions pipeline_compile_options = {};
  // Switch to active device
  CUDA_CHECK(cudaSetDevice(device));
  char log[2048]; // For error reporting from OptiX creation functions
  size_t sizeof_log = sizeof(log);
  //
  // Create module
  //
  {
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 32;
    pipeline_compile_options.numAttributeValues = 1;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName =
        "SLANG_globalParams";
    pipeline_compile_options.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
  }
  OptixModule module = nullptr;
  OptixModule module2 = nullptr;
  {
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    size_t inputSize = 0;
    std::string input;
    if (enable_backward) {
      input = Forward::load_ptx_data();
    } else {
      input = Forward::load_fast_ptx_data();
    }
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        context, &module_compile_options, &pipeline_compile_options,
        input.c_str(), input.size(), log, &sizeof_log, &module));

    input = Forward::load_ptx_data2();
  }
  //
  // Create program groups
  //
  {
    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
    OptixProgramGroupDesc raygen_prog_group_desc = {};   //
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg_float";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log,
                                            &sizeof_log, &raygen_prog_group));
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log,
                                            &sizeof_log, &miss_prog_group));
    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleAH = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    hitgroup_prog_group_desc.hitgroup.moduleIS = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__ellipsoid";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitgroup_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log,
                                            &sizeof_log, &hitgroup_prog_group));
  }
  //
  // Link pipeline
  //
  {
    const uint32_t max_trace_depth = 1;
    OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group,
                                          hitgroup_prog_group};
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OPTIX_CHECK_LOG(optixPipelineCreate(
        context, &pipeline_compile_options, &pipeline_link_options,
        program_groups, sizeof(program_groups) / sizeof(program_groups[0]), log,
        &sizeof_log, &pipeline));
    OptixStackSizes stack_sizes = {};
    for (auto &prog_group : program_groups) {
      OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes, max_trace_depth,
        0, // maxCCDepth
        0, // maxDCDEpth
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        1 // maxTraversableDepth
        ));
  }
  //
  // Set up shader binding table
  //
  {
    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record),
                          raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(raygen_record), &rg_sbt,
                          raygen_record_size, cudaMemcpyHostToDevice));
    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
    MissSbtRecord ms_sbt;
    ms_sbt.data = {0.3f, 0.1f, 0.2f};
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(miss_record), &ms_sbt,
                          miss_record_size, cudaMemcpyHostToDevice));
    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record),
                          hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(hitgroup_record), &hg_sbt,
                          hitgroup_record_size, cudaMemcpyHostToDevice));
    sbt.raygenRecord = raygen_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = 1;
  }

  {

    params.half_attribs.data = model.half_attribs;
    params.half_attribs.size = model.num_prims;

    params.means.data = (float3 *)model.means;
    params.means.size = model.num_prims;
    params.scales.data = (float3 *)model.scales;
    params.scales.size = model.num_prims;
    params.quats.data = (float4 *)model.quats;
    params.quats.size = model.num_prims;

    params.densities.data = (float *)model.densities;
    params.densities.size = model.num_prims;
    params.features.data = model.features;
    params.features.size = model.num_prims * model.feature_size;

    num_prims = model.num_prims;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));
  }
}

void Forward::reset_features(const Primitives &model) {
    params.features.data = model.features;
    params.features.size = model.num_prims * model.feature_size;
}

Forward::~Forward() noexcept(false) {
  if (d_param != 0)
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(d_param, 0))));
  if (sbt.raygenRecord != 0)
    CUDA_CHECK(
        cudaFree(reinterpret_cast<void *>(std::exchange(sbt.raygenRecord, 0))));
  if (sbt.missRecordBase != 0)
    CUDA_CHECK(cudaFree(
        reinterpret_cast<void *>(std::exchange(sbt.missRecordBase, 0))));
  if (sbt.hitgroupRecordBase != 0)
    CUDA_CHECK(cudaFree(
        reinterpret_cast<void *>(std::exchange(sbt.hitgroupRecordBase, 0))));
  if (sbt.callablesRecordBase)
    CUDA_CHECK(cudaFree(
        reinterpret_cast<void *>(std::exchange(sbt.callablesRecordBase, 0))));
  if (sbt.exceptionRecord)
    CUDA_CHECK(cudaFree(
        reinterpret_cast<void *>(std::exchange(sbt.exceptionRecord, 0))));
  sbt = {};
  if (stream != nullptr)
    CUDA_CHECK(cudaStreamDestroy(std::exchange(stream, nullptr)));
  if (pipeline != nullptr)
    OPTIX_CHECK(optixPipelineDestroy(std::exchange(pipeline, nullptr)));
  if (raygen_prog_group != nullptr)
    OPTIX_CHECK(
        optixProgramGroupDestroy(std::exchange(raygen_prog_group, nullptr)));
  if (miss_prog_group != nullptr)
    OPTIX_CHECK(
        optixProgramGroupDestroy(std::exchange(miss_prog_group, nullptr)));
  if (hitgroup_prog_group != nullptr)
    OPTIX_CHECK(
        optixProgramGroupDestroy(std::exchange(hitgroup_prog_group, nullptr)));
  if (module != nullptr)
    OPTIX_CHECK(optixModuleDestroy(std::exchange(module, nullptr)));
}
