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

#include <optix_stubs.h>
#include "glm/glm.hpp"
#include "GAS.h"
#include <chrono>

#ifndef __DEFINED_OUTPUT_BUFFERS__
CUdeviceptr D_GAS_OUTPUT_BUFFER = 0;
size_t OUTPUT_BUFFER_SIZE = 0;
CUdeviceptr D_TEMP_BUFFER_GAS = 0;
size_t TEMP_BUFFER_SIZE = 0;
CUdeviceptr D_COMPACT_GAS_BUFFER = 0;
size_t COMPACT_GAS_BUFFER_SIZE = 0;
#define __DEFINED_OUTPUT_BUFFERS__ 0
#endif

using namespace std::chrono;

GAS::GAS() noexcept
    : device(-1),
      context(nullptr),
      gas_handle(0) {}


GAS::GAS(GAS &&other) noexcept
    : device(std::exchange(other.device, -1)),
      context(std::exchange(other.context, nullptr)),
      gas_handle(std::exchange(other.gas_handle, 0))
{}

void GAS::release() {
    bool device_set = false;
    gas_handle = 0;
}

GAS::~GAS() noexcept(false) {
    if (this->device != -1) {
        release();
    }
    const auto device = std::exchange(this->device, -1);
}

void GAS::build(const Primitives &model) {
    auto start = high_resolution_clock::now();
    auto full_start = high_resolution_clock::now();
    release();

    CUDA_CHECK(cudaSetDevice(device));
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    uint32_t aabb_input_flags[1];
    aabb_input_flags[0] = OPTIX_GEOMETRY_FLAG_NONE;

    CUdeviceptr d_aabbs = (CUdeviceptr)model.aabbs;
    OptixBuildInput aabb_input = {};
    aabb_input.type                        = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabbs;
    aabb_input.customPrimitiveArray.numPrimitives = model.num_prims;

    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                context,
                &accel_options,
                &aabb_input,
                1, // Number of build inputs
                &gas_buffer_sizes
                ) );

    // Handle allocation of the GAS
    if (OUTPUT_BUFFER_SIZE <= gas_buffer_sizes.outputSizeInBytes) {
        if (D_GAS_OUTPUT_BUFFER != 0) {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( D_GAS_OUTPUT_BUFFER ) ) );
        }
        OUTPUT_BUFFER_SIZE = size_t(gas_buffer_sizes.outputSizeInBytes);
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &D_GAS_OUTPUT_BUFFER ),
                    OUTPUT_BUFFER_SIZE
                    ) );
    }

    if (TEMP_BUFFER_SIZE <= gas_buffer_sizes.tempSizeInBytes) {
        if (D_TEMP_BUFFER_GAS != 0) {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( D_TEMP_BUFFER_GAS ) ) );
        }
        TEMP_BUFFER_SIZE = size_t(gas_buffer_sizes.tempSizeInBytes);
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &D_TEMP_BUFFER_GAS ),
                    TEMP_BUFFER_SIZE
                    ) );
    }

    size_t *d_compactedSize;
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_compactedSize ),
                sizeof(size_t)
                ) );
    OptixAccelEmitDesc property = {};
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = ( CUdeviceptr )d_compactedSize;
    
    OPTIX_CHECK( optixAccelBuild(
                context,
                0,                  // CUDA stream
                &accel_options,
                &aabb_input,
                1,                  // num build inputs
                D_TEMP_BUFFER_GAS,
                gas_buffer_sizes.tempSizeInBytes,
                D_GAS_OUTPUT_BUFFER,
                gas_buffer_sizes.outputSizeInBytes,
                &gas_handle,
                &property,            // emitted property list
                1                   // num emitted properties
                ) );

    size_t compactedSize;
    cudaMemcpy(&compactedSize, d_compactedSize,
        sizeof(size_t),
        cudaMemcpyDeviceToHost);

    if (compactedSize < gas_buffer_sizes.outputSizeInBytes) {

        if (COMPACT_GAS_BUFFER_SIZE <= compactedSize) {
            if (D_COMPACT_GAS_BUFFER != 0) {
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( D_COMPACT_GAS_BUFFER ) ) );
            }
            COMPACT_GAS_BUFFER_SIZE = compactedSize;
            CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &D_COMPACT_GAS_BUFFER ),
                        COMPACT_GAS_BUFFER_SIZE
                        ) );
            OPTIX_CHECK( optixAccelCompact( context, 0, gas_handle, D_COMPACT_GAS_BUFFER, COMPACT_GAS_BUFFER_SIZE, &gas_handle ) );
        }

    }
}
