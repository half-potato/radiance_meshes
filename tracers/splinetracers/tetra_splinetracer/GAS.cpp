#include <optix_stubs.h>
#include "glm/glm.hpp"
#include "GAS.h"
#include <chrono>

#ifndef __DEFINED_OUTPUT_BUFFERS__
CUdeviceptr D_GAS_OUTPUT_BUFFER = 0;
size_t OUTPUT_BUFFER_SIZE = 0;
CUdeviceptr D_TEMP_BUFFER_GAS = 0;
size_t TEMP_BUFFER_SIZE = 0;
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
      // d_gas_output_buffer(std::exchange(other.d_gas_output_buffer, 0))
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
    //
    // Use default options for simplicity.  In a real use case we would want to
    // enable compaction, etc
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    // auto vertices = model.vertex;
    //
    CUdeviceptr d_vertices = (CUdeviceptr)model.vertices;
    CUdeviceptr d_indices  = (CUdeviceptr)model.indices;

    // Our build input is a simple list of non-indexed triangle vertices
    // if (!enable_backwards) {
    //     triangle_input_flags[0] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    // } else {
    //     triangle_input_flags[0] = OPTIX_GEOMETRY_FLAG_NONE;
    // }
    // uint32_t triangle_input_flags[1];
    uint32_t triangle_input_flags[1];
    // triangle_input_flags[0] = OPTIX_GEOMETRY_FLAG_NONE;
    triangle_input_flags[0] = OPTIX_GEOMETRY_FLAG_NONE;// OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
    // triangle_input_flags[1] = OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT;
    // triangle_input_flags[0] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    OptixBuildInput triangle_input = {};
    triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices   = static_cast<uint32_t>( model.num_vertices );
    triangle_input.triangleArray.vertexBuffers = &d_vertices;

    triangle_input.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes  = sizeof(glm::ivec3);
    triangle_input.triangleArray.numIndexTriplets    = (int)model.num_indices;
    triangle_input.triangleArray.indexBuffer         = d_indices;
    // printf("Nums: %i, %i\n", static_cast<uint32_t>( model.num_vertices ), (int)model.num_indices);

    triangle_input.triangleArray.flags         = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                context,
                &accel_options,
                &triangle_input,
                1, // Number of build inputs
                &gas_buffer_sizes
                ) );

    // Handle allocation of the GAS
    if (OUTPUT_BUFFER_SIZE <= gas_buffer_sizes.outputSizeInBytes) {
        if (D_GAS_OUTPUT_BUFFER != 0) {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( D_GAS_OUTPUT_BUFFER ) ) );
        }
        OUTPUT_BUFFER_SIZE = size_t(1.1*gas_buffer_sizes.outputSizeInBytes);
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &D_GAS_OUTPUT_BUFFER ),
                    OUTPUT_BUFFER_SIZE
                    ) );
    }

    if (TEMP_BUFFER_SIZE <= gas_buffer_sizes.tempSizeInBytes) {
        if (D_TEMP_BUFFER_GAS != 0) {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( D_TEMP_BUFFER_GAS ) ) );
        }
        TEMP_BUFFER_SIZE = size_t(1.1*gas_buffer_sizes.tempSizeInBytes);
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &D_TEMP_BUFFER_GAS ),
                    TEMP_BUFFER_SIZE
                    ) );
    }

    // start = high_resolution_clock::now();
    
    OPTIX_CHECK( optixAccelBuild(
                context,
                0,                  // CUDA stream
                &accel_options,
                &triangle_input,
                1,                  // num build inputs
                D_TEMP_BUFFER_GAS,
                gas_buffer_sizes.tempSizeInBytes,
                D_GAS_OUTPUT_BUFFER,
                gas_buffer_sizes.outputSizeInBytes,
                &gas_handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ) );
    // CUDA_SYNC_CHECK();
    // CUDA_CHECK(cudaStreamSynchronize(0));
    // auto ms3 = float(duration_cast<microseconds>(high_resolution_clock::now() - start).count()) / 1000.f;
    // printf("GASed in %f ms\n", ms3);
    // start = high_resolution_clock::now();

    // We can now free the scratch space buffer used during build and the vertex
    // inputs, since they are not needed by our trivial shading method
    // CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
    // CUDA_SYNC_CHECK();
    // CUDA_CHECK(cudaStreamSynchronize(0));
    // CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_vertices        ) ) );

    // auto ms5 = float(duration_cast<microseconds>(high_resolution_clock::now() - full_start).count()) / 1000.f;
    // printf("Done in %f ms\n", ms5);
}
