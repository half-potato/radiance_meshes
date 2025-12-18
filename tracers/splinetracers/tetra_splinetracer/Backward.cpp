#include "Backward.h"
#include "CUDABuffer.h"
// #include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

void Backward::trace_rays(const OptixTraversableHandle &handle,
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
                         float3 *dL_drayd)
{
    CUDA_CHECK(cudaSetDevice(device));
    {

        params.iters.data = iters;
        params.iters.size = num_rays;
        params.last_face.data = last_face;
        params.last_face.size = num_rays;
        params.last_dirac.data = last_dirac;
        params.last_dirac.size = num_rays;
        params.last_state.data = last_state;
        params.last_state.size = num_rays;

        params.feature_size = feature_size;

        params.ray_origins.data = ray_origins;
        params.ray_origins.size = num_rays;
        params.ray_directions.data = ray_directions;
        params.ray_directions.size = num_rays;

        params.dL_dC.data = dL_dC;
        params.dL_dC.size = num_rays;
        params.dL_dmean.data = dL_dmean;
        params.dL_dmean.size = num_rays;
        params.dL_dscale.data = dL_dscale;
        params.dL_dscale.size = num_rays;
        params.dL_dquat.data = dL_dquat;
        params.dL_dquat.size = num_rays;
        params.dL_dheight.data = dL_dheight;
        params.dL_dheight.size = num_rays;
        params.dL_dfeature.data = dL_dfeature;
        params.dL_dfeature.size = num_rays;
        params.dL_drayo.data = dL_drayo;
        params.dL_drayo.size = num_rays;
        params.dL_drayd.data = dL_drayd;
        params.dL_drayd.size = num_rays;

        params.handle = handle;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( BackwardParams ) ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( d_param ),
                    &params, sizeof( params ),
                    cudaMemcpyHostToDevice
                    ) );

        OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( BackwardParams ), &sbt, num_rays, 1, 1 ) );

        CUDA_SYNC_CHECK();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

Backward::Backward(const OptixDeviceContext &context, int8_t device, const Primitives &model) : device(device), context(context) {
    // Initialize fields
    OptixPipelineCompileOptions pipeline_compile_options = {};

    // Switch to active device
    CUDA_CHECK(cudaSetDevice(device));

    char log[2048];  // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);

    //
    // Create module
    //
    {

        pipeline_compile_options.usesMotionBlur        = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues      = 4;
        pipeline_compile_options.numAttributeValues    = 3;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
        pipeline_compile_options.pipelineLaunchParamsVariableName = "SLANG_globalParams";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    }

    OptixModule module = nullptr;
    {
        OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
        module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

        size_t      inputSize  = 0;

        const std::string input = Backward::load_ptx_data();
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                    context,
                    &module_compile_options,
                    &pipeline_compile_options,
                    input.c_str(),
                    input.size(),
                    log, &sizeof_log,
                    &module
                    ) );
    }

    //
    // Create program groups
    //
    // OptixProgramGroup raygen_prog_group   = nullptr;
    // OptixProgramGroup miss_prog_group     = nullptr;
    // OptixProgramGroup hitgroup_prog_group = nullptr;
    {
        OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg_backward";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context,
                    &raygen_prog_group_desc,
                    1,   // num program groups
                    &program_group_options,
                    log, &sizeof_log,
                    &raygen_prog_group
                    ) );

        OptixProgramGroupDesc miss_prog_group_desc  = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context,
                    &miss_prog_group_desc,
                    1,   // num program groups
                    &program_group_options,
                    log, &sizeof_log,
                    &miss_prog_group
                    ) );

        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleAH            = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
        hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    context,
                    &hitgroup_prog_group_desc,
                    1,   // num program groups
                    &program_group_options,
                    log, &sizeof_log,
                    &hitgroup_prog_group
                    ) );
    }

    //
    // Link pipeline
    //
    // OptixPipeline pipeline = nullptr;
    {
        const uint32_t    max_trace_depth  = 1;
        OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth          = max_trace_depth;
        pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        OPTIX_CHECK_LOG( optixPipelineCreate(
                    context,
                    &pipeline_compile_options,
                    &pipeline_link_options,
                    program_groups,
                    sizeof( program_groups ) / sizeof( program_groups[0] ),
                    log, &sizeof_log,
                    &pipeline
                    ) );

        OptixStackSizes stack_sizes = {};
        for( auto& prog_group : program_groups )
        {
            OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                 0,  // maxCCDepth
                                                 0,  // maxDCDEpth
                                                 &direct_callable_stack_size_from_traversal,
                                                 &direct_callable_stack_size_from_state, &continuation_stack_size ) );
        OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                                direct_callable_stack_size_from_state, continuation_stack_size,
                                                1  // maxTraversableDepth
                                                ) );
    }

    //
    // Set up shader binding table
    //
    // OptixShaderBindingTable sbt = {};
    {
        CUdeviceptr  raygen_record;
        const size_t raygen_record_size = sizeof( BWRayGenSbtRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
        BWRayGenSbtRecord rg_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( raygen_record ),
                    &rg_sbt,
                    raygen_record_size,
                    cudaMemcpyHostToDevice
                    ) );

        CUdeviceptr miss_record;
        size_t      miss_record_size = sizeof( BWMissSbtRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
        BWMissSbtRecord ms_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( miss_record ),
                    &ms_sbt,
                    miss_record_size,
                    cudaMemcpyHostToDevice
                    ) );

        CUdeviceptr hitgroup_record;
        size_t      hitgroup_record_size = sizeof( BWHitGroupSbtRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
        BWHitGroupSbtRecord hg_sbt;

        OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( hitgroup_record ),
                    &hg_sbt,
                    hitgroup_record_size,
                    cudaMemcpyHostToDevice
                    ) );

        sbt.raygenRecord                = raygen_record;
        sbt.missRecordBase              = miss_record;
        sbt.missRecordStrideInBytes     = sizeof( BWMissSbtRecord );
        sbt.missRecordCount             = 1;
        sbt.hitgroupRecordBase          = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof( BWHitGroupSbtRecord );
        sbt.hitgroupRecordCount         = 1;
    }
    {
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
    }
}


Backward::Backward(Backward &&other) noexcept
    : device(std::exchange(other.device, -1)),
      context(std::exchange(other.context, nullptr)),
      pipeline(std::exchange(other.pipeline, nullptr)),
      raygen_prog_group(std::exchange(other.raygen_prog_group, nullptr)),
      miss_prog_group(std::exchange(other.miss_prog_group, nullptr)),
      hitgroup_prog_group(std::exchange(other.hitgroup_prog_group, nullptr)),
      module(std::exchange(other.module, nullptr)),
      sbt(std::exchange(other.sbt, {})),
      stream(std::exchange(other.stream, nullptr)),
      d_param(std::exchange(other.d_param, 0)) {}
      


Backward::~Backward() noexcept(false) {
    const auto device = std::exchange(this->device, -1);
    if (device == -1) {
        return;
    }
    CUDA_CHECK(cudaSetDevice(device));
    if (d_param != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(d_param, 0))));
    if (sbt.raygenRecord != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.raygenRecord, 0))));
    if (sbt.missRecordBase != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.missRecordBase, 0))));
    if (sbt.hitgroupRecordBase != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.hitgroupRecordBase, 0))));
    if (sbt.callablesRecordBase)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.callablesRecordBase, 0))));
    if (sbt.exceptionRecord)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.exceptionRecord, 0))));
    sbt = {};
    if (stream != nullptr)
        CUDA_CHECK(cudaStreamDestroy(std::exchange(stream, nullptr)));
    if (pipeline != nullptr)
        OPTIX_CHECK(optixPipelineDestroy(std::exchange(pipeline, nullptr)));
    if (raygen_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(raygen_prog_group, nullptr)));
    if (miss_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(miss_prog_group, nullptr)));
    if (hitgroup_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(hitgroup_prog_group, nullptr)));
    if (module != nullptr)
        OPTIX_CHECK(optixModuleDestroy(std::exchange(module, nullptr)));
}


