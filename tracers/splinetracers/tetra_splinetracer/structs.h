#pragma once
#include "glm/glm.hpp"
#include <optix.h>

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

template <typename T> struct StructuredBuffer {
  T *data;
  size_t size;
};

struct SplineState//((packed))
{
  float2 distortion_parts;
  float2 cum_sum;
  float depth;
  float2 padding;
  // Spline state
  float t;
  float4 drgb;

  // Volume Rendering State
  float logT;
  float3 C;
};

// Always on GPU
struct Primitives {
  uint2 *side_index; 
  float *densities; 
  size_t num_prims;
  float *features; 
  size_t feature_size;

  glm::vec3 *vertices;
  glm::ivec3 *indices;
  size_t num_vertices;
  size_t num_indices;
  size_t num_faces;
};

