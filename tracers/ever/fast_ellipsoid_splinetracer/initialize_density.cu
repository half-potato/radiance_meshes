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

#define TRI_PER_G 4
#define PT_PER_G 4
#include "Forward.h"
#include "cuda_util.h"
#include "glm/glm.hpp"
#include "initialize_density.h"
#include "structs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define SQR(x) (x)*(x)

__device__ static const float SH_C0 = 0.28209479177387814f;

__device__ glm::vec3
get_Trayo(
    const glm::vec3 center,
    const glm::vec4 quat,
    const glm::vec3 size,
    const glm::vec3 rayo)
{
  const float r = quat.x;
  const float x = quat.y;
  const float y = quat.z;
  const float z = quat.w;

  const glm::mat3 Rt = {
      1.0 - 2.0 * (y * y + z * z),
      2.0 * (x * y - r * z),
      2.0 * (x * z + r * y),

      2.0 * (x * y + r * z),
      1.0 - 2.0 * (x * x + z * z),
      2.0 * (y * z - r * x),

      2.0 * (x * z - r * y),
      2.0 * (y * z + r * x),
      1.0 - 2.0 * (x * x + y * y)};
  const glm::mat3 R = glm::transpose(Rt);
  //
  const glm::mat3 invS = {
      1 / size.x, 0, 0, 0, 1 / size.y, 0, 0, 0, 1 / size.z,
  };
  const glm::mat3 S = {
      size.x, 0, 0, 0, size.y, 0, 0, 0, size.z,
  };

  const glm::vec3 Trayo = (Rt * (rayo - center)) / size;
  return Trayo;
}

__global__ void
kern_prefilter(
    const OptixAabb *aabbs,
    const size_t num_prims,
    const float tmin,
    const glm::vec3 *rayos,
    int *touch_indices,
    int *touch_count) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= num_prims)
    return;
  OptixAabb aabb = aabbs[i];
  const glm::vec3 rayo = rayos[0];
  // Jim Arvo, Graphics Gems.
  float dmin = 0;
  if (rayo.x < aabb.minX)
    dmin += SQR(rayo.x - aabb.minX);
  else if (rayo.x > aabb.maxX)
    dmin += SQR(rayo.x - aabb.maxX);

  if (rayo.y < aabb.minY)
    dmin += SQR(rayo.y - aabb.minY);
  else if (rayo.y > aabb.maxY)
    dmin += SQR(rayo.y - aabb.maxY);

  if (rayo.z < aabb.minZ)
    dmin += SQR(rayo.z - aabb.minZ);
  else if (rayo.z > aabb.maxZ)
    dmin += SQR(rayo.z - aabb.maxZ);
  
  if (dmin <= tmin*tmin) {
    int pos = atomicAdd(touch_count, 1);
    touch_indices[pos] = i;
  }

}

__global__ void
kern_initialize_density(
    const glm::vec3 *means, const glm::vec3 *scales,
    const glm::vec4 *quats, const float *densities,
    const float *features,
    const size_t num_prims,
    const size_t num_rays,
    const float tmin,
    const glm::vec3 *rayos,
    const glm::vec3 *rayds,
    float *initial_drgb,
    int *touch_indices,
    int *touch_count)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= *touch_count) return;
  if (j >= num_rays) return;
  glm::vec3 rayo = rayos[j] + tmin * glm::normalize(rayds[j]);
  {
    // In the future: Speed up using shared memeory
    const int prim_ind = touch_indices[i];
    const glm::vec4 quat = glm::normalize(quats[prim_ind]);
    const glm::vec3 center = means[prim_ind];
    const glm::vec3 size = scales[prim_ind];

    const glm::vec3 Trayo = get_Trayo(center, quat, size, rayo);
    const float dist = Trayo.x*Trayo.x + Trayo.y*Trayo.y + Trayo.z*Trayo.z;
    if (dist <= 1) {
      const float density = densities[prim_ind];
      // In the future: deal with spherical harmonics properly
      const glm::vec3 color = {
          features[prim_ind * 3 + 0] * SH_C0 + 0.5,
          features[prim_ind * 3 + 1] * SH_C0 + 0.5,
          features[prim_ind * 3 + 2] * SH_C0 + 0.5,
      };
      atomicAdd(initial_drgb + 4 * j + 0, density);
      atomicAdd(initial_drgb + 4 * j + 1, density * color.x);
      atomicAdd(initial_drgb + 4 * j + 2, density * color.y);
      atomicAdd(initial_drgb + 4 * j + 3, density * color.z);
    }
  }
}

void initialize_density(Params *params, OptixAabb *aabbs,
    int *d_touch_count, int *d_touch_inds) {
  const size_t block_size = 1024;
  const size_t ray_block_size = 64;
  const size_t second_block_size = 16;
  int num_prims = params->means.size;
  int num_rays = params->initial_drgb.size;

  size_t size = sizeof(float) * 4;
  dim3 grid_dim (
    (num_prims + block_size - 1) / block_size,
    (num_rays + ray_block_size - 1) / ray_block_size
  );
  dim3 block_dim (block_size, ray_block_size);

  bool initialize_tensors = d_touch_count == NULL;
  if (initialize_tensors) {
    cudaMalloc((void**)&d_touch_inds, num_prims * sizeof(int));
    cudaMalloc((void**)&d_touch_count, sizeof(int));
  }
  cudaMemset(d_touch_count, 0, sizeof(int));

  kern_prefilter<<<grid_dim.x, block_dim.x>>>(
      aabbs,
      num_prims, params->tmin,
      (glm::vec3 *)(params->ray_origins.data),
      d_touch_inds, d_touch_count);

  int touch_count;
  cudaMemcpy(&touch_count, d_touch_count, sizeof(int), cudaMemcpyDeviceToHost);


  if (touch_count > 0) {
    dim3 init_grid_dim (
      (num_rays + ray_block_size - 1) / ray_block_size,
      (touch_count + second_block_size - 1) / second_block_size,
      1
    );
    dim3 init_block_dim (ray_block_size, second_block_size, 1);
    kern_initialize_density<<<init_grid_dim, init_block_dim>>>(
        (glm::vec3 *)(params->means.data), (glm::vec3 *)(params->scales.data),
        (glm::vec4 *)(params->quats.data), (float *)(params->densities.data),
        (float *)(params->features.data),
        num_prims, num_rays, params->tmin,
        (glm::vec3 *)(params->ray_origins.data),
        (glm::vec3 *)(params->ray_directions.data),
        (float *)(params->initial_drgb.data),
        d_touch_inds, d_touch_count);

    CUDA_SYNC_CHECK();
  }

  if (initialize_tensors) {
    cudaFree(d_touch_inds);
    cudaFree(d_touch_count);
  }
}

__global__ void
kern_initialize_density_so(const glm::vec3 *means, const glm::vec3 *scales,
                           const glm::vec4 *quats, const float *densities,
                           const float *features, const size_t num_prims,
                           const glm::vec3 *rayo, float *initial_drgb) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= num_prims)
    return;

  const glm::vec4 quat = glm::normalize(quats[i]);
  const float density = densities[i];
  const glm::vec3 center = means[i];
  const glm::vec3 size = scales[i];

  const float r = quat.x;
  const float x = quat.y;
  const float y = quat.z;
  const float z = quat.w;

  const glm::mat3 Rt = {
      1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z),
      2.0 * (x * z + r * y),

      2.0 * (x * y + r * z),       1.0 - 2.0 * (x * x + z * z),
      2.0 * (y * z - r * x),

      2.0 * (x * z - r * y),       2.0 * (y * z + r * x),
      1.0 - 2.0 * (x * x + y * y)};
  const glm::mat3 R = glm::transpose(Rt);
  //
  const glm::mat3 invS = {
      1 / size.x, 0, 0, 0, 1 / size.y, 0, 0, 0, 1 / size.z,
  };
  const glm::mat3 S = {
      size.x, 0, 0, 0, size.y, 0, 0, 0, size.z,
  };

  const glm::vec3 Trayo = (Rt * (rayo[0] - center)) / size;
  float dist = Trayo.x*Trayo.x + Trayo.y*Trayo.y + Trayo.z*Trayo.z;
  if (dist <= 1) {
    glm::vec3 color = {
        features[i * 3 + 0] * SH_C0 + 0.5,
        features[i * 3 + 1] * SH_C0 + 0.5,
        features[i * 3 + 2] * SH_C0 + 0.5,
    };
    atomicAdd(initial_drgb + 0, density);
    atomicAdd(initial_drgb + 1, density * color.x);
    atomicAdd(initial_drgb + 2, density * color.y);
    atomicAdd(initial_drgb + 3, density * color.z);
  }
}

void initialize_density_so(Params *params) {
  const size_t block_size = 1024;
  int num_prims = params->means.size;

  kern_initialize_density_so<<<(num_prims + block_size - 1) / block_size,
                               block_size>>>(
      (glm::vec3 *)(params->means.data), (glm::vec3 *)(params->scales.data),
      (glm::vec4 *)(params->quats.data), (float *)(params->densities.data),
      (float *)(params->features.data), num_prims,
      (glm::vec3 *)(params->ray_origins.data),
      (float *)(params->initial_drgb.data));

  CUDA_SYNC_CHECK();
}

void initialize_density_zero(Params *params) {
}
