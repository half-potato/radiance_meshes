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

#define TRI_PER_G 8
#define PT_PER_G 6
#include "create_aabbs.h"
#include "glm/glm.hpp"
#include "structs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::stringstream ss;                                                    \
      ss << "CUDA call (" << #call << " ) failed with error: '"                \
         << cudaGetErrorString(error) << "' (" __FILE__ << ":" << __LINE__     \
         << ")\n";                                                             \
      throw Exception(ss.str().c_str());                                       \
    }                                                                          \
  } while (0)

#define CUDA_SYNC_CHECK()                                                      \
  do {                                                                         \
    cudaDeviceSynchronize();                                                   \
    cudaError_t error = cudaGetLastError();                                    \
    if (error != cudaSuccess) {                                                \
      std::stringstream ss;                                                    \
      ss << "CUDA error on synchronize with error '"                           \
         << cudaGetErrorString(error) << "' (" __FILE__ << ":" << __LINE__     \
         << ")\n";                                                             \
      throw Exception(ss.str().c_str());                                       \
    }                                                                          \
  } while (0)

// A non-throwing variant for use in destructors.
// An iostream must be provided for output (e.g. std::cerr).
#define CUDA_CHECK_NOTHROW(call)                                               \
  do {                                                                         \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA call (" << #call << " ) failed with error: '"         \
                << cudaGetErrorString(error) << "' (" __FILE__ << ":"          \
                << __LINE__ << ")\n";                                          \
      std::terminate();                                                        \
    }                                                                          \
  } while (0)

class Exception : public std::runtime_error {
public:
  Exception(const char *msg) : std::runtime_error(msg) {}
};

__global__ void
kern_create_aabbs(const glm::vec3 *means, const glm::vec3 *scales,
                      const glm::vec4 *quats, const float *densities,
                      const size_t num_prims, OptixAabb *aabbs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 0 || i >= num_prims)
    return;

  const glm::vec4 quat = glm::normalize(quats[i]);
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
  float s = 1.0;
  glm::mat3 S = glm::mat3(1.0);
  S[0][0] = s*size.x;
  S[1][1] = s*size.y;
  S[2][2] = s*size.z;

  glm::mat4 M = glm::mat4(S * Rt);
  M[0][3] = center.x;
  M[1][3] = center.y;
  M[2][3] = center.z;

  aabbs[i] = {
    .minX = (float)(center.x - sqrt(M[0][0]*M[0][0] + M[0][1]*M[0][1] + M[0][2]*M[0][2])),
    .minY = (float)(center.y - sqrt(M[1][0]*M[1][0] + M[1][1]*M[1][1] + M[1][2]*M[1][2])),
    .minZ = (float)(center.z - sqrt(M[2][0]*M[2][0] + M[2][1]*M[2][1] + M[2][2]*M[2][2])),
    .maxX = (float)(center.x + sqrt(M[0][0]*M[0][0] + M[0][1]*M[0][1] + M[0][2]*M[0][2])),
    .maxY = (float)(center.y + sqrt(M[1][0]*M[1][0] + M[1][1]*M[1][1] + M[1][2]*M[1][2])),
    .maxZ = (float)(center.z + sqrt(M[2][0]*M[2][0] + M[2][1]*M[2][1] + M[2][2]*M[2][2])),
  };
}

void create_aabbs(Primitives &prims) {
  const size_t block_size = 1024;
  if (prims.prev_alloc_size < prims.num_prims) {
    if (prims.prev_alloc_size > 0) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(prims.aabbs)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&prims.aabbs),
                          prims.num_prims * sizeof(OptixAabb)));
  }
  kern_create_aabbs<<<(prims.num_prims + block_size - 1) / block_size,
                          block_size>>>(
      (glm::vec3 *)prims.means, (glm::vec3 *)prims.scales,
      (glm::vec4 *)prims.quats, prims.densities, prims.num_prims,
      prims.aabbs);
  CUDA_SYNC_CHECK();
}
