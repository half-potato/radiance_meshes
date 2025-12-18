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

#define GLM_FORCE_CUDA

#include "cuda_util.h"
#include "glm/glm.hpp"
#include "structs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "sh.h"

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {1.0925484305920792f, -1.0925484305920792f,
                                  0.31539156525252005f, -1.0925484305920792f,
                                  0.5462742152960396f};
__device__ const float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f,
                                  -0.4570457994644658f, 0.3731763325901154f,
                                  -0.4570457994644658f, 1.445305721320277f,
                                  -0.5900435899266435f};

__device__ float softplus(float x, float beta) {
  return 1/beta * log(1+ exp(beta * x));
}

__device__ glm::vec3 d_eval_sh(int deg, glm::vec3 dir, const glm::vec3 *sh) {
  glm::vec3 result = SH_C0 * sh[0] + 0.5f;

  if (deg > 0) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result + SH_C2[0] * xy * sh[4] + SH_C2[1] * yz * sh[5] +
               SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
               SH_C2[3] * xz * sh[7] + SH_C2[4] * (xx - yy) * sh[8];

      if (deg > 2) {
        result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                 SH_C3[1] * xy * z * sh[10] +
                 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                 SH_C3[5] * z * (xx - yy) * sh[14] +
                 SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
      }
    }
  }
  return glm::max(result, 0.f);
}

__global__ void kern_eval_sh(const glm::vec3 *means, const float *shs,
                             const glm::vec3 origin, float *colors, int deg,
                             int max_coeffs, int num_prims) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 0 || idx >= num_prims)
    return;
  glm::vec3 *sh = ((glm::vec3 *)(shs + idx * max_coeffs));
  glm::vec3 mean = means[idx];

  glm::vec3 dir = mean - origin;
  dir = dir / glm::length(dir);

  glm::vec3 color = d_eval_sh(deg, dir, sh);
  // convert to SH deg 0
  colors[idx * 3 + 0] = (color.x - 0.5) / SH_C0;
  colors[idx * 3 + 1] = (color.y - 0.5) / SH_C0;
  colors[idx * 3 + 2] = (color.z - 0.5) / SH_C0;
}

Primitives eval_sh(Primitives prims, int deg, int max_coeffs, glm::vec3 origin,
                   float *color_buffer) {
  const size_t block_size = 1024;
  Primitives return_prims = prims;
  kern_eval_sh<<<(prims.num_prims + block_size - 1) / block_size, block_size>>>(
      (glm::vec3 *)prims.means, prims.features, origin, color_buffer, deg,
      max_coeffs, prims.num_prims);
  CUDA_SYNC_CHECK();
  return_prims.features = color_buffer;
  return_prims.feature_size = 3;
  return return_prims;
}

