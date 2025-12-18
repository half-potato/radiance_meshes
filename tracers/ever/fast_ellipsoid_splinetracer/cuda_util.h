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

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::stringstream ss;                                                    \
      ss << "CUDA call (" << #call << " ) failed with error: '"                \
         << cudaGetErrorString(error) << "' (" __FILE__ << ":" << __LINE__     \
         << ")\n";                                                             \
      throw CudaException(ss.str().c_str());                                   \
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
      throw CudaException(ss.str().c_str());                                   \
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

class CudaException : public std::runtime_error {
public:
  CudaException(const char *msg) : std::runtime_error(msg) {}
};
