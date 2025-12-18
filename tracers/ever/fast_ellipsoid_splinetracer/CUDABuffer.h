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

#pragma once

#include <assert.h>
#include <vector>
#include <exception.h>

struct CUDABuffer {
  inline CUdeviceptr d_pointer() const { return (CUdeviceptr)d_ptr; }

  void resize(size_t size) {
    if (d_ptr)
      free();
    alloc(size);
  }

  void alloc(size_t size) {
    assert(d_ptr == nullptr);
    this->sizeInBytes = size;
    CUDA_CHECK(cudaMalloc((void **)&d_ptr, sizeInBytes));
  }

  void free() {
    CUDA_CHECK(cudaFree(d_ptr));
    d_ptr = nullptr;
    sizeInBytes = 0;
  }

  template <typename T> void alloc_and_upload(const std::vector<T> &vt) {
    alloc(vt.size() * sizeof(T));
    upload((const T *)vt.data(), vt.size());
  }

  template <typename T> void upload(const T *t, size_t count) {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(
        cudaMemcpy(d_ptr, (void *)t, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  template <typename T> void download(T *t, size_t count) {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(
        cudaMemcpy((void *)t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
  }

  size_t sizeInBytes{0};
  void *d_ptr{nullptr};
};
