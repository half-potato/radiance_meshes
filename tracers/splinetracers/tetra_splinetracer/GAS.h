#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <optix.h>
#include <stdio.h>
#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "TriangleMesh.h"
#include "CUDABuffer.h"
#include "structs.h"

class GAS {
   public:
    OptixTraversableHandle gas_handle = 0;
    GAS() noexcept;
    GAS(const OptixDeviceContext &context, const uint8_t device, const bool enable_backwards, const bool fast_build) : device(device), context(context), enable_backwards(enable_backwards), fast_build(fast_build) {}
    GAS(
        const OptixDeviceContext &context,
        const uint8_t device,
        const Primitives &model, 
        const bool enable_backwards=false,
        const bool fast_build=false) : GAS(context, device, enable_backwards, fast_build) {
        build(model);
    }

    ~GAS() noexcept(false);
    GAS(const GAS &) = delete;
    GAS &operator=(const GAS &) = delete;
    GAS(GAS &&other) noexcept;
    GAS &operator=(GAS &&other) {
        using std::swap;
        if (this != &other) {
            GAS tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    friend void swap(GAS &first, GAS &second) {
        using std::swap;
        swap(first.context, second.context);
        swap(first.device, second.device);
        swap(first.gas_handle, second.gas_handle);
    }

    bool defined() const {
        return gas_handle != 0;
    }

   private:
    void build(const Primitives &model);
    bool enable_backwards, fast_build;

    void release();
    OptixDeviceContext context = nullptr;
    int8_t device = -1;
};

