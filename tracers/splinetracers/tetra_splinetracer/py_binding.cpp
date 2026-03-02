
#include <iostream>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <torch/extension.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

// #include "Backward.h"
// #include "CollectIds.h"
#include "Forward.h"
#include "GAS.h"
#include "TriangleMesh.h"
#include "exception.h"

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x)                                                        \
  TORCH_CHECK(x.device() == this->device, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x)                                                         \
  TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INT(x)                                                         \
  TORCH_CHECK(x.dtype() == torch::kInt32, #x " must have int32 type")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT_DIM3(x)                                                    \
  CHECK_INPUT(x);                                                              \
  CHECK_DEVICE(x);                                                             \
  CHECK_FLOAT(x);                                                              \
  TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")
#define CHECK_FLOAT_DIM4(x)                                                    \
  CHECK_INPUT(x);                                                              \
  CHECK_DEVICE(x);                                                             \
  CHECK_FLOAT(x);                                                              \
  TORCH_CHECK(x.size(-1) == 4, #x " must have last dimension with size 4")
#define CHECK_FLOAT_DIM4_CPU(x)                                                \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_FLOAT(x);                                                              \
  TORCH_CHECK(x.size(-1) == 4, #x " must have last dimension with size 4")
#define CHECK_FLOAT_DIM3_CPU(x)                                                \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_FLOAT(x);                                                              \
  TORCH_CHECK(x.size(-1) == 3, #x " must have last dimension with size 3")

// Get raw data pointer bypassing has_storage() check, which gives
// false negatives with PYTORCH_ALLOC_CONF=expandable_segments:True.
inline void* unsafe_data_ptr(const torch::Tensor& t) {
  auto* impl = t.unsafeGetTensorImpl();
  const auto& s = impl->unsafe_storage();
  if (!s) return nullptr;
  return static_cast<char*>(const_cast<void*>(s.data()))
       + impl->storage_offset() * impl->dtype().itemsize();
}

#define CHECK_TENSOR_VALID(x)                                                  \
  TORCH_CHECK(x.defined(), #x " is undefined (null tensor)");                 \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x);                                                         \
  TORCH_CHECK(unsafe_data_ptr(x) != nullptr, #x " has null data pointer")

// Check for pending CUDA errors — a previous async op may have failed
#define CHECK_CUDA_STATE(label)                                                \
  {                                                                            \
    cudaError_t _err = cudaGetLastError();                                     \
    if (_err != cudaSuccess) {                                                 \
      TORCH_CHECK(false, label ": pending CUDA error BEFORE this point: ",    \
                  cudaGetErrorString(_err));                                    \
    }                                                                          \
  }

static void context_log_cb(unsigned int level, const char *tag,
                           const char *message, void * /*cbdata */) {
  // std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) <<
  // tag << "]: "
  //     << message << "\n";
}

struct tsOptixContext {
public:
  OptixDeviceContext context = nullptr;
  uint device;
  tsOptixContext(const torch::Device &device) : device(device.index()) {
    CUDA_CHECK(cudaSetDevice(device.index()));
    {
      // Initialize CUDA
      CUDA_CHECK(cudaFree(0));
      // Initialize the OptiX API, loading all API entry points
      OPTIX_CHECK(optixInit());
      // Specify context options
      OptixDeviceContextOptions options = {};
      options.logCallbackFunction = &context_log_cb;
      options.logCallbackLevel = 4;
      // Associate a CUDA context (and therefore a specific GPU) with this
      // device context
      CUcontext cuCtx = 0; // zero means take the current context
      OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    }
  }
  ~tsOptixContext() { OPTIX_CHECK(optixDeviceContextDestroy(context)); }
};

struct tsPyPrimitives {
public:
  Primitives model;
  torch::Device device;
  tsPyPrimitives(const torch::Device &device) : device(device) {}
  void add_primitives(
                      const torch::Tensor &vertices,
                      const torch::Tensor &face_indices,
                      const torch::Tensor &side_index,
                      const torch::Tensor &densities,
                      const torch::Tensor &colors) {
    CHECK_TENSOR_VALID(vertices);
    CHECK_TENSOR_VALID(face_indices);
    CHECK_TENSOR_VALID(side_index);
    CHECK_TENSOR_VALID(densities);
    CHECK_TENSOR_VALID(colors);

    const int64_t numPrimitives = densities.size(0);
    const int64_t numFaces = side_index.size(0);

    CHECK_FLOAT_DIM3(vertices);
    CHECK_FLOAT(colors);
    CHECK_FLOAT(densities);
    CHECK_INT(side_index);
    TORCH_CHECK(colors.size(0) == numPrimitives,
                "colors.size(0)=", colors.size(0), " != densities.size(0)=", numPrimitives)
    TORCH_CHECK(side_index.size(0) == face_indices.size(0),
                "side_index.size(0)=", side_index.size(0), " != face_indices.size(0)=", face_indices.size(0))
    TORCH_CHECK(face_indices.size(1) == 3,
                "face_indices.size(1)=", face_indices.size(1), " expected 3")
    TORCH_CHECK(side_index.size(1) == 2,
                "side_index.size(1)=", side_index.size(1), " expected 2")

    model.feature_size = colors.size(1);
    model.side_index = reinterpret_cast<uint2 *>(unsafe_data_ptr(side_index));
    model.densities = reinterpret_cast<float *>(unsafe_data_ptr(densities));
    model.features = reinterpret_cast<float *>(unsafe_data_ptr(colors));
    model.vertices = reinterpret_cast<glm::vec3 *>(unsafe_data_ptr(vertices));
    model.indices = reinterpret_cast<glm::ivec3 *>(unsafe_data_ptr(face_indices));
    model.num_vertices = vertices.size(0);
    model.num_indices = face_indices.size(0);
    model.num_faces = numFaces;
    model.num_prims = numPrimitives;
  }
  void set_features(const torch::Tensor &colors) {
    CHECK_TENSOR_VALID(colors);
    CHECK_FLOAT(colors);
    TORCH_CHECK(colors.size(0) == model.num_prims,
                "set_features: colors.size(0)=", colors.size(0),
                " != num_prims=", model.num_prims);
    model.features = reinterpret_cast<float *>(unsafe_data_ptr(colors));
  }
};

struct tsPyGAS {
public:
  GAS gas;
  tsPyGAS(const tsOptixContext &context, const torch::Device &device,
        const tsPyPrimitives &model, const bool enable_anyhit,
        const bool fast_build, const bool enable_rebuild)
      : gas(context.context, device.index(), model.model, enable_anyhit,
            fast_build) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "[GAS] WARNING: pending CUDA error after GAS build: "
                << cudaGetErrorString(err) << std::endl;
    }
  }
  // void update(const tsPyPrimitives &model) {
  //     gas.build(model.model, true);
  // }
};

struct tsSavedForBackward {
public:
  torch::Tensor states, diracs, faces, touch_count, iters;
  size_t num_prims;
  size_t num_rays;
  size_t num_float_per_state;
  torch::Device device;
  tsSavedForBackward(torch::Device device)
      : num_prims(0), num_rays(0), num_float_per_state(sizeof(SplineState) / sizeof(float)),
        device(device) {}
  // Accept pre-allocated tensors from Python (avoids torch::zeros inside C++ after OptiX)
  tsSavedForBackward(size_t num_prims, torch::Device device,
                     const torch::Tensor &states_,
                     const torch::Tensor &diracs_,
                     const torch::Tensor &faces_,
                     const torch::Tensor &touch_count_,
                     const torch::Tensor &iters_)
      : num_prims(num_prims), num_float_per_state(sizeof(SplineState) / sizeof(float)),
        device(device), states(states_), diracs(diracs_), faces(faces_),
        touch_count(touch_count_), iters(iters_), num_rays(states_.size(0)) {}
  uint *iters_data_ptr() { return reinterpret_cast<uint *>(unsafe_data_ptr(iters)); }
  uint *touch_count_data_ptr() { return reinterpret_cast<uint *>(unsafe_data_ptr(touch_count)); }
  uint *faces_data_ptr() { return reinterpret_cast<uint *>(unsafe_data_ptr(faces)); }
  float4 *diracs_data_ptr() {
    return reinterpret_cast<float4 *>(unsafe_data_ptr(diracs));
  }
  SplineState *states_data_ptr() {
    return reinterpret_cast<SplineState *>(unsafe_data_ptr(states));
  }
  torch::Tensor get_states() { return states; }
  torch::Tensor get_diracs() { return diracs; }
  torch::Tensor get_faces() { return faces; }
  torch::Tensor get_iters() { return iters; }
  torch::Tensor get_touch_count() { return touch_count; }
};

struct tsPyForward {
public:
  Forward forward;
  torch::Device device;
  size_t num_prims;
  uint sh_degree;
  tsPyForward(const tsOptixContext &context, const torch::Device &device,
            const tsPyPrimitives &model, const bool enable_backward)
      : device(device),
        forward(context.context, device.index(), model.model, enable_backward),
        num_prims(model.model.num_prims),
        sh_degree(sqrt(model.model.feature_size) - 1) {}
  void update_model(const tsPyPrimitives &model) {
    forward.reset_features(model.model);
  }
  // All output tensors are pre-allocated from Python to avoid torch::zeros
  // inside C++ after OptiX operations (which can corrupt PyTorch's allocator).
  py::dict trace_rays(const tsPyGAS &gas, const torch::Tensor &ray_origins,
                      const torch::Tensor &ray_directions, float tmin,
                      float tmax, const size_t max_iters,
                      const torch::Tensor start_tet_ids,
                      const torch::Tensor &color,
                      const torch::Tensor &tri_collection,
                      const torch::Tensor &states,
                      const torch::Tensor &diracs,
                      const torch::Tensor &faces,
                      const torch::Tensor &touch_count,
                      const torch::Tensor &iters_buf) {
    torch::AutoGradMode enable_grad(false);

    // --- Validate inputs ---
    CHECK_TENSOR_VALID(ray_origins);
    CHECK_TENSOR_VALID(ray_directions);
    CHECK_TENSOR_VALID(start_tet_ids);
    CHECK_FLOAT_DIM3(ray_origins);
    CHECK_FLOAT_DIM3(ray_directions);
    CHECK_INT(start_tet_ids);
    TORCH_CHECK(ray_origins.size(0) == ray_directions.size(0),
                "trace_rays: ray count mismatch: origins=", ray_origins.size(0),
                " directions=", ray_directions.size(0));
    TORCH_CHECK(start_tet_ids.size(0) == ray_origins.size(0),
                "trace_rays: start_tet_ids.size(0)=", start_tet_ids.size(0),
                " != num_rays=", ray_origins.size(0));

    // --- Validate pre-allocated output buffers ---
    CHECK_TENSOR_VALID(color);
    CHECK_TENSOR_VALID(tri_collection);
    CHECK_TENSOR_VALID(states);
    CHECK_TENSOR_VALID(diracs);
    CHECK_TENSOR_VALID(faces);
    CHECK_TENSOR_VALID(touch_count);
    CHECK_TENSOR_VALID(iters_buf);

    const size_t num_rays = ray_origins.numel() / 3;

    tsSavedForBackward saved_for_backward(num_prims, device,
                                          states, diracs, faces,
                                          touch_count, iters_buf);

    forward.trace_rays(gas.gas.gas_handle, num_rays,
                       reinterpret_cast<float3 *>(unsafe_data_ptr(ray_origins)),
                       reinterpret_cast<float3 *>(unsafe_data_ptr(ray_directions)),
                       unsafe_data_ptr(color),
                       tmin, tmax, max_iters,
                       saved_for_backward.iters_data_ptr(),
                       saved_for_backward.faces_data_ptr(),
                       saved_for_backward.touch_count_data_ptr(),
                       saved_for_backward.diracs_data_ptr(),
                       saved_for_backward.states_data_ptr(),
                       reinterpret_cast<int *>(unsafe_data_ptr(tri_collection)),
                       reinterpret_cast<int *>(unsafe_data_ptr(start_tet_ids)));

    return py::dict("color"_a = color, "saved"_a = saved_for_backward,
                    "tri_collection"_a = tri_collection);
  }
};

PYBIND11_MODULE(tetra_splinetracer_cpp_extension, m) {
  py::class_<tsOptixContext>(m, "OptixContext")
      .def(py::init<const torch::Device &>());
  py::class_<tsSavedForBackward>(m, "SavedForBackward")
      .def_property_readonly("states", &tsSavedForBackward::get_states)
      .def_property_readonly("diracs", &tsSavedForBackward::get_diracs)
      .def_property_readonly("touch_count", &tsSavedForBackward::get_touch_count)
      .def_property_readonly("iters", &tsSavedForBackward::get_iters)
      .def_property_readonly("faces", &tsSavedForBackward::get_faces);
  py::class_<tsPyPrimitives>(m, "Primitives")
      .def(py::init<const torch::Device &>())
      .def("add_primitives", &tsPyPrimitives::add_primitives)
      .def("set_features", &tsPyPrimitives::set_features);
  py::class_<tsPyGAS>(m, "GAS").def(
      py::init<const tsOptixContext &, const torch::Device &,
               const tsPyPrimitives &, const bool, const bool, const bool>());
  // .def("update", &tsPyGAS::update);
  py::class_<tsPyForward>(m, "Forward")
      .def(py::init<const tsOptixContext &, const torch::Device &,
                    const tsPyPrimitives &, const bool>())
      .def("trace_rays", &tsPyForward::trace_rays)
      .def("update_model", &tsPyForward::update_model);
}
