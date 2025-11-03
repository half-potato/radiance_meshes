
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
  TORCH_CHECK(x.dtype() == torch::kInt32, #x " must have float32 type")
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
    const int64_t numPrimitives = densities.size(0);
    const int64_t numFaces = side_index.size(0);
    CHECK_FLOAT_DIM3(vertices);
    CHECK_FLOAT(colors);
    TORCH_CHECK(colors.size(0) == numPrimitives,
                "All inputs (colors) must have the same 0 dimension")
    TORCH_CHECK(side_index.size(0) == face_indices.size(0),
                "All inputs (side_index, face_indices) must have the same 0 dimension")
    TORCH_CHECK(face_indices.size(1) == 3,
                "Face indices must have shape (N, 3)")
    TORCH_CHECK(side_index.size(1) == 2,
                "Side index must have shape (N, 2)")
    CHECK_INT(side_index);
    CHECK_INT(face_indices);
    CHECK_CONTIGUOUS(side_index);
    CHECK_CONTIGUOUS(face_indices);
    CHECK_CONTIGUOUS(densities);
    CHECK_CONTIGUOUS(colors);
    TORCH_CHECK(densities.size(0) == numPrimitives,
                "All inputs (densities) must have the same 0 dimension")
    model.feature_size = colors.size(1);
    model.side_index = reinterpret_cast<uint2 *>(side_index.data_ptr());
    model.densities = reinterpret_cast<float *>(densities.data_ptr());
    model.features = reinterpret_cast<float *>(colors.data_ptr());
    model.vertices = reinterpret_cast<glm::vec3 *>(vertices.data_ptr());
    model.indices = reinterpret_cast<glm::ivec3 *>(face_indices.data_ptr());
    model.num_vertices = vertices.size(0);
    model.num_indices = face_indices.size(0);
    model.num_faces = numFaces;
    model.num_prims = numPrimitives;
  }
};

struct tsPyGAS {
public:
  GAS gas;
  tsPyGAS(const tsOptixContext &context, const torch::Device &device,
        const tsPyPrimitives &model, const bool enable_anyhit,
        const bool fast_build, const bool enable_rebuild)
      : gas(context.context, device.index(), model.model, enable_anyhit,
            fast_build) {}
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
  tsSavedForBackward(size_t num_rays, size_t num_prims, torch::Device device)
      : num_prims(num_prims), num_float_per_state(sizeof(SplineState) / sizeof(float)),
        device(device) {
    allocate(num_rays);
  }
  uint *iters_data_ptr() { return reinterpret_cast<uint *>(iters.data_ptr()); }
  uint *touch_count_data_ptr() { return reinterpret_cast<uint *>(touch_count.data_ptr()); }
  uint *faces_data_ptr() { return reinterpret_cast<uint *>(faces.data_ptr()); }
  float4 *diracs_data_ptr() {
    return reinterpret_cast<float4 *>(diracs.data_ptr());
  }
  SplineState *states_data_ptr() {
    return reinterpret_cast<SplineState *>(states.data_ptr());
  }
  torch::Tensor get_states() { return states; }
  torch::Tensor get_diracs() { return diracs; }
  torch::Tensor get_faces() { return faces; }
  torch::Tensor get_iters() { return iters; }
  torch::Tensor get_touch_count() { return touch_count; }
  void allocate(size_t num_rays) {
    states = torch::zeros({(long)num_rays, num_float_per_state},
                          torch::device(device).dtype(torch::kFloat32));
    diracs = torch::zeros({(long)num_rays, 4},
                          torch::device(device).dtype(torch::kFloat32));
    faces = torch::zeros({(long)num_rays},
                         torch::device(device).dtype(torch::kInt32));
    touch_count = torch::zeros({(long)num_prims},
                         torch::device(device).dtype(torch::kInt32));
    iters = torch::zeros({(long)num_rays},
                         torch::device(device).dtype(torch::kInt32));
    this->num_rays = num_rays;
  }
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
  py::dict trace_rays(const tsPyGAS &gas, const torch::Tensor &ray_origins,
                      const torch::Tensor &ray_directions, float tmin,
                      float tmax, const size_t max_iters,
                      const torch::Tensor start_tet_ids) {
    torch::AutoGradMode enable_grad(false);
    CHECK_FLOAT_DIM3(ray_origins);
    CHECK_FLOAT_DIM3(ray_directions);
    const size_t num_rays = ray_origins.numel() / 3;
    torch::Tensor color;
    color = torch::zeros({(long)num_rays, 4},
                         torch::device(device).dtype(torch::kFloat32));
    torch::Tensor tri_collection =
        torch::zeros({(long)num_rays * max_iters},
                     torch::device(device).dtype(torch::kInt32));
    tsSavedForBackward saved_for_backward(num_rays, num_prims, device);
    forward.trace_rays(gas.gas.gas_handle, num_rays,
                       reinterpret_cast<float3 *>(ray_origins.data_ptr()),
                       reinterpret_cast<float3 *>(ray_directions.data_ptr()),
                       reinterpret_cast<void *>(color.data_ptr()), 
                       tmin, tmax, max_iters, 
                       saved_for_backward.iters_data_ptr(),
                       saved_for_backward.faces_data_ptr(),
                       saved_for_backward.touch_count_data_ptr(),
                       saved_for_backward.diracs_data_ptr(),
                       saved_for_backward.states_data_ptr(),
                       reinterpret_cast<int *>(tri_collection.data_ptr()),
                       reinterpret_cast<int *>(start_tet_ids.data_ptr()));
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
      .def("add_primitives", &tsPyPrimitives::add_primitives);
  py::class_<tsPyGAS>(m, "GAS").def(
      py::init<const tsOptixContext &, const torch::Device &,
               const tsPyPrimitives &, const bool, const bool, const bool>());
  // .def("update", &tsPyGAS::update);
  py::class_<tsPyForward>(m, "Forward")
      .def(py::init<const tsOptixContext &, const torch::Device &,
                    const tsPyPrimitives &, const bool>())
      .def("trace_rays", &tsPyForward::trace_rays);
}
