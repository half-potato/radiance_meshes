# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

enable_language(CUDA)

set(slang_dir ${CMAKE_CURRENT_SOURCE_DIR}/slang)
set(main_slang_dir ${CMAKE_CURRENT_SOURCE_DIR}/../slang)

slang_compile_and_embed(fast_ptx_code_file
  ${slang_dir}/fast_shaders.slang
  ${main_slang_dir}
  ${slang_dir}
)

slang_compile_and_embed(ptx_code_file
  ${slang_dir}/shaders.slang
  ${main_slang_dir}
  ${slang_dir}
)


cuda_compile_and_embed(ptx_code_file2
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cu
)

pybind11_add_module(fast_ellipsoid_splinetracer_cpp_extension 
  ${ptx_code_file}
  ${fast_ptx_code_file}
  ${ptx_code_file2}
  # ${backward_ptx_code_file}
  # ${collect_ids_ptx_code_file}
  ${generated_files}
  Forward.h
  Forward.cpp
  # Backward.h
  # Backward.cpp
  GAS.cpp
  GAS.h
  structs.h
  CUDABuffer.h
  create_aabbs.h
  create_aabbs.cu
  initialize_density.h
  initialize_density.cu
  # CollectIds.h
  # CollectIds.cpp
  #ply_file_loader.cc
  #ply_file_loader.h
  # exception.h
  py_binding.cpp
  # ${cmake_options}
)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-lto")

set_source_files_properties(create_aabbs.cu PROPERTIES COMPILE_FLAGS "-Xcompiler -fno-lto"
    # CUDA_COMPILE_PTX ON
    # LANGUAGE CUDA
)
set_source_files_properties(initialize_density.cu PROPERTIES COMPILE_FLAGS "-Xcompiler -fno-lto")

target_link_libraries(fast_ellipsoid_splinetracer_cpp_extension PRIVATE
  ${CUDA_LIBRARIES} ${TORCH_PYTHON_LIBRARIES} ${TORCH_LIBRARIES} torch torch_cpu c10 torch_cuda m
  )
set_target_properties(fast_ellipsoid_splinetracer_cpp_extension
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/splinetracer/extension"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
