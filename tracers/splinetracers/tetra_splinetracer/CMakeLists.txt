#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
enable_language(CUDA)

set(slang_dir ${CMAKE_CURRENT_SOURCE_DIR}/slang)
set(main_slang_dir ${CMAKE_CURRENT_SOURCE_DIR}/../slang)
slang_compile_and_embed(ptx_code_file
  ${slang_dir}/triangle.slang
  ${main_slang_dir}
  ${slang_dir}
  # ${main_slang_dir}/safe-math.slang
  # ${main_slang_dir}/sh.slang
  # ${main_slang_dir}/optix.slang
  # ${main_slang_dir}/optix-intrinsics.slang
  # ${main_slang_dir}/fast-math.slang
  # ${slang_dir}/spline-machine.slang
  # ${slang_dir}/tri-intersect.slang
)


pybind11_add_module(tetra_splinetracer_cpp_extension 
  ${ptx_code_file}
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
  TriangleMesh.cpp
  TriangleMesh.h
  # CollectIds.h
  # CollectIds.cpp
  exception.h
  py_binding.cpp
  # ${cmake_options}
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-lto")

# set_target_properties( splinetracer
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#
# set_target_properties( splinetracer_cpp_extension
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(tetra_splinetracer_cpp_extension PRIVATE
  ${CUDA_LIBRARIES} ${TORCH_LIBRARIES}
  m
  )
set_target_properties(tetra_splinetracer_cpp_extension
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/splinetracer/extension"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
