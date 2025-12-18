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

find_program(SLANGC slangc
  DOC "Path to the slangc executable.")

# this macro defines cmake rules that execute the following three steps:
# 1) compile the given slang file ${slang_file} to an intermediary PTX file
# 2) use the 'bin2c' tool (that comes with CUDA) to
#    create a second intermediary (.c-)file which defines a const string variable
#    (named '${c_var_name}') whose (constant) value is the PTX output
#    from the previous step.
# 3) assign the name of the intermediary .c file to the cmake variable
#    'output_var', which can then be added to cmake targets.
macro(slang_compile_and_embed output_var slang_file)
  set(ADDITIONAL_FILES ${ARGN})
    # Initialize a new list to store the modified additional files
  set(INCLUDE_DIRECTORIES)

  # Loop through each file in ADDITIONAL_FILES and append -I
  foreach(file ${ADDITIONAL_FILES})
      list(APPEND INCLUDE_DIRECTORIES "-I${file}")
  endforeach()

  set(c_var_name ${output_var})
  set(ptx_file ${output_var}.ptx)
  set(intermediate_file1 ${output_var}_nopre.cu)
  set(intermediate_file2 ${output_var}.cu)
  add_custom_command(
    OUTPUT ${ptx_file}
    # COMMAND ${SLANGC} ${slang_file} ${INCLUDE_DIRECTORIES} -o ${intermediate_file1} -dump-intermediates -line-directive-mode none && cat /opt/optix/preamble.h ${intermediate_file1} > ${intermediate_file2} && nvcc -I${OptiX_INSTALL_DIR}/include --ptx ${intermediate_file2} -ccbin $ENV{CC}

    COMMAND ${SLANGC} ${slang_file} ${INCLUDE_DIRECTORIES} -o ${ptx_file} -dump-intermediates -line-directive-mode none -Xnvrtc... -I${OptiX_INSTALL_DIR}/include

    # COMMAND ${SLANGC} ${slang_file} -o ${intermediate_file1} -dump-intermediates -line-directive-mode none && cat /opt/optix/preamble.h ${intermediate_file1} > ${intermediate_file2} && nvcc -I${OptiX_INSTALL_DIR}/include -lineinfo --ptx ${intermediate_file2} -ccbin /usr/bin/gcc-11
    DEPENDS ${slang_file} ${ARGN}
    COMMENT "compile ptx from ${slang_file}"
  )
  set(embedded_file ${ptx_file}_embedded.c)
  add_custom_command(
    OUTPUT ${embedded_file}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
    DEPENDS ${ptx_file}
    COMMENT "embed ptx from ${slang_file}"
  )
  set(${output_var} ${embedded_file})
endmacro()

# this macro defines cmake rules that execute the following three steps:
# 1) compile the given slang file ${slang_file} to an intermediary PTX file
# 2) use the 'bin2c' tool (that comes with CUDA) to
#    create a second intermediary (.c-)file which defines a const string variable
#    (named '${c_var_name}') whose (constant) value is the PTX output
#    from the previous step.
# 3) assign the name of the intermediary .c file to the cmake variable
#    'output_var', which can then be added to cmake targets.
macro(cuda_compile_and_embed output_var cuda_file)
  set(ADDITIONAL_FILES ${ARGN})
    # Initialize a new list to store the modified additional files
  set(INCLUDE_DIRECTORIES)

  # Loop through each file in ADDITIONAL_FILES and append -I
  foreach(file ${ADDITIONAL_FILES})
      list(APPEND INCLUDE_DIRECTORIES "-I${file}")
  endforeach()

  set(c_var_name ${output_var})
  set(ptx_file ${output_var}.ptx)
  set(intermediate_file1 ${output_var}_nopre.cu)
  set(intermediate_file2 ${output_var}.cu)
  add_custom_command(
    OUTPUT ${ptx_file}
    COMMAND nvcc -I${OptiX_INSTALL_DIR}/include --ptx ${cuda_file} -o ${ptx_file} -ccbin $ENV{CC}
    DEPENDS ${cuda_file} ${ARGN}
    COMMENT "compile ptx from ${cuda_file}"
  )
  set(embedded_file ${ptx_file}_embedded.c)
  add_custom_command(
    OUTPUT ${embedded_file}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
    DEPENDS ${ptx_file}
    COMMENT "embed ptx from ${cuda_file}"
  )
  set(${output_var} ${embedded_file})
endmacro()
