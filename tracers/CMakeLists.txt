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

# This enforces a particular version of CMake that we require to process the script files
# properly.  We rely on VERSION_GREATER_EQUAL which requires CMake v. 3.7.
cmake_minimum_required(VERSION 3.7 FATAL_ERROR)

# This sets up the name of our project.  For our purposes the main thing this controls is
# the name of the VS solution file.
project(splinetracers VERSION 1.0)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

# As of CMake 2.6 policies were introduced in order to provide a mechanism for
# adding backwards compatibility one feature at a time.  We will just specify
# that all policies will use version 2.8.12 semantics.
cmake_policy(VERSION 2.8.12)

if( POLICY CMP0072 )
  # FindOpenGL prefers GLVND by default when available
  cmake_policy(SET CMP0072 NEW)
endif()

if( POLICY CMP0074 )
  # find_package uses <PackageName>_ROOT variables.
  cmake_policy(SET CMP0074 NEW)
endif()

# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/third_party/CMake"
  ${CMAKE_MODULE_PATH}
  )

# Set the default build to Release.  Note this doesn't do anything for the VS
# default build target which defaults to Debug when you first start it.
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)


include(FetchContent)
# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_BUILD_TYPE "Release")

# find_library(libcnpy cnpy ${CONDA_PREFIX}/lib)
include_directories(${CONDA_PREFIX}/include)
include_directories(/usr/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

find_package(Torch REQUIRED)
#find_package(CGAL REQUIRED)
# find_package(pybind11 REQUIRED)

string(REGEX REPLACE "^([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1" TORCH_VERSION_CLEAN "${TORCH_VERSION}")
message(STATUS "Clean Torch version: ${TORCH_VERSION_CLEAN}")
if(${TORCH_VERSION_CLEAN} VERSION_GREATER "2.5.0")
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
  message(STATUS "Found Torch version: ${TORCH_VERSION}. CXX11ABI=1")
else()
  message(STATUS "Found Torch version: ${TORCH_VERSION}. CXX11ABI=0")
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

# Add pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1  # Change this to the desired version
)
FetchContent_MakeAvailable(pybind11)
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

# Tells CMake to build all the libraries as shared libraries by default.  This can be
# overrided by individual libraries later.
option(BUILD_SHARED_LIBS "Build shared libraries" ON)


##########
# Process our custom setup scripts here.

# Enable C++11. Needs to be done before the include of ConfigCompilerFlags.cmake below.
set(GCC_LIBSTDCPP11 ON)

# Check for specific machine/compiler options.
# include (ConfigCompilerFlags)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# Turn off the warning that NVCC issues when generating PTX from our CUDA samples.  This
# is a custom extension to the FindCUDA code distributed by CMake.
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

# For Xcode 5, gcc is actually clang, so we have to tell CUDA to treat the compiler as
# clang, so that it doesn't mistake it for something else.
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()

# CUDA 8 is broken for generating dependencies during configure
option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)

# Find at least a 5.0 version of CUDA.
find_package(CUDA 5.0 REQUIRED)

# CUDA 11.7 adds support for OptiX targetted input files.
if( CUDA_VERSION VERSION_GREATER_EQUAL 11.7 )
  option(SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" ON)
else()
  # You can try to force this on
  option(SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" OFF)
endif()

# This code looks funny, but CMake doesn't have an equality operator for boolean types
# (only integer and string). By doing NOT NOT VAL, you can force the original value into 0
# or 1 and allow the EQUAL operator to function correctly.
if( NOT ( NOT ( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT ) ) EQUAL ( NOT ( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ) ) )
  message( STATUS "Unsetting values associated with OptiX code generation" )
  # This allows us to reset dependent options if you change it.
  unset( SAMPLES_INPUT_GENERATE_OPTIXIR CACHE )
  unset( SAMPLES_INPUT_GENERATE_PTX CACHE )
  unset( GENERATE_DEBUG_DEVICE_CODE CACHE )
endif()
set(SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ${SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT} CACHE INTERNAL "Previous configured value" FORCE)

if( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT )
  option( SAMPLES_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" ON )
  option( SAMPLES_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" OFF )
else()
  option( SAMPLES_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" OFF )
  option( SAMPLES_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" ON )
endif()

# Determine if we are going to use the static CRT on windows.
if(WIN32)
  option(RELEASE_USE_STATIC_CRT "Build using the static CRT library" ON)
endif()

# Helper for fixing compiler flags
function(replace_flag var old_flag new_flag)
  string(REPLACE "${old_flag}" "${new_flag}" ${var} ${${var}})
  set(${var} "${${var}}" CACHE STRING "Default compiler flags" FORCE)
endfunction()

function(replace_flags old_flag new_flag)
  foreach(build "" _DEBUG _MINSIZEREL _RELEASE _RELWITHDEBINFO)
    replace_flag(CMAKE_C_FLAGS${build}   "${old_flag}" "${new_flag}")
    replace_flag(CMAKE_CXX_FLAGS${build} "${old_flag}" "${new_flag}")
  endforeach()
endfunction()

if(WIN32)
  if(RELEASE_USE_STATIC_CRT)
    replace_flags("/MD" "/MT")
  else()
    replace_flags("/MT" "/MD")
  endif()
endif(WIN32)

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CUDA_MIN_SM_TARGET sm_50 CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

function(optix_add_cuda_flag_config config flag)
  string(TOUPPER "${config}" config)
  list(FIND CUDA_NVCC_FLAGS${config} ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS${config} ${flag})
    set(CUDA_NVCC_FLAGS${config} ${CUDA_NVCC_FLAGS${config}} CACHE STRING ${CUDA_NVCC_FLAGS_DESCRIPTION} FORCE)
  endif()
endfunction()

function(optix_add_cuda_flag flag)
  optix_add_cuda_flag_config( "" ${flag} )
endfunction()

# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if( NOT PASSED_FIRST_CONFIGURE )
  set(CUDA_NVCC_FLAGS_DESCRIPTION "Semi-colon delimit multiple arguments.")
  string(REPLACE "sm_" "compute_" CUDA_MIN_SM_COMPUTE_TARGET ${CUDA_MIN_SM_TARGET})

  list(FIND CUDA_NVCC_FLAGS "-arch" index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -arch ${CUDA_MIN_SM_TARGET})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()

  #optix_add_cuda_flag("--use_fast_math")
  optix_add_cuda_flag("-lineinfo")
  if( DEFINED CMAKE_CONFIGURATION_TYPES )
    foreach( config ${CMAKE_CONFIGURATION_TYPES} )
      if( ${config} STREQUAL "Debug" )
        optix_add_cuda_flag_config( _${config} "-G" )
        optix_add_cuda_flag_config( _${config} "-O0" )
      endif()
    endforeach()
  else()
    if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
      optix_add_cuda_flag( "-G" )
      optix_add_cuda_flag( "-O0" )
    endif()
  endif()
  


  if( CUDA_VERSION VERSION_LESS "3.0" )
    optix_add_cuda_flag("--keep")
  endif()

  # Some CUDA 11.x toolkits erroneously complain about sm_50 being deprecated
  if(CUDA_VERSION VERSION_GREATER "11.0")
    optix_add_cuda_flag("-Wno-deprecated-gpu-targets")
  endif()

  if(CMAKE_CXX_STANDARD EQUAL 11)
    set(SAMPLES_NVRTC_CXX "-std=c++11")
  else()
    set(SAMPLES_NVRTC_CXX "")
  endif()
  set(CUDA_NVRTC_FLAGS ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} -lineinfo -default-device -rdc true -D__x86_64 CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
endif()

mark_as_advanced(CUDA_NVRTC_FLAGS)

# This passes a preprocessor definition to cl.exe when processing CUDA code.
if(USING_WINDOWS_CL)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Create a flag for mac which will allow apps to add the local cuda toolkit
# install path to the app's rpath.
if( APPLE )
  set( CUDA_TOOLKIT_RPATH_FLAG "-Wl,-rpath,${CUDA_TOOLKIT_ROOT_DIR}/lib" )
endif()

# Locate the NVRT distribution.  Search the SDK first, then look in the system.
set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../" CACHE PATH "Path to OptiX installed location.")

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)

# Add the path to the OptiX headers to our include paths.
include_directories(
  "${OptiX_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
  )

# Select whether to use NVRTC or NVCC to generate PTX
set(CUDA_NVRTC_ENABLED OFF CACHE BOOL "Use NVRTC to compile PTX at run-time instead of NVCC at build-time")
if( CUDA_NVRTC_ENABLED AND SAMPLES_INPUT_GENERATE_OPTIXIR )
  if( NOT SAMPLES_INPUT_GENERATE_PTX )
    message( SEND_ERROR "CUDA_NVRTC_ENABLED is only compatible with SAMPLES_INPUT_GENERATE_PTX. Please disable CUDA_NVRTC_ENABLED unless enabling SAMPLES_INPUT_GENERATE_PTX." )
  endif()
endif()

find_program(BIN2C bin2c
  DOC "Path to the cuda-sdk bin2c executable.")

##################################################################
# Macros
macro(cuda_compile_and_embed output_var generated_files_name)
    set(c_var_name ${output_var})
    set(generated_files "${${generated_files_name}}")
    list(GET "${generated_files_name}" 0 ptx_file)
    get_filename_component(embedded_file ${ptx_file}_embedded.cpp NAME)
    set(embedded_file "${BUILD_DIRECTORY}/${embedded_file}")
    add_custom_command(
        OUTPUT ${embedded_file}
        COMMAND ${BIN2C} --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
        DEPENDS ${ptx_files} ${generated_files}
        COMMENT "compiling (and embedding ptx from) ${cuda_file}")
    set(${output_var} ${embedded_file})
    list(APPEND "${generated_files_name}" ${embedded_file})
endmacro()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/configure_slang.cmake)

##################################################################
# SUtil compilation

set(SAMPLES_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx")
set(SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CUDA_GENERATED_OUTPUT_DIR ${SAMPLES_PTX_DIR})

if( WIN32 )
  string(REPLACE "/" "\\\\" SAMPLES_PTX_DIR ${SAMPLES_PTX_DIR})
else( WIN32 )
  if( USING_GNU_C AND NOT APPLE )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DM_PI=3.14159265358979323846" )
  endif()
endif( WIN32 )

set(SAMPLES_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda")

set(SAMPLES_SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/support")

# NVRTC include paths relative to the sample path
set(SAMPLES_RELATIVE_INCLUDE_DIRS "\\
  \"cuda\", \\
  \"sutil\", \\
  \".\", ")

# NVRTC absolute include paths to the headers used to build the samples
set(SAMPLES_ABSOLUTE_INCLUDE_DIRS "\\
  \"${OptiX_INCLUDE}\", \\
  \"${CUDA_INCLUDE_DIRS}\", ")

# Build a null-terminated option list for NVRTC
set(CUDA_NVRTC_OPTIONS)
foreach(flag ${CUDA_NVRTC_FLAGS})
  set(CUDA_NVRTC_OPTIONS "${CUDA_NVRTC_OPTIONS} \\\n  \"${flag}\",")
endforeach()
set(CUDA_NVRTC_OPTIONS "${CUDA_NVRTC_OPTIONS}")

# Path to sutil.h that all the samples need
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}
                     "${CMAKE_BINARY_DIR}/include"
                     ${CMAKE_CURRENT_BINARY_DIR}
                     ${CUDA_INCLUDE_DIRS}
                     )

set(SAMPLES_CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

# Helper macro to generate PTX from the CUDA files in sutil.
macro(OPTIX_sutil_compile_to_optix_input generated_files)
  if( SAMPLES_INPUT_GENERATE_OPTIXIR )
    message("sutil OPTIXIR")
    CUDA_WRAP_SRCS( sutil OPTIXIR generated_files2 ${ARGN} )
    list(APPEND ${generated_files} ${generated_files2})
  endif()
  if( SAMPLES_INPUT_GENERATE_PTX AND NOT CUDA_NVRTC_ENABLED)
    message("sutil PTX")
    CUDA_WRAP_SRCS( sutil PTX generated_files3 ${ARGN} )
    list(APPEND ${generated_files} ${generated_files3})
  endif()
  message("${generated_files} = ${${generated_files}}")
endmacro()

# These calls will group PTX and CUDA files into their own directories in the Visual
# Studio projects.
macro(OPTIX_add_source_groups)
  if( SAMPLES_INPUT_GENERATE_PTX )
    if( NOT CUDA_NVRTC_ENABLED )
      source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
    endif()
  endif()
  if( SAMPLES_INPUT_GENERATE_OPTIXIR )
    source_group("OptixIR Files"  REGULAR_EXPRESSION ".+\\.optixir$")
  endif()
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")
endmacro()

#########################################################
#  List of samples found in subdirectories.
#
# If you wish to start your own sample, you can copy one of the sample's directories.
# Just make sure you rename all the occurances of the sample's name in the C code as well
# and the CMakeLists.txt file.
# add_subdirectory(abseil-cpp)
#find_package(absl REQUIRED)
# find_package(tiny-cuda-nn REQUIRED)


add_subdirectory( splinetracers/tetra_splinetracer   )

# Third-party support libraries.
# add_subdirectory(splinetracers/support)

#################################################################

# Now that everything is done, indicate that we have finished configuring at least once.
# We use this variable to set certain defaults only on the first pass, so that we don't
# continually set them over and over again.
set(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already Configured once?")
