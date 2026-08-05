# Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

#### Establish project ####

set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

project(visrtx_device VERSION 0.14.0 LANGUAGES CXX CUDA)

include(GenerateExportHeader)
include(CMakeDependentOption)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(appendSearchPaths)
include(EmbedPTX)

#### RTX device options ####

# Override for a dev build that must coexist with an official install, e.g.
# -DVISRTX_LIBRARY_NAME=visrtx_dev.
set(VISRTX_LIBRARY_NAME visrtx CACHE STRING
  "ANARI library name: filename libanari_library_<name>.so and entrypoint anari_library_<name>_new_library")
if(NOT VISRTX_LIBRARY_NAME MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
  message(FATAL_ERROR
    "VISRTX_LIBRARY_NAME must be a valid C identifier (it is pasted into a symbol name): '${VISRTX_LIBRARY_NAME}'")
endif()

option(VISRTX_ENABLE_NEURAL "Enable Neural Graphics Primitives" OFF)
option(VISRTX_ENABLE_NVTX "Enable NVTX profiling instrumentation" OFF)
mark_as_advanced(VISRTX_ENABLE_NVTX)
option(VISRTX_ENABLE_NVML "Enable NVML for runtime driver checks" ON)
mark_as_advanced(VISRTX_ENABLE_NVML)
option(
  VISRTX_ENABLE_MDL_SUPPORT
  "Whether to enable NVIDIA MDL support for VisRTX materials"
  OFF
)
option(VISRTX_USE_MDL_FOR_PHYSICALLY_BASED "Use MDL for physically based materials" OFF)
cmake_dependent_option(
  VISRTX_ENABLE_MATERIALX_SUPPORT
  "Whether to enable MaterialX (.mtlx) materials (requires MDL support)"
  OFF
  "VISRTX_ENABLE_MDL_SUPPORT"
  OFF
)

# Enable possibly costly debug support
option(VISRTX_ENABLE_HDRI_SAMPLING_DEBUG "Enable HDRI sampling debug support" OFF)
mark_as_advanced(VISRTX_ENABLE_HDRI_SAMPLING_DEBUG)

#### Resolve common dependencies ####

find_package(anari 0.16.0 REQUIRED)
find_package(CUDAToolkit 12 REQUIRED)

# Minimum GPU the built device runs on. Every OptiX version VisRTX builds
# against accepts SM 5.0 (Maxwell) as its floor, so 50 is the widest reach and
# costs nothing over a higher value. Raise it to 75 for Turing and newer only:
# CUDA 13 removed pre-Turing code generation, so that is mandatory there.
if(CUDAToolkit_FOUND)
  if(CUDAToolkit_VERSION_MAJOR LESS_EQUAL 12)
    set(VISRTX_MIN_ARCH 50 CACHE STRING "Minimum CUDA compute capability (50, 75, 86, ...)")
  else()
    set(VISRTX_MIN_ARCH 75 CACHE STRING "Minimum CUDA compute capability (50, 75, 86, ...)")
  endif()
endif()

if(NOT VISRTX_MIN_ARCH MATCHES "^[0-9][0-9]+$" OR VISRTX_MIN_ARCH LESS 50)
  message(FATAL_ERROR
    "VISRTX_MIN_ARCH must be a compute capability of 50 or above (OptiX's own floor), got '${VISRTX_MIN_ARCH}'")
endif()

# Cooperative vectors need Ada. Raise the floor rather than overriding any
# single consumer of it, so the host and OptiX module PTX targets and the MDL
# backend stay in agreement. A floor already above 89 is left alone.
#
# This is a normal (non-cache) set: it shadows the cached value for this
# configure and its subdirectories, so every consumer below sees 89, while the
# cache keeps the user's original entry. Reconfigure re-raises cleanly, so the
# cache never silently disagrees with the build.
if(VISRTX_ENABLE_NEURAL AND VISRTX_MIN_ARCH LESS 89)
  message(STATUS
    "VISRTX_ENABLE_NEURAL raises VISRTX_MIN_ARCH from ${VISRTX_MIN_ARCH} to 89")
  set(VISRTX_MIN_ARCH 89)
endif()

# Finally validate MIN_ARCH againt the target CUDA toolkit
if(VISRTX_MIN_ARCH LESS 75 AND CUDAToolkit_VERSION_MAJOR GREATER 12)
  message(FATAL_ERROR
    "VISRTX_MIN_ARCH=${VISRTX_MIN_ARCH} requires CUDA 12.x: CUDA ${CUDAToolkit_VERSION_MAJOR} "
    "removed code generation for pre-Turing architectures. "
    "Build with CUDA 12, or configure with -DVISRTX_MIN_ARCH=75.")
endif()
find_package(glm MODULE REQUIRED)

## Get OptiX headers ##

set(OPTIX_FETCH_VERSION "7.7" CACHE STRING "Version of OptiX to get")
set_property(CACHE OPTIX_FETCH_VERSION
  PROPERTY STRINGS "7.7" "8.0" "8.1" "9.0")

if(${OPTIX_FETCH_VERSION} STREQUAL "7.7")
  set(OPTIX_URL https://github.com/NVIDIA/optix-dev/archive/refs/tags/v7.7.0.zip)
elseif(${OPTIX_FETCH_VERSION} STREQUAL "8.0")
  set(OPTIX_URL https://github.com/NVIDIA/optix-dev/archive/refs/tags/v8.0.0.zip)
elseif(${OPTIX_FETCH_VERSION} STREQUAL "8.1")
  set(OPTIX_URL https://github.com/NVIDIA/optix-dev/archive/refs/tags/v8.1.0.zip)
elseif(${OPTIX_FETCH_VERSION} STREQUAL "9.0")
  set(OPTIX_URL https://github.com/NVIDIA/optix-dev/archive/refs/tags/v9.0.0.zip)
else()
  message(FATAL_ERROR "Invalid/unknown version of OptiX selected")
endif()

if(VISRTX_ENABLE_NEURAL AND NOT ${OPTIX_FETCH_VERSION} STREQUAL "9.0")
  message(FATAL_ERROR "OptiX 9.0+ must be used to enable Neural Graphics Primitives support in VisRTX")
endif()

anari_sdk_fetch_project(NAME optix_headers URL ${OPTIX_URL})
appendSearchPaths(${optix_headers_LOCATION})

find_package(OptiX REQUIRED)

#### Validate Neural Graphics Primitives support ####

#### Setup CMAKE_CUDA_ARCHITECTURES for device compilation ####

# Host-side kernels ship as PTX only, at the floor architecture. The CUDA
# driver JIT-compiles compute_<VISRTX_MIN_ARCH> PTX to whatever GPU is present
# at launch (any GPU at or above the floor), so one build runs everywhere
# without baking a cubin per architecture. This mirrors the OptiX module PTX
# target and keeps the two in agreement: both are compute_<floor>, so a GPU at
# the floor JITs the host kernels and the OptiX modules alike instead of
# loading one and failing on the other. Build time and binary size stay flat
# regardless of how many architectures the toolkit knows about.
#
# Override with -DCMAKE_CUDA_ARCHITECTURES=<...> to emit cubins instead (e.g.
# "75-real;80-real" for offline, JIT-free launches, or "all-major" to restore
# the per-arch cubin list). This is a host-kernel-only knob: the OptiX module
# PTX targets are pinned to compute_<VISRTX_MIN_ARCH>-virtual in
# device/CMakeLists.txt and do NOT follow this override — a -real or multi-arch
# value here would emit cubins/multi-blobs optixModuleCreate cannot load. The MDL
# backend's sm_version likewise tracks VISRTX_MIN_ARCH (see libmdl/CMakeLists.txt),
# not this variable, so all three consumers stay in agreement regardless of the
# host-side override.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if (CMAKE_VERSION VERSION_LESS "3.23")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
  else()
    set(CMAKE_CUDA_ARCHITECTURES ${VISRTX_MIN_ARCH}-virtual)
  endif()
endif()

message(STATUS "Compiling CUDA with architectures '${CMAKE_CUDA_ARCHITECTURES}'")

#### Subdirectories ####

add_subdirectory(external)
add_subdirectory(shaders)
if(VISRTX_ENABLE_MDL_SUPPORT)
  add_subdirectory(libmdl)
endif()
if(VISRTX_ENABLE_MATERIALX_SUPPORT)
  # MaterialX must resolve out-of-tree (system/package config). An in-tree
  # FetchContent build is never installed, so it emits no build-tree package
  # config and its target names collide with the copy USD's pxrConfig imports
  # via find_dependency(MaterialX) — the two cannot coexist in one build.
  find_package(MaterialX 1.39 REQUIRED)
  # Compile-time last resort of the runtime search chain (ADR 0008); also the
  # distribution the transcoder unit tests pin.
  set(MATERIALX_LIBRARIES_DIR "${MaterialX_DIR}/../../../libraries" CACHE PATH "" FORCE)
endif()
add_subdirectory(device)
add_subdirectory(apps)
