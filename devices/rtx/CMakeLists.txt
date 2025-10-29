# Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(visrtx_device VERSION 0.13.0 LANGUAGES CXX CUDA)

include(GenerateExportHeader)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(appendSearchPaths)
include(EmbedPTX)

#### RTX device options ####

option(
  VISRTX_ENABLE_MDL_SUPPORT
  "Whether to enable NVIDIA MDL support for VisRTX materials"
  OFF
)
option(VISRTX_ENABLE_NEURAL "Enable Neural Graphics Primitives" OFF)
option(VISRTX_ENABLE_NVTX "Enable NVTX profiling instrumentation" OFF)
mark_as_advanced(VISRTX_ENABLE_NVTX)

option(VISRTX_USE_MDL_FOR_PHYSICALLY_BASED "Use MDL for physically based materials" OFF)

# Enable possibly costly debug support
option(VISRTX_ENABLE_HDRI_SAMPLING_DEBUG "Enable HDRI sampling debug support" OFF)
mark_as_advanced(VISRTX_ENABLE_HDRI_SAMPLING_DEBUG)

#### Resolve common dependencies ####

find_package(anari 0.15.0 REQUIRED)
find_package(CUDAToolkit 12 REQUIRED)
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

anari_sdk_fetch_project(NAME optix_headers URL ${OPTIX_URL})
appendSearchPaths(${optix_headers_LOCATION})

find_package(OptiX REQUIRED)

#### Validate Neural Graphics Primitives support ####

if(VISRTX_ENABLE_NEURAL AND NOT ${OPTIX_FETCH_VERSION} STREQUAL "9.0")
  message(FATAL_ERROR "OptiX 9.0+ must be used to enable Neural Graphics Primitives support in VisRTX")
endif()

#### Subdirectories ####

add_subdirectory(external)
add_subdirectory(shaders)
if(VISRTX_ENABLE_MDL_SUPPORT)
  add_subdirectory(libmdl)
endif()
add_subdirectory(device)
add_subdirectory(apps)
