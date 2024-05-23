# Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

option(VISRTX_BUILD_INTERACTIVE_EXAMPLE "Build interactive example" OFF)
if (NOT VISRTX_BUILD_INTERACTIVE_EXAMPLE)
  return()
endif()

## Fetch match3D ##

include(FetchContent)

set(match3D_LOC ${CMAKE_BINARY_DIR}/match3D_src)

FetchContent_Populate(match3D_src
  URL https://github.com/jeffamstutz/match3D/archive/refs/heads/main.zip
  SOURCE_DIR ${match3D_LOC}
)
set(match3D_DIR ${match3D_LOC}/cmake)
set(match3D_CACHE_IN_SOURCE_TREE ON)
find_package(match3D REQUIRED COMPONENTS stb_image)

mark_as_advanced(
  FETCHCONTENT_BASE_DIR
  FETCHCONTENT_FULLY_DISCONNECTED
  FETCHCONTENT_QUIET
  FETCHCONTENT_UPDATES_DISCONNECTED
)

## Find CUDA for frame buffer interop ##

find_package(CUDAToolkit REQUIRED)

## Find VisRTX to use extension feature testing utility ##

find_package(VisRTX REQUIRED)

## viewer app ##

project(viewer)
add_executable(${PROJECT_NAME}
  main.cpp
  Orbit.cpp
  Scene.cpp
  ui_scenes.cpp
  Viewer.cpp
)
target_link_libraries(${PROJECT_NAME} PRIVATE
  match3D::match3D
  VisRTX::anari_library_visrtx
  glm_visrtx
  CUDA::cudart
  tiny_obj_loader
)
