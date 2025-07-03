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

message(STATUS "CMake version: ${CMAKE_VERSION}")

set(CMAKE_BUILD_TYPE_INIT "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if (CMAKE_VERSION VERSION_LESS "3.23")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
  else()
    set(CMAKE_CUDA_ARCHITECTURES all-major)
  endif()
endif()

## Establish project ##

project(VisRTX VERSION 0.13.0 LANGUAGES C CXX)

enable_testing()

include(GNUInstallDirs)

## Options ####################################################################

option(VISRTX_BUILD_TSD "Build TSD sample applications" OFF)
mark_as_advanced(VISRTX_BUILD_TSD)

option(
  VISRTX_ENABLE_MDL_SUPPORT
  "Whether to enable NVIDIA MDL support for VisRTX materials"
  OFF
)

## Dependencies ###############################################################

# ANARI-SDK
set(ANARI_REQUIRED_VERSION 0.15.0)
find_package(anari ${ANARI_REQUIRED_VERSION} REQUIRED)

# glm
find_package(glm QUIET)
if (TARGET glm::glm)
  message(STATUS "Found glm: (external) ${glm_DIR}")
else()
  # Use locally provided version of glm
  set(glm_DIR ${CMAKE_CURRENT_LIST_DIR}/external/glm/lib/cmake/glm)
  find_package(glm REQUIRED)
  message(STATUS "Found glm: (internal) ${glm_DIR}")
endif()
mark_as_advanced(glm_DIR)

add_library(glm_visrtx INTERFACE IMPORTED)
target_link_libraries(glm_visrtx INTERFACE glm::glm)
target_compile_definitions(glm_visrtx INTERFACE GLM_ENABLE_EXPERIMENTAL)
if(WIN32)
  target_compile_definitions(glm_visrtx INTERFACE _USE_MATH_DEFINES NOMINMAX)
endif()

# Other included dependencies
add_subdirectory(external)

## Build targets ##############################################################

option(VISRTX_BUILD_RTX_DEVICE "Build CUDA/OptiX device" ON)
option(VISRTX_BUILD_GL_DEVICE "Build OpenGL device" ON)

set(VISRTX_CMAKE_INSTALL_DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

add_subdirectory(devices)

if (VISRTX_BUILD_RTX_DEVICE)
  add_subdirectory(examples)
endif()
add_subdirectory(tests)
add_subdirectory(shaders)
if (VISRTX_BUILD_TSD)
  add_subdirectory(tsd)
endif()

## Configure CMake find_package() config files ################################

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
INSTALL_DESTINATION
  ${VISRTX_CMAKE_INSTALL_DESTINATION}
)

write_basic_package_version_file(
    "${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
DESTINATION
  ${VISRTX_CMAKE_INSTALL_DESTINATION}
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cmake
  DESTINATION
    ${VISRTX_CMAKE_INSTALL_DESTINATION}
  FILES_MATCHING
    PATTERN *.cmake
)

option(VISRTX_PRECOMPILE_SHADERS "Run visrtxCompileShaders after install?" ON)
if (TARGET visrtxCompileShaders AND VISRTX_PRECOMPILE_SHADERS)
  install(CODE "message(STATUS \"Precompile VisRTX OptiX shaders\")")
  install(CODE
    "execute_process(COMMAND ${CMAKE_BINARY_DIR}/visrtxCompileShaders)"
  )
endif()

