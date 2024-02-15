# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE_INIT "Release")

set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_MESSAGE LAZY)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if (CMAKE_VERSION VERSION_LESS "3.23")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
  endif()
endif()

project(VisRTX VERSION 0.8.0 LANGUAGES C CXX)

include(GNUInstallDirs)

option(
  VISRTX_SET_SOVERSION
  "(Unix) Set the library SOVERSION propert and create namelinks?"
  OFF
)
mark_as_advanced(VISRTX_SET_SOVERSION)

if (VISRTX_SET_SOVERSION)
  set(VISRTX_CMAKE_INSTALL_DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}
  )
else()
  set(VISRTX_CMAKE_INSTALL_DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
endif()

## Dependencies ###############################################################

# ANARI-SDK
set(ANARI_REQUIRED_VERSION 0.8.0)
find_package(anari ${ANARI_REQUIRED_VERSION} REQUIRED)

# CUDA
find_package(CUDAToolkit 11.3.1)
find_package(OptiX7)

# OpenGL
find_package(OpenGL)

# glm (locally included)
set(glm_DIR ${CMAKE_CURRENT_LIST_DIR}/external/glm/lib/cmake/glm)
find_package(glm REQUIRED)
mark_as_advanced(glm_DIR)
add_library(glm_visrtx INTERFACE IMPORTED)
target_link_libraries(glm_visrtx INTERFACE glm::glm)
if(WIN32)
  target_compile_definitions(glm_visrtx INTERFACE _USE_MATH_DEFINES NOMINMAX)
endif()

# Other included dependencies
add_subdirectory(external)

## Build targets ##############################################################

enable_testing()

add_subdirectory(devices)
add_subdirectory(examples)
add_subdirectory(tests)

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
