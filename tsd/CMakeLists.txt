## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
  message(STATUS "CMake version: ${CMAKE_VERSION}")
  set(ANARI_REQUIRED_VERSION 0.15.0)
  set(TSD_STANDALONE_BUILD 1)
else()
  set(TSD_STANDALONE_BUILD 0)
endif()

## CMake setup ##

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE_INIT Release)
set(CMAKE_INSTALL_MESSAGE LAZY)

set(CMAKE_CUDA_ARCHITECTURES all-major)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

## Global options ##

option(BUILD_INTERACTIVE_APPS "Build interactive applications" ON)
option(TSD_USE_HDF5 "Support loading AMR grids from HDF5" OFF)
option(TSD_USE_ASSIMP "Support loading geometry with libassimp" OFF)
option(TSD_USE_SDL3 "Use SDL3 interop in render pipeline" ON)
option(TSD_USE_TBB "Use TBB code paths where relevant" ON)
option(TSD_USE_USD "Use OpenUSD where relevant" OFF)
option(TSD_USE_TORCH "Use PyTorch for importing neural geometries" OFF)
option(TSD_USE_VTK "Use VTK for importing VTK file formats" OFF)

if (APPLE)
  set(TSD_USE_CUDA OFF)
else()
  option(TSD_USE_CUDA "Use CUDA code paths where relevant" ON)
endif()

## Establish project ##

project(tsd LANGUAGES C CXX)

## Major dependencies ##

if (BUILD_INTERACTIVE_APPS)
  find_package(anari ${ANARI_REQUIRED_VERSION} REQUIRED COMPONENTS viewer)
else()
  find_package(anari ${ANARI_REQUIRED_VERSION} REQUIRED)
endif()

if (TSD_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

## Testing Setup ##

include(CTest)
if (BUILD_TESTING)
  enable_testing()
endif()

## Build Libraries ##

add_subdirectory(apps)
add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(tests)

## Cleanup CMake variables ##

include(cmake/mark_cache_variables_as_advanced.cmake)
