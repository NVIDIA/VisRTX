## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
  message(STATUS "CMake version: ${CMAKE_VERSION}")
  set(ANARI_REQUIRED_VERSION 0.11.0)
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

set(CMAKE_CUDA_ARCHITECTURES all)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

## Global options ##

option(TSD_ENABLE_SERIALIZATION "Use conduit to serialize contexts" OFF)
option(TSD_USE_HDF5 "Support loading AMR grids from HDF5" OFF)
option(TSD_USE_ASSIMP "Support loading geometry with libassimp" OFF)

## Establish project ##

project(tsd LANGUAGES CXX)

find_package(anari ${ANARI_REQUIRED_VERSION} REQUIRED COMPONENTS viewer)

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
