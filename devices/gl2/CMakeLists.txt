## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.11)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_LIST_DIR})
  set(STANDALONE_SOURCE_TREE TRUE)
else()
  set(STANDALONE_SOURCE_TREE FALSE)
endif()

if (STANDALONE_SOURCE_TREE)
  message(STATUS "CMake version: ${CMAKE_VERSION}")
endif()

## Language setup ##

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_BUILD_TYPE_INIT Release)

set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_MESSAGE LAZY)

## Establish project ##

project(anari_library_visgl2 LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(GNUInstallDirs)

## Dependencies ##

find_package(anari REQUIRED)

## Build device target ##

add_library(${PROJECT_NAME} SHARED
  VisGL2Device.cpp
  VisGL2DeviceGlobalState.cpp
  VisGL2Library.cpp
  Object.cpp
  camera/Camera.cpp
  frame/Frame.cpp
  renderer/Renderer.cpp
  scene/Group.cpp
  scene/Instance.cpp
  scene/World.cpp
  scene/light/Light.cpp
  scene/surface/Surface.cpp
  scene/surface/geometry/Geometry.cpp
  scene/surface/material/Material.cpp
  scene/surface/material/sampler/Sampler.cpp
  scene/volume/Volume.cpp
  scene/volume/spatial_field/SpatialField.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE anari::helium glm_visrtx)

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME} EXPORT_MACRO_NAME "VISGL2_EXPORT")

target_include_directories(${PROJECT_NAME}
PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

## Code generation ##

anari_generate_queries(
  DEVICE_TARGET ${PROJECT_NAME}
  CPP_NAMESPACE visgl2
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/VisGL2Definitions.json
)

## Install ##

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
