# Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(GenerateExportHeader)

## Establish project ##

project(anari_library_visrtx VERSION ${PROJECT_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)

## Find dependencies ##

find_package(CUDAToolkit 11.3.1 REQUIRED)
find_package(OptiX7 REQUIRED)

include(EmbedPTX)

option(VISRTX_ENABLE_NVTX "Enable NVTX profiling instrumentation" OFF)
mark_as_advanced(VISRTX_ENABLE_NVTX)

## Code generation

option(VISRTX_ENABLE_CODE_GEN "Enable Python code generation for query updates" OFF)
mark_as_advanced(VISRTX_ENABLE_CODE_GEN)
if (VISRTX_ENABLE_CODE_GEN)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_custom_target(generate_visrtx_device
    COMMAND ${Python3_EXECUTABLE} ${VISRTX_CODE_GEN_ROOT}/generate_queries.py
      --json ${VISRTX_CODE_GEN_ROOT}
      --prefix VisRTXDevice
      --device ${CMAKE_CURRENT_SOURCE_DIR}/visrtx_device.json
      --namespace visrtx
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS visrtx_device.json
  )

  add_custom_target(generate_visrtx_debug_objects
    COMMAND ${Python3_EXECUTABLE} ${VISRTX_CODE_GEN_ROOT}/generate_debug_objects.py
      --json ${VISRTX_CODE_GEN_ROOT}
      --device ${CMAKE_CURRENT_SOURCE_DIR}/visrtx_device.json
      --prefix VisRTXDevice
      --namespace visrtx
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS visrtx_device.json
  )

  add_custom_target(generate_all)
  add_dependencies(generate_all generate_visrtx_device)
  add_dependencies(generate_all generate_visrtx_debug_objects)
endif()

## Build library target ##

add_library(${PROJECT_NAME} SHARED
  Object.cpp
  optix_visrtx.cpp
  VisRTXDevice.cpp
  VisRTXDeviceDebugFactory.cpp
  VisRTXDeviceQueries.cpp

  array/Array.cpp
  array/Array1D.cpp
  array/Array2D.cpp
  array/Array3D.cpp
  array/ObjectArray.cpp

  camera/Camera.cpp
  camera/Orthographic.cpp
  camera/Perspective.cpp

  frame/Denoiser.cu
  frame/Frame.cu

  renderer/AmbientOcclusion.cpp
  renderer/Debug.cpp
  renderer/DiffusePathTracer.cpp
  renderer/Raycast.cpp
  renderer/Renderer.cpp
  renderer/SciVis.cpp

  scene/Group.cpp
  scene/Instance.cpp
  scene/World.cpp

  scene/light/Ambient.cpp
  scene/light/Directional.cpp
  scene/light/Light.cpp
  scene/light/Point.cpp

  scene/surface/Surface.cpp

  scene/surface/geometry/Cones.cpp
  scene/surface/geometry/Cylinders.cpp
  scene/surface/geometry/Geometry.cpp
  scene/surface/geometry/Quads.cpp
  scene/surface/geometry/Spheres.cpp
  scene/surface/geometry/Triangles.cpp

  scene/surface/material/Material.cpp
  scene/surface/material/Matte.cpp
  scene/surface/material/PBR.cpp
  scene/surface/material/TransparentMatte.cpp

  scene/surface/material/sampler/ColorMap.cpp
  scene/surface/material/sampler/Image2D.cpp
  scene/surface/material/sampler/PrimitiveSampler.cpp
  scene/surface/material/sampler/Sampler.cpp

  scene/volume/Volume.cpp
  scene/volume/SciVisVolume.cpp

  scene/volume/spatial_field/SpatialField.cpp
  scene/volume/spatial_field/StructuredRegularField.cpp

  utility/DeferredCommitBuffer.cpp
  utility/DeferredUploadBuffer.cpp
  utility/instrument.cpp
  utility/TimeStamp.cpp
)

generate_export_header(${PROJECT_NAME}
  EXPORT_MACRO_NAME "VISRTX_DEVICE_INTERFACE"
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

set_source_files_properties(frame/Frame.cu frame/Denoiser.cu
  PROPERTIES COMPILE_FLAGS "--extended-lambda"
)

set_source_files_properties(scene/Intersectors_ptx.cu
  PROPERTIES COMPILE_FLAGS "--expt-relaxed-constexpr"
)

target_include_directories(${PROJECT_NAME}
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
  anari::anari
PRIVATE
  anari::anari_utilities
  anari::anari_library_debug
  glm::glm
  OptiX7::OptiX7
  CUDA::cuda_driver
  CUDA::curand
  $<$<BOOL:${VISRTX_ENABLE_NVTX}>:CUDA::nvToolsExt>
)

target_compile_definitions(${PROJECT_NAME}
PRIVATE
  VISRTX_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  VISRTX_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  VISRTX_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  $<$<BOOL:${VISRTX_ENABLE_NVTX}>:USE_NVTX>
)

target_compile_definitions(${PROJECT_NAME} PRIVATE visrtx_EXPORTS)

## Generate OptiX programs as embedded PTX ##

function(GenerateEmbeddedPTX DIR BASE_NAME)
  set(INPUT_CU_FILE ${DIR}/${BASE_NAME}_ptx.cu)
  set(INPUT_TARGET ${BASE_NAME}_ptx_gen)
  set(OUTPUT_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${BASE_NAME}_ptx.h)

  add_library(${INPUT_TARGET} OBJECT ${INPUT_CU_FILE})
  target_link_libraries(${INPUT_TARGET}
  PRIVATE
    anari::anari
    anari::anari_utilities
    glm::glm
    OptiX7::OptiX7
    CUDA::curand
  )
  target_include_directories(${INPUT_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
  set_property(TARGET ${INPUT_TARGET} PROPERTY CUDA_PTX_COMPILATION ON)
  set_property(TARGET ${INPUT_TARGET} PROPERTY CUDA_ARCHITECTURES OFF)

  EmbedPTX(
    OUTPUT_HEADER_FILE ${OUTPUT_HEADER}
    INPUT_TARGET ${INPUT_TARGET}
    OUTPUT_TARGETS ${PROJECT_NAME}
  )
endfunction()

GenerateEmbeddedPTX(renderer AmbientOcclusion)
GenerateEmbeddedPTX(renderer Debug)
GenerateEmbeddedPTX(renderer DiffusePathTracer)
GenerateEmbeddedPTX(renderer Raycast)
GenerateEmbeddedPTX(renderer SciVis)

GenerateEmbeddedPTX(scene Intersectors)

## Install library + headers ##

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set_target_properties(${PROJECT_NAME}
    PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

install(TARGETS ${PROJECT_NAME}
  EXPORT VisRTX_Exports
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    NAMELINK_SKIP
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
FILES
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}_export.h
  ${CMAKE_CURRENT_LIST_DIR}/include/anari/backend/visrtx/visrtx.h
DESTINATION
  ${CMAKE_INSTALL_INCLUDEDIR}/anari/backend/visrtx
)
