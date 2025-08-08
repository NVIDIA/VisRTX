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

include(GenerateExportHeader)

## Establish project ##

project(anari_library_visrtx VERSION ${PROJECT_VERSION} LANGUAGES CXX CUDA)

## RTX device options ##

option(VISRTX_ENABLE_NEURAL "Enable Neural Graphics Primitives" OFF)
option(VISRTX_ENABLE_NVTX "Enable NVTX profiling instrumentation" OFF)
mark_as_advanced(VISRTX_ENABLE_NVTX)

# Enable possibly costly debug support
option(VISRTX_ENABLE_HDRI_SAMPLING_DEBUG "Enable HDRI sampling debug support" OFF)
mark_as_advanced(VISRTX_ENABLE_HDRI_SAMPLING_DEBUG)

## Compilation setup ##

set(VISRTX_CUDA_COMPILE_OPTIONS
  "--expt-relaxed-constexpr" "--extended-lambda" # Lambdas from device code
  # Silence the spamming of glm warning about `__host__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration`
  "-Xcudafe=--diag_suppress=20012"
  # Silence the multitude of warnings about deprecated/to be removed GPU architectures
  "-Wno-deprecated-gpu-targets"
  # Debug info support
  "$<$<CONFIG:DEBUG>:-O0;-lineinfo>" # -G would be nice for debug, but does not work with glm...
  "$<$<CONFIG:RELWITHDEBINFO>:-lineinfo>"
  # Neural Graphics Primitives support
  "$<$<BOOL:${VISRTX_ENABLE_NEURAL}>:-DVISRTX_USE_NEURAL>"
  # MDL support
  "$<$<BOOL:${VISRTX_ENABLE_MDL_SUPPORT}>:-DUSE_MDL>"
  "$<$<BOOL:${VISRTX_ENABLE_HDRI_SAMPLING_DEBUG}>:-DVISRTX_ENABLE_HDRI_SAMPLING_DEBUG>"
)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN TRUE)

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

## Validate Neural Graphics Primitives support ##

if(VISRTX_ENABLE_NEURAL AND NOT ${OPTIX_FETCH_VERSION} STREQUAL "9.0")
  message(FATAL_ERROR "OptiX 9.0+ must be used to enable Neural Graphics Primitives support in VisRTX")
endif()

## Find dependencies ##

find_package(CUDAToolkit 12 REQUIRED)
find_package(OptiX REQUIRED)
if(VISRTX_ENABLE_MDL_SUPPORT)
  find_package(MDL_SDK REQUIRED)
  option(VISRTX_USE_MDL_FOR_PHYSICALLY_BASED "Use MDL for physically based materials" OFF)
endif()

include(EmbedPTX)

## Build library target ##

function(embed_resource resource_file_name source_file_name variable_name)
    file(READ ${resource_file_name} hex_content HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1, " content ${hex_content})
    set(array_definition "extern \"C\" const unsigned char ${variable_name}[] = {${content}0x00};")
    file(WRITE "${source_file_name}" "${array_definition}")
endfunction()

if(VISRTX_ENABLE_MDL_SUPPORT)
    embed_resource(${CMAKE_SOURCE_DIR}/shaders/visrtx/default.mdl "${CMAKE_CURRENT_BINARY_DIR}/mdl/default.mdl.cpp" "VISRTX_DEFAULT_MDL")
    embed_resource(${CMAKE_SOURCE_DIR}/shaders/visrtx/physically_based.mdl "${CMAKE_CURRENT_BINARY_DIR}/mdl/physically_based.mdl.cpp" "VISRTX_PHYSICALLY_BASED_MDL")
endif()

project_add_library(SHARED)

set(SOURCES
  Object.cpp
  optix_visrtx.cpp
  VisRTXDevice.cpp
  VisRTXFeatureUtility.cpp
  VisRTXLibrary.cpp

  array/Array.cpp
  array/Array1D.cpp
  array/Array2D.cpp
  array/Array3D.cpp
  array/ObjectArray.cpp
  array/UploadableArray.cpp

  camera/Camera.cpp
  camera/Orthographic.cpp
  camera/Perspective.cpp
  camera/UnknownCamera.cpp

  frame/Denoiser.cu
  frame/Frame.cu

  renderer/AmbientOcclusion.cpp
  renderer/Debug.cpp
  renderer/DiffusePathTracer.cpp
  renderer/DirectLight.cpp
  renderer/Raycast.cpp
  renderer/Renderer.cpp
  renderer/Test.cpp
  renderer/UnknownRenderer.cpp

  shaders/MatteShader.cpp
  shaders/PhysicallyBasedShader.cpp

  scene/Group.cpp
  scene/Instance.cpp
  scene/World.cpp
  scene/ComputeTangent.cu

  scene/light/Directional.cpp
  scene/light/HDRI.cu
  scene/light/Light.cpp
  scene/light/Point.cpp
  scene/light/Spot.cpp
  scene/light/UnknownLight.cpp

  scene/surface/Surface.cpp

  scene/surface/geometry/Cone.cpp
  scene/surface/geometry/Curve.cpp
  scene/surface/geometry/Cylinder.cpp
  scene/surface/geometry/Geometry.cpp
  $<$<BOOL:${VISRTX_ENABLE_NEURAL}>:scene/surface/geometry/Neural.cpp>
  scene/surface/geometry/Quad.cpp
  scene/surface/geometry/Sphere.cu
  scene/surface/geometry/Triangle.cpp
  scene/surface/geometry/UnknownGeometry.cpp

  scene/surface/material/Material.cpp
  scene/surface/material/Matte.cpp
  scene/surface/material/PBR.cpp
  scene/surface/material/UnknownMaterial.cpp

  scene/surface/material/sampler/CompressedImage2D.cpp
  scene/surface/material/sampler/Image1D.cpp
  scene/surface/material/sampler/Image2D.cpp
  scene/surface/material/sampler/Image3D.cpp
  scene/surface/material/sampler/PrimitiveSampler.cpp
  scene/surface/material/sampler/Sampler.cpp
  scene/surface/material/sampler/TransformSampler.cpp
  scene/surface/material/sampler/UnknownSampler.cpp

  scene/volume/TransferFunction1D.cpp
  scene/volume/UnknownVolume.cpp
  scene/volume/Volume.cpp

  scene/volume/space_skipping/UniformGrid.cu

  scene/volume/spatial_field/SpatialField.cpp
  scene/volume/spatial_field/NvdbRegularField.cpp
  scene/volume/spatial_field/StructuredRegularField.cpp
  scene/volume/spatial_field/UnknownSpatialField.cpp

  utility/CudaImageTexture.cpp
  utility/DeferredArrayUploadBuffer.cpp
  utility/instrument.cpp
  utility/MemoryAllocation.cpp
  utility/PPM.cpp

  $<$<BOOL:${VISRTX_ENABLE_MDL_SUPPORT}>:
    mdl/Logger.cpp
    mdl/ptx.cpp
    mdl/MaterialRegistry.cpp
    mdl/SamplerRegistry.cpp
    scene/surface/material/MDL.cpp
    scene/surface/material/PhysicallyBasedMDL.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/mdl/default.mdl.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/mdl/physically_based.mdl.cpp
  >

)

if(VISRTX_ENABLE_MDL_SUPPORT)
  set_source_files_properties(
    scene/surface/material/Material.cpp
    PROPERTIES COMPILE_DEFINITIONS $<$<BOOL:${VISRTX_USE_MDL_FOR_PHYSICALLY_BASED}>:USE_MDL_FOR_PHYSICALLY_BASED>
  )
endif()

project_sources(PRIVATE ${SOURCES})
set(CUDA_SOURCES ${SOURCES})
list(FILTER CUDA_SOURCES INCLUDE REGEX ".*\\.cu$")

set_source_files_properties(${CUDA_SOURCES}
  PROPERTIES COMPILE_OPTIONS "${VISRTX_CUDA_COMPILE_OPTIONS}"
)

generate_export_header(${PROJECT_NAME}
  EXPORT_MACRO_NAME "VISRTX_DEVICE_INTERFACE"
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)

project_include_directories(
  SYSTEM
    PRIVATE
      ${PROJECT_INCLUDE_DIRS}
)

project_link_libraries(
PUBLIC
  anari::anari
PRIVATE
  anari::helium
  glm_visrtx
  OptiX::OptiX
  CUDA::cuda_driver
  CUDA::curand
  CUDA::nvml
  fmt::fmt
  nonstd::expected-lite
  nonstd::scope-lite
  nonstd::span-lite
  libnanovdb
  $<$<BOOL:${VISRTX_ENABLE_NVTX}>:CUDA::nvToolsExt>
  $<$<BOOL:${VISRTX_ENABLE_MDL_SUPPORT}>:libmdl>
  $<$<BOOL:${VISRTX_ENABLE_MDL_SUPPORT}>:MDL_SDK::MDL_SDK>
  $<$<BOOL:${VISRTX_ENABLE_MDL_SUPPORT}>:stb_image>
)

project_compile_definitions(
PRIVATE
  VISRTX_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  VISRTX_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  VISRTX_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  $<$<BOOL:${VISRTX_ENABLE_NVTX}>:USE_NVTX>
  $<$<BOOL:${VISRTX_ENABLE_MDL_SUPPORT}>:USE_MDL>
  $<$<BOOL:${VISRTX_ENABLE_NEURAL}>:VISRTX_USE_NEURAL>
  $<$<BOOL:${VISRTX_ENABLE_HDRI_SAMPLING_DEBUG}>:VISRTX_ENABLE_HDRI_SAMPLING_DEBUG>
)

if(NOT WIN32)
  option(VISRTX_SHADER_PARALLEL_BUILD "Build shaders in parallel" ON)
  if (VISRTX_SHADER_PARALLEL_BUILD)
    project_compile_definitions(PRIVATE VISRTX_PARALLEL_MODULE_BUILD)
  endif()
endif()

project_compile_definitions(PRIVATE visrtx_EXPORTS)

## Generate OptiX programs as embedded PTX ##

function(GenerateEmbeddedPTX DIR BASE_NAME)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs ENTRIES)
  cmake_parse_arguments(
    GEN_EMBEDDED_PTX
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  set(INPUT_TARGET ${BASE_NAME}_ptx)

  set(INPUT_CU_FILE ${DIR}/${BASE_NAME}_ptx.cu)
  set(OUTPUT_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${BASE_NAME}_ptx.h)

  if (GEN_EMBEDDED_PTX_ENTRIES)
    string(REPLACE ";" "," GEN_EMBEDDED_PTX_ENTRIES "${GEN_EMBEDDED_PTX_ENTRIES}")
    set(GEN_EMBEDDED_PTX_ARGS "--entries" "${GEN_EMBEDDED_PTX_ENTRIES}")
  endif()

  set_source_files_properties(${INPUT_CU_FILE}
    # -rdc=true is for relocatable code, can be resolved at link time.
    PROPERTIES COMPILE_OPTIONS "${VISRTX_CUDA_COMPILE_OPTIONS};-rdc=true;${GEN_EMBEDDED_PTX_ARGS}"
  )

  add_library(${INPUT_TARGET} OBJECT ${INPUT_CU_FILE})
  target_link_libraries(${INPUT_TARGET}
  PRIVATE
    anari::anari
    anari::helium
    glm_visrtx
    OptiX::OptiX
    CUDA::curand
    libnanovdb
  )
  target_include_directories(${INPUT_TARGET}
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_LIST_DIR}
    ${MDL_SDK_INCLUDE_DIRS}
  )

  if(VISRTX_ENABLE_NEURAL)
    set(CUDA_ARCHS 89-virtual)
  else()
    set(CUDA_ARCHS 52-virtual)
  endif()

  set_target_properties(${INPUT_TARGET} PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES ${CUDA_ARCHS}
  )

  EmbedPTX(
    OUTPUT_HEADER_FILE ${OUTPUT_HEADER}
    INPUT_TARGET ${INPUT_TARGET}
    OUTPUT_TARGETS ${PROJECT_NAME}
  )
endfunction()

GenerateEmbeddedPTX(renderer AmbientOcclusion)
GenerateEmbeddedPTX(renderer Debug)
GenerateEmbeddedPTX(renderer DiffusePathTracer)
GenerateEmbeddedPTX(renderer DirectLight)
GenerateEmbeddedPTX(renderer Raycast)
GenerateEmbeddedPTX(renderer Test)
GenerateEmbeddedPTX(scene Intersectors)
GenerateEmbeddedPTX(shaders MatteShader ENTRIES __direct__callable__evalSurfaceMaterial)
GenerateEmbeddedPTX(shaders PhysicallyBasedShader ENTRIES __direct__callable__evalSurfaceMaterial)
if(VISRTX_ENABLE_MDL_SUPPORT)
  GenerateEmbeddedPTX(shaders MDLShader ENTRIES __direct__callable__evalSurfaceMaterial)
  GenerateEmbeddedPTX(shaders MDLTexture)
endif()


## Query code generation ##

file(GLOB DEVICE_JSONS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.json")
foreach(file ${DEVICE_JSONS})
  file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file} INPUT ${file})
endforeach()

anari_generate_queries(
  DEVICE_TARGET ${PROJECT_NAME}
  CPP_NAMESPACE visrtx
  JSON_ROOT_LOCATION ${CMAKE_CURRENT_BINARY_DIR}
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_BINARY_DIR}/visrtx_device.json
  JSON_EXTENSION_FILES ${DEVICE_JSONS}
)

## Install library + headers ##

if (VISRTX_USE_SOVERSION)
  set_target_properties(${PROJECT_NAME}
      PROPERTIES VERSION "${CMAKE_PROJECT_VERSION}" SOVERSION "${CMAKE_PROJECT_VERSION_MAJOR}")
endif()

install(TARGETS ${PROJECT_NAME}
  EXPORT VisRTX_Exports
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
FILES
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}_export.h
  ${CMAKE_CURRENT_LIST_DIR}/include/anari/ext/visrtx/visrtx.h
DESTINATION
  ${CMAKE_INSTALL_INCLUDEDIR}/anari/ext/visrtx
)

install(EXPORT VisRTX_Exports
  DESTINATION ${VISRTX_CMAKE_INSTALL_DESTINATION}
  NAMESPACE VisRTX::
)

## Build tools ##

add_subdirectory(tools)
