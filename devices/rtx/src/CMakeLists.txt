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

#### Establish project ####

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(anari_library_visrtx VERSION ${PROJECT_VERSION} LANGUAGES CXX CUDA)

#### Compilation setup ####

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if (CMAKE_VERSION VERSION_LESS "3.23")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
  else()
    set(CMAKE_CUDA_ARCHITECTURES all-major)
  endif()
endif()

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

#### Dependencies ####

find_package(nanovdb MODULE REQUIRED)

#### Build library target ####

function(embed_resource resource_file_name source_file_name variable_name)
  # Generate a unique target name based on the variable name
  set(target_name "embed_${variable_name}")

  # Create a custom command to generate the C++ source file
  add_custom_command(
    OUTPUT ${source_file_name}
    COMMAND ${CMAKE_COMMAND}
      -DRESOURCE_FILE=${resource_file_name}
      -DSOURCE_FILE=${source_file_name}
      -DVARIABLE_NAME=${variable_name}
      -P ${CMAKE_CURRENT_LIST_DIR}/../cmake/embed_resource.cmake
    DEPENDS ${resource_file_name}
    COMMENT "Embedding ${resource_file_name} into ${source_file_name}"
    VERBATIM
  )

  # Create a custom target to ensure the file is generated
  add_custom_target(${target_name} DEPENDS ${source_file_name})

  # Make the main target depend on this custom target
  add_dependencies(${PROJECT_NAME} ${target_name})
endfunction()

project_add_library(SHARED)

if(VISRTX_ENABLE_MDL_SUPPORT)
    embed_resource(
      ${CMAKE_CURRENT_BINARY_DIR}/../shaders/visrtx/default.mdl
      "${CMAKE_CURRENT_BINARY_DIR}/mdl/default.mdl.cpp"
      "VISRTX_DEFAULT_MDL"
    )
    embed_resource(
      ${CMAKE_CURRENT_BINARY_DIR}/../shaders/visrtx/physically_based.mdl
      "${CMAKE_CURRENT_BINARY_DIR}/mdl/physically_based.mdl.cpp"
      "VISRTX_PHYSICALLY_BASED_MDL"
    )
endif()

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

  scene/light/sampling/CDF.cu
  scene/light/Directional.cpp
  scene/light/HDRI.cpp
  scene/light/Light.cpp
  scene/light/Point.cpp
  scene/light/Rect.cpp
  scene/light/Ring.cpp
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
  OptiX::OptiX
  CUDA::cuda_driver
  CUDA::curand
  CUDA::nvml
  visrtx::glm
  visrtx::fmt
  visrtx::nanovdb
  nonstd::expected-lite
  nonstd::scope-lite
  nonstd::span-lite
  $<$<BOOL:${VISRTX_ENABLE_NVTX}>:CUDA::nvToolsExt>
  $<$<BOOL:${VISRTX_ENABLE_MDL_SUPPORT}>:libmdl>
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

#### Generate OptiX programs as embedded PTX ####

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
    OptiX::OptiX
    CUDA::curand
    visrtx::glm
    visrtx::nanovdb
    $<$<BOOL:${VISRTX_ENABLE_MDL_SUPPORT}>:libmdl>
  )
  target_include_directories(${INPUT_TARGET}
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_LIST_DIR}
  )

  if(VISRTX_ENABLE_NEURAL)
    set(CUDA_ARCHS 89-virtual)
  elseif(${CUDAToolkit_VERSION_MAJOR} GREATER 12)
    set(CUDA_ARCHS 75-virtual)
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


#### Query code generation ####

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

#### Install library + headers ####

include(GNUInstallDirs)

set(VISRTX_CMAKE_INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VisRTX)

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

###############################################################################
#### Configure CMake find_package() config files ##############################
###############################################################################

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/VisRTXConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/VisRTXConfig.cmake"
INSTALL_DESTINATION
  ${VISRTX_CMAKE_INSTALL_DESTINATION}
)

write_basic_package_version_file(
    "VisRTXConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/VisRTXConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/VisRTXConfigVersion.cmake
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
if (VISRTX_PRECOMPILE_SHADERS)
  install(CODE "message(STATUS \"Precompiling VisRTX OptiX shaders...\")")
  install(CODE
    "execute_process(COMMAND ${CMAKE_BINARY_DIR}/visrtxCompileShaders)"
  )
endif()
