## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_rendering)

project_add_library(STATIC)

project_sources(
PRIVATE
  index/RenderIndex.cpp
  index/RenderIndexFlatRegistry.cpp
  index/RenderIndexAllLayers.cpp
  pipeline/passes/AnariAxesRenderPass.cpp
  pipeline/passes/AnariSceneRenderPass.cpp
  pipeline/passes/ClearBuffersPass.cpp
  pipeline/passes/CopyFromColorBufferPass.cpp
  pipeline/passes/CopyToColorBufferPass.cpp
  pipeline/passes/CopyToSDLTexturePass.cpp
  pipeline/passes/MultiDeviceSceneRenderPass.cpp
  pipeline/passes/OutlineRenderPass.cpp
  pipeline/passes/PickPass.cpp
  pipeline/passes/RenderPass.cpp
  pipeline/passes/SaveToFilePass.cpp
  pipeline/passes/VisualizeAOVPass.cpp
  pipeline/RenderPipeline.cpp
  view/CameraUpdateDelegate.cpp
  view/ManipulatorToAnari.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
)

project_link_libraries(PUBLIC tsd_core PRIVATE $<BUILD_INTERFACE:stb_image>)

if (TSD_USE_CUDA)
  find_package(thrust REQUIRED HINTS ${CUDAToolkit_LIBRARY_DIR}/cmake)
  project_link_libraries(PRIVATE CUDA::cudart Thrust::Thrust)
  set_source_files_properties(
    pipeline/passes/AnariAxesRenderPass.cpp
    pipeline/passes/AnariSceneRenderPass.cpp
    pipeline/passes/OutlineRenderPass.cpp
    pipeline/passes/RenderPass.cpp
    pipeline/passes/VisualizeAOVPass.cpp
    PROPERTIES
      COMPILE_OPTIONS "--extended-lambda;--expt-relaxed-constexpr;-Xcudafe=--diag_suppress=20012;-Wno-deprecated-gpu-targets"
      LANGUAGE CUDA
  )
  project_compile_definitions(PRIVATE ENABLE_CUDA)
elseif(TSD_USE_TBB)
  find_package(TBB REQUIRED)
  project_link_libraries(PRIVATE TBB::tbb)
  project_compile_definitions(PRIVATE ENABLE_TBB)
endif()

if (TSD_USE_SDL3)
  if (NOT TARGET SDL3::SDL3)
    find_package(SDL3 REQUIRED)
  endif()
  project_link_libraries(PUBLIC SDL3::SDL3)
  project_compile_definitions(PUBLIC ENABLE_SDL)
endif()
