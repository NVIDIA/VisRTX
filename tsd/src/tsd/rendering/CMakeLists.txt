## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_rendering)

project_add_library(STATIC)

project_sources(
PRIVATE
  render_index/RenderIndex.cpp
  render_index/RenderIndexFlatRegistry.cpp
  render_index/RenderIndexAllLayers.cpp
  render_pipeline/passes/AnariAxesRenderPass.cpp
  render_pipeline/passes/AnariSceneRenderPass.cpp
  render_pipeline/passes/CopyToSDLTexturePass.cpp
  render_pipeline/passes/OutlineRenderPass.cpp
  render_pipeline/passes/PickPass.cpp
  render_pipeline/passes/RenderPass.cpp
  render_pipeline/passes/VisualizeDepthPass.cpp
  render_pipeline/RenderPipeline.cpp
  view/ManipulatorToAnari.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
)

project_link_libraries(PUBLIC tsd_core)

if (TSD_USE_CUDA)
  project_link_libraries(PRIVATE CUDA::cudart)
  set_source_files_properties(
    render_pipeline/passes/AnariAxesRenderPass.cpp
    render_pipeline/passes/AnariSceneRenderPass.cpp
    render_pipeline/passes/OutlineRenderPass.cpp
    render_pipeline/passes/RenderPass.cpp
    render_pipeline/passes/VisualizeDepthPass.cpp
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
  project_link_libraries(PUBLIC tsd SDL3::SDL3)
  project_compile_definitions(PUBLIC ENABLE_SDL)
endif()
