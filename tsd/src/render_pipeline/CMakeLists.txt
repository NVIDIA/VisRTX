## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(render_pipeline LANGUAGES CXX)

find_package(OpenGL REQUIRED)

project_add_library(STATIC)
project_sources(PRIVATE RenderPass.cpp RenderPipeline.cpp)
project_include_directories(PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)
project_link_libraries(PUBLIC tsd OpenGL::GL)

if (APPLE)
  set(TSD_RENDER_PIPELINE_USE_CUDA OFF)
else()
  option(
    TSD_RENDER_PIPELINE_USE_CUDA
    "Use CUDA for the render pipeline library"
    ON
  )
endif()

if (TSD_RENDER_PIPELINE_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  project_link_libraries(PRIVATE CUDA::cudart)
  set_source_files_properties(
    RenderPass.cpp
    PROPERTIES
      COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr"
      LANGUAGE CUDA
  )
  project_compile_definitions(PRIVATE ENABLE_CUDA)
else()
  find_package(TBB REQUIRED)
  project_link_libraries(PRIVATE TBB::tbb)
endif()
