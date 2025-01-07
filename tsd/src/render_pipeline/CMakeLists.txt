## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(render_pipeline LANGUAGES CXX)

project_add_library(STATIC)
project_sources(PRIVATE RenderPass.cpp RenderPipeline.cpp)
project_include_directories(PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)
project_link_libraries(PUBLIC tsd)

if (TSD_USE_CUDA)
  project_link_libraries(PRIVATE CUDA::cudart)
  set_source_files_properties(
    RenderPass.cpp
    PROPERTIES
      COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr"
      LANGUAGE CUDA
  )
  project_compile_definitions(PRIVATE ENABLE_CUDA)
elseif(TSD_USE_TBB)
  find_package(TBB REQUIRED)
  project_link_libraries(PRIVATE TBB::tbb)
  project_compile_definitions(PRIVATE ENABLE_TBB)
endif()

if (TSD_USE_OPENGL)
  find_package(OpenGL REQUIRED)
  project_link_libraries(PUBLIC tsd OpenGL::GL)
  project_compile_definitions(PUBLIC ENABLE_OPENGL)
endif()

