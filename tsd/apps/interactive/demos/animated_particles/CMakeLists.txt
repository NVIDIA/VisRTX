## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

if (NOT TSD_USE_CUDA)
  return()
endif()

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/../external/glm)
find_package(glm REQUIRED)

project(tsdDemoAnimatedParticles)
project_add_executable(
  particle_system.cu
  SimulationControls.cpp
  tsdDemoAnimatedParticles.cpp
)
project_link_libraries(tsd_ui_imgui glm::glm)
set_source_files_properties(
  particle_system.cu
  PROPERTIES
    COMPILE_OPTIONS "--extended-lambda;--expt-relaxed-constexpr;-Xcudafe=--diag_suppress=20012;-Wno-deprecated-gpu-targets"
    LANGUAGE CUDA
)
