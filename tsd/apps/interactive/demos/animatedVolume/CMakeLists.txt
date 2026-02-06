## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

if (NOT TSD_USE_CUDA)
  return()
endif()

project(tsdDemoAnimatedVolume)
project_add_executable(jacobi3D.cu SolverControls.cpp tsdDemoAnimatedVolume.cpp)
project_link_libraries(tsd_ui_imgui)
