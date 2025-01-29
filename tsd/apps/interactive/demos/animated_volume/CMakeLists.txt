## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

if (NOT TSD_USE_CUDA)
  return()
endif()

project(tsdDemoAnimatedVolume)
project_add_executable(jacobi3D.cu SolverControls.cpp tsdDemoAnimatedVolume.cpp)
project_link_libraries(PRIVATE tsd_viewer_common)
