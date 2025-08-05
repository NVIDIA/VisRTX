## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

add_subdirectory(demos)
add_subdirectory(viewer)
option(BUILD_PTC_VIEWER "Build MPI viewer" OFF)
if (BUILD_PTC_VIEWER)
  add_subdirectory(ptc_viewer)
endif()
