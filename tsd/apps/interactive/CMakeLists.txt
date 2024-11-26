## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

add_subdirectory(common)

add_subdirectory(array_instancing_demo)
add_subdirectory(demos)
add_subdirectory(material_explorer)
add_subdirectory(viewer)
add_subdirectory(volume_viewer)

option(BUILD_PTC_VIEWER "Build MPI viewer" OFF)
if (BUILD_PTC_VIEWER)
  add_subdirectory(ptc_viewer)
endif()
