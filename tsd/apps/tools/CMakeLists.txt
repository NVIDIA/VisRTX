## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(obj2header)
project_add_executable(obj2header.cpp)
project_link_libraries(tsd_core tsd_io tsd_ext_tinyobjloader)

project(tsdPrint)
project_add_executable(tsdPrint.cpp)
project_link_libraries(tsd_core)

project(tsdRender)
project_add_executable(tsdRender.cpp)
project_link_libraries(
  tsd_app
  tsd_core
  tsd_io
  tsd_rendering
  tsd_ext_tinyexr_impl
  stb_image
)

project(tsdVolumeToNanoVDB)
project_add_executable(tsdVolumeToNanoVDB.cpp)
project_link_libraries(tsd_core tsd_io)
