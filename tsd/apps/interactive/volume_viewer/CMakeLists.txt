## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdVolumeViewer)
project_add_executable(tsdVolumeViewer.cpp)
project_link_libraries(PUBLIC tsd_viewer_common)
