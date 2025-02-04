## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdViewer)
project_add_executable(
  windows/DatabaseEditor.cpp
  tsdViewer.cpp
)
project_link_libraries(PUBLIC tsd_viewer_common)
