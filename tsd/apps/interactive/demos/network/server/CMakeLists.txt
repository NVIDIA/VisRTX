## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdServer)
project_add_executable(
  CopyFromColorBufferPass.cpp
  RenderServer.cpp
  tsdServer.cpp
)
project_link_libraries(tsd_app tsd_network)
