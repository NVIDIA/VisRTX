## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdMPIViewer)
project_add_executable(
  DistributedSceneController.cpp
  DistributedViewport.cpp
  mpiViewer.cpp
)
project_link_libraries(PUBLIC tsd_ui_imgui tsd_mpi)
