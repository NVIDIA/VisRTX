## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

find_package(MPI REQUIRED)

project(ptcViewer)
project_add_executable(DistributedViewport.cpp ptcViewer.cpp)
project_link_libraries(PRIVATE tsd_ui_imgui MPI::MPI_CXX)
project_include_directories(PRIVATE external)
