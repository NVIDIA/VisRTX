## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_mpi)

project_add_library(INTERFACE)

find_package(MPI REQUIRED)

project_link_libraries(INTERFACE MPI::MPI_CXX)
project_include_directories(
INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
)
