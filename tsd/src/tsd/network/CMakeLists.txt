## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_network)

project_add_library(STATIC)

project_sources(PRIVATE NetworkChannel.cpp)

# Find Boost (Asio)
find_package(Boost 1.66 REQUIRED COMPONENTS system)

project_link_libraries(PUBLIC Boost::system tsd_core)
project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
)
