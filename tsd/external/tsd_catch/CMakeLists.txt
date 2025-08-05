## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

if (BUILD_TESTING)
  project(tsd_ext_catch2)
  project_add_library(STATIC catch_main.cpp)
  project_include_directories(PUBLIC ${CMAKE_CURRENT_LIST_DIR}/catch2)
endif()