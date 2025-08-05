## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_ext_tinyexr)
project_add_library(INTERFACE)
project_include_directories(INTERFACE ${CMAKE_CURRENT_LIST_DIR})

project(tsd_ext_tinyexr_impl)
project_add_library(INTERFACE)
project_sources(INTERFACE tinyexr.cpp)
