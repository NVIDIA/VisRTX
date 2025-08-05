## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_ext_mikktspace LANGUAGES C)
project_add_library(STATIC mikktspace.c)
project_include_directories(PUBLIC ${CMAKE_CURRENT_LIST_DIR})
