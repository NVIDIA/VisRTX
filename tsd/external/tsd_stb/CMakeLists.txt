## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_stb LANGUAGES C)
project_add_library(STATIC stb_image_write.c stb_image.c)
project_include_directories(INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)
