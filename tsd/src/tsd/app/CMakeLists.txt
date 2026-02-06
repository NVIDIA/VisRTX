## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_app)

project_add_library(STATIC)
project_sources(PRIVATE Core.cpp renderAnimationSequence.cpp)
project_link_libraries(PUBLIC tsd_core tsd_io tsd_rendering)
project_include_directories(PUBLIC ${CMAKE_CURRENT_LIST_DIR})
