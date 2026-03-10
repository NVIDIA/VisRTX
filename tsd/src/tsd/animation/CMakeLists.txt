## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_animation)

project_add_library(STATIC)

project_sources(
PRIVATE
  Apply.cpp
  Evaluate.cpp
  SceneAnimation.cpp
  TimeSamples.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
)

project_link_libraries(PUBLIC tsd_scene)
