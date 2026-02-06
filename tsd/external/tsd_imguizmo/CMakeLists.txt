## Copyright 2025-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_ext_imguizmo LANGUAGES CXX)

anari_sdk_fetch_project(
  NAME ${PROJECT_NAME}
  URL https://github.com/CedricGuillemet/ImGuizmo/archive/71f14292205c3317122b39627ed98efce137086a.zip
  MD5 16e691b39d4baec349d769768e77d0e9
)

project_add_library(STATIC)

project_sources(
PRIVATE
  ${tsd_ext_imguizmo_LOCATION}/ImGuizmo.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${tsd_ext_imguizmo_LOCATION}>
)

project_link_libraries(
PRIVATE
  anari::helium
  anari::anari_viewer
)