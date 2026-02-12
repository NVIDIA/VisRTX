## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_lua)

project_add_library(STATIC)

project_sources(
PRIVATE
  LuaContext.cpp
  LuaBindings.cpp
  bindings/CoreBindings.cpp
  bindings/MathBindings.cpp
  bindings/ObjectBindings.cpp
  bindings/ParameterHelpers.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
)

project_link_libraries(
PUBLIC
  tsd_core
  tsd_io
  tsd_rendering
PRIVATE
  $<BUILD_INTERFACE:tsd_ext_sol2>
  $<BUILD_INTERFACE:stb_image>
)

project_compile_definitions(
PUBLIC
  TSD_USE_LUA=1
PRIVATE
  TSD_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(UNIX)
  target_link_options(tsd_lua PUBLIC -rdynamic)
endif()
