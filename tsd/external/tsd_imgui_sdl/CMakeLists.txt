## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_ext_imgui_sdl LANGUAGES CXX)

anari_sdk_fetch_project(
  NAME ${PROJECT_NAME}
  URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.7-docking.zip
  MD5 2cfbf7b7790076d6debe5060ec1fb47f
)

find_package(SDL3 QUIET)
if(NOT TARGET SDL3::SDL3)
  mark_as_advanced(SDL3_DIR)
  anari_sdk_fetch_project(
    NAME tsd_ext_sdl3
    URL https://github.com/libsdl-org/SDL/archive/refs/tags/release-3.2.12.zip
    MD5 62da6ae22e546cdacdb6dfef520391bf
  )
  add_subdirectory(
    ${tsd_ext_sdl3_LOCATION}
    ${CMAKE_CURRENT_BINARY_DIR}/tsd_ext_sdl3
    EXCLUDE_FROM_ALL
  )
else()
  mark_as_advanced(CLEAR SDL3_DIR)
endif()

project_add_library(STATIC)

project_sources(
PRIVATE
  ${tsd_ext_imgui_sdl_LOCATION}/imgui.cpp
  ${tsd_ext_imgui_sdl_LOCATION}/imgui_draw.cpp
  ${tsd_ext_imgui_sdl_LOCATION}/imgui_demo.cpp
  ${tsd_ext_imgui_sdl_LOCATION}/imgui_tables.cpp
  ${tsd_ext_imgui_sdl_LOCATION}/imgui_widgets.cpp
  ${tsd_ext_imgui_sdl_LOCATION}/backends/imgui_impl_sdl3.cpp
  ${tsd_ext_imgui_sdl_LOCATION}/backends/imgui_impl_sdlrenderer3.cpp
  ${tsd_ext_imgui_sdl_LOCATION}/misc/cpp/imgui_stdlib.cpp
)

project_link_libraries(PUBLIC SDL3::SDL3)

project_include_directories(
PUBLIC
  ${tsd_ext_imgui_sdl_LOCATION}
  ${tsd_ext_imgui_sdl_LOCATION}/backends
  ${tsd_ext_imgui_sdl_LOCATION}/misc/cpp
)
