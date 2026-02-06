## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdDemoViskores)
project_add_executable(tsdDemoViskores.cpp NodeEditor.cpp NodeInfoWindow.cpp)
find_package(viskores_graph 1.1.0 REQUIRED)
project_link_libraries(
  tsd_ui_imgui
  tsd_ext_imnodes
  viskores_graph::viskores_graph
)
project_compile_definitions(PRIVATE -DIMGUI_DEFINE_MATH_OPERATORS)
