## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdViewer)
project_add_executable(tsdViewer.cpp)
project_link_libraries(PUBLIC tsd_ui_imgui)
if(TSD_USE_LUA)
  project_compile_definitions(PRIVATE TSD_USE_LUA=1)
endif()