## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdDemoArrayInstancing)
project_add_executable(InstancingControls.cpp tsdDemoArrayInstancing.cpp)
project_link_libraries(tsd_ui_imgui)
