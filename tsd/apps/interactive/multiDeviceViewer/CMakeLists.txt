## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdMultiDeviceViewer)
project_add_executable(multiDeviceViewer.cpp)
project_link_libraries(PUBLIC tsd_ui_imgui)
