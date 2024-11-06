## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdArrayInstancingDemo)
project_add_executable(InstancingControls.cpp tsdArrayInstancingDemo.cpp)
project_link_libraries(tsd_viewer_common)
