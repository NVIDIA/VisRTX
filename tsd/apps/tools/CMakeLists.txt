## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(obj2header)
project_add_executable(obj2header.cpp)
project_link_libraries(tsd tsd_tiny_obj_loader)
