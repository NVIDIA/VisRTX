## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsdTestClient)
project_add_executable(tsdTestClient.cpp)
project_link_libraries(tsd_network stb_image)
