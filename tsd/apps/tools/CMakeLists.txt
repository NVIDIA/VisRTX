## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(conduit2tsd)
project_add_executable(conduit2tsd.cpp)
project_link_libraries(tsd)

project(obj2header)
project_add_executable(obj2header.cpp)
project_link_libraries(tsd tsd_tiny_obj_loader)

project(printTSD)
project_add_executable(printTSD.cpp)
project_link_libraries(tsd)

project(tsdRender)
project_add_executable(tsdRender.cpp)
project_link_libraries(tsd tsd_stb render_pipeline)
