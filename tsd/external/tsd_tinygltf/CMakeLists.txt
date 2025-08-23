## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_ext_tinygltf)

project_add_library(STATIC)
project_include_directories(PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
project_include_directories(PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src)
project_link_libraries(PRIVATE stb_image)
project_compile_definitions(PUBLIC TINYGLTF_NOEXCEPTION=1)
project_sources(PRIVATE src/tiny_gltf.cc)
