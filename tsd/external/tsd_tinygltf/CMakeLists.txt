## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_ext_tinygltf)

project_add_library(STATIC)
project_include_directories(PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
project_include_directories(PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src)

# Link to stb_image and set up include paths
if (TARGET stb_image)
  project_link_libraries(PRIVATE stb_image)
  # Get the include directory from stb_image target
  get_target_property(STB_IMAGE_INCLUDE_DIR stb_image INTERFACE_INCLUDE_DIRECTORIES)
  if (STB_IMAGE_INCLUDE_DIR)
    project_include_directories(PRIVATE ${STB_IMAGE_INCLUDE_DIR})
  endif()
endif()

project_compile_definitions(PUBLIC TINYGLTF_NOEXCEPTION=1)
project_sources(PRIVATE src/tiny_gltf.cc)
