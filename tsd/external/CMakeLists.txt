## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

add_subdirectory(tsd_catch)
add_subdirectory(tsd_flip)
add_subdirectory(tsd_mikktspace)
add_subdirectory(tsd_nanovdb)
add_subdirectory(tsd_tiny_obj_loader)
add_subdirectory(tsd_tinyply)
add_subdirectory(tsd_tinyexr)
add_subdirectory(tsd_tinygltf)

if (TSD_STANDALONE_BUILD)
  add_subdirectory(../../external/fmtlib ${CMAKE_CURRENT_BINARY_DIR}/tsd_fmt)
  add_subdirectory(../../external/stb_image ${CMAKE_CURRENT_BINARY_DIR}/tsd_stb)
endif()
