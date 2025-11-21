## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

add_subdirectory(tsd_agx)
add_subdirectory(tsd_catch)
add_subdirectory(tsd_flip)
add_subdirectory(tsd_imguizmo)
add_subdirectory(tsd_imnodes)
add_subdirectory(tsd_mikktspace)
add_subdirectory(tsd_nanovdb)
add_subdirectory(tsd_tinyexr)
add_subdirectory(tsd_tinygltf)
add_subdirectory(tsd_tinyobjloader)
add_subdirectory(tsd_tinyply)

if (NOT TARGET fmt::fmt)
  add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/../../devices/rtx/external/fmtlib
    ${CMAKE_CURRENT_BINARY_DIR}/tsd_fmt
  )
  add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/../../devices/rtx/external/stb_image
    ${CMAKE_CURRENT_BINARY_DIR}/tsd_stb
  )
endif()
