## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

add_subdirectory(tsd_agx)
add_subdirectory(tsd_catch)
add_subdirectory(tsd_flip)
add_subdirectory(tsd_imnodes)
add_subdirectory(tsd_imoguizmo)
add_subdirectory(tsd_mikktspace)
add_subdirectory(tsd_nanovdb)
add_subdirectory(tsd_tinyexr)
add_subdirectory(tsd_tinygltf)
add_subdirectory(tsd_tinyobjloader)
add_subdirectory(tsd_tinyply)

add_subdirectory(
  ${CMAKE_CURRENT_LIST_DIR}/../../devices/rtx/external
  ${CMAKE_CURRENT_BINARY_DIR}/tsd_visrtx_external
)

if(TSD_BUILD_UI_LIBRARY)
  add_subdirectory(tsd_imguizmo)
endif()

if(TSD_USE_LUA)
  add_subdirectory(tsd_lua)
  add_subdirectory(tsd_sol2 EXCLUDE_FROM_ALL)
endif()
