## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_viewer_common)
project_add_library(OBJECT)
project_sources(PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/modals/AppSettings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/windows/IsosurfaceEditor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/windows/Log.cpp
  ${CMAKE_CURRENT_LIST_DIR}/windows/ObjectEditor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/windows/ObjectTree.cpp
  ${CMAKE_CURRENT_LIST_DIR}/windows/TransferFunctionEditor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/windows/Viewport.cpp
  ${CMAKE_CURRENT_LIST_DIR}/AppCore.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Manipulator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tsd_ui.cpp
)
project_link_libraries(
PUBLIC
  tsd
  tsd_stb
  tsd_tinyexr
  render_pipeline
PRIVATE
  anari::anari_viewer
)

##
## Gather anari::anari_viewer include paths for apps to have when needed
##

# Direct include dirs
get_target_property(
  ANARI_VIEWER_INCLUDE_DIRS
  anari::anari_viewer
  INTERFACE_INCLUDE_DIRECTORIES
)

# Get list of incoming dependent targets to anari::anari_viewer
get_target_property(
  ANARI_VIEWER_LINK_LIBRARIES
  anari::anari_viewer
  INTERFACE_LINK_LIBRARIES
)

# Append the list of include paths from each dependent target
foreach(LINK_TARGET ${ANARI_VIEWER_LINK_LIBRARIES})
  get_target_property(
    LINK_TARGET_INCLUDE_DIRS
    ${LINK_TARGET}
    INTERFACE_INCLUDE_DIRECTORIES
  )
  list(APPEND ANARI_VIEWER_INCLUDE_DIRS ${LINK_TARGET_INCLUDE_DIRS})
endforeach()

project_include_directories(
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  INTERFACE
    ${ANARI_VIEWER_INCLUDE_DIRS}
)

