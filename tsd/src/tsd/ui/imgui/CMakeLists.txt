## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_ui_imgui)

project_add_library(OBJECT)

project_sources(PRIVATE
  modals/AppSettingsDialog.cpp
  modals/BlockingTaskModal.cpp
  modals/ImportFileDialog.cpp
  windows/CameraPoses.cpp
  windows/DatabaseEditor.cpp
  windows/IsosurfaceEditor.cpp
  windows/LayerTree.cpp
  windows/Log.cpp
  windows/ObjectEditor.cpp
  windows/TransferFunctionEditor.cpp
  windows/Viewport.cpp
  windows/Window.cpp
  tsd_ui.cpp
)

project_link_libraries(
PUBLIC
  tsd_core
  tsd_rendering
  tsd_io
  tsd_ext_tinyexr
  stb_image
PRIVATE
  anari::anari_viewer
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
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
