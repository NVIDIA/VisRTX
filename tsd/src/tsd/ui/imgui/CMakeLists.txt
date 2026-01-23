## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_ui_imgui)

find_package(anari REQUIRED COMPONENTS viewer)

project_add_library(OBJECT)

project_sources(PRIVATE
  modals/AppSettingsDialog.cpp
  modals/BlockingTaskModal.cpp
  modals/ExportNanoVDBFileDialog.cpp
  modals/OfflineRenderModal.cpp
  modals/ImportFileDialog.cpp
  modals/Modal.cpp
  windows/Animations.cpp
  windows/CameraPoses.cpp
  windows/DatabaseEditor.cpp
  windows/IsosurfaceEditor.cpp
  windows/LayerTree.cpp
  windows/Log.cpp
  windows/MultiDeviceViewport.cpp
  windows/ObjectEditor.cpp
  windows/TransferFunctionEditor.cpp
  windows/Viewport.cpp
  windows/Window.cpp
  Application.cpp
  tsd_ui_imgui.cpp
)

if(TSD_USE_LUA)
  project_sources(PRIVATE
    windows/Terminal.cpp
    ExtensionManager.cpp
  )
  project_link_libraries(PRIVATE tsd_lua tsd_ext_sol2)
  project_compile_definitions(PRIVATE TSD_USE_LUA=1)
endif()

project_link_libraries(
PUBLIC
  tsd_app
  tsd_core
  tsd_io
  tsd_rendering
  tsd_ext_imguizmo
  tsd_ext_tinyexr
  stb_image
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
  if (LINK_TARGET_INCLUDE_DIRS)
    list(APPEND ANARI_VIEWER_INCLUDE_DIRS ${LINK_TARGET_INCLUDE_DIRS})
  endif()
endforeach()

project_include_directories(
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  INTERFACE
    ${ANARI_VIEWER_INCLUDE_DIRS}
)
