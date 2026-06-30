## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

add_library(tsd_scivis_studio_model STATIC
  CameraRig.cpp
  Dataset.cpp
  DatasetIO.cpp
  LightRig.cpp
  Project.cpp
  ProjectAssetTransaction.cpp
  ProjectContext.cpp
  ProjectSerialization.cpp
  RenderShot.cpp
  RenderShotCLI.cpp
  Shot.cpp
)

target_include_directories(tsd_scivis_studio_model
PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(tsd_scivis_studio_model
PUBLIC
  tsd_app
  tsd_core
  tsd_io
  tsd_rendering
  tsd_scene
)

set(SCIVIS_STUDIO_DEFAULT_LAYOUT_FILE
  "${CMAKE_CURRENT_LIST_DIR}/default_ui_layout.txt")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${SCIVIS_STUDIO_DEFAULT_LAYOUT_FILE}")

file(READ
  "${SCIVIS_STUDIO_DEFAULT_LAYOUT_FILE}"
  SCIVIS_STUDIO_DEFAULT_LAYOUT)

configure_file(
  DefaultLayout.h.in
  "${CMAKE_CURRENT_BINARY_DIR}/DefaultLayout.h"
  @ONLY)

add_executable(scivisStudio
  Application.cpp
  modals/AddFileAnimationDatasetDialog.cpp
  modals/AddStaticDatasetDialog.cpp
  modals/ProjectLocationDialog.cpp
  scivisStudio.cpp
  windows/CameraRigEditor.cpp
  windows/DatasetEditor.cpp
  windows/LightRigEditor.cpp
  windows/ProjectWindow.cpp
  windows/ShotEditor.cpp
)

target_include_directories(scivisStudio
PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
)

target_link_libraries(scivisStudio
PRIVATE
  tsd_scivis_studio_model
  tsd_ui_imgui
)

add_executable(scivisStudioRenderShot
  scivisStudioRenderShot.cpp
)

target_link_libraries(scivisStudioRenderShot
PRIVATE
  tsd_scivis_studio_model
)
