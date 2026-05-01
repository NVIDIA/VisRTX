## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

add_library(tsd_scivis_studio_model STATIC
  Dataset.cpp
  Project.cpp
  ProjectContext.cpp
  ProjectSerialization.cpp
  Shot.cpp
  ShotCameraRig.cpp
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

add_executable(scivisStudio
  Application.cpp
  RenderShot.cpp
  modals/AddDatasetDialog.cpp
  modals/ConfirmDiscardDialog.cpp
  modals/ProjectLocationDialog.cpp
  scivisStudio.cpp
  windows/CameraRigEditor.cpp
  windows/DatasetEditor.cpp
  windows/ProjectWindow.cpp
  windows/ShotEditor.cpp
)

target_link_libraries(scivisStudio
PRIVATE
  tsd_scivis_studio_model
  tsd_ui_imgui
)
