## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_ui_imgui)

project_add_library(OBJECT)

project_sources(PRIVATE
  modals/AppSettingsDialog.cpp
  modals/BlockingTaskModal.cpp
  modals/ExportNanoVDBFileDialog.cpp
  modals/OfflineRenderModal.cpp
  modals/ImportFileDialog.cpp
  modals/Modal.cpp
  modals/CuttingPlaneDialog.cpp
  modals/VorticityDialog.cpp
  windows/Animations.cpp
  windows/Timeline.cpp
  windows/BaseViewport.cpp
  windows/CameraPoses.cpp
  windows/DatabaseEditor.cpp
  windows/IsosurfaceEditor.cpp
  windows/LayerTree.cpp
  windows/Log.cpp
  windows/MultiDeviceViewport.cpp
  windows/ObjectEditor.cpp
  windows/Terminal.cpp
  windows/TransferFunctionEditor.cpp
  windows/Viewport.cpp
  windows/Window.cpp
  Application.cpp
  ArrayPreview.cpp
  ExtensionManager.cpp
  tsd_ui_imgui.cpp
)

if(TSD_USE_LUA)
  project_link_libraries(PRIVATE tsd_lua tsd_ext_sol2)
  project_compile_definitions(PRIVATE TSD_USE_LUA)
endif()

project_link_libraries(
PUBLIC
  tsd_app
  tsd_core
  tsd_io
  tsd_rendering
  tsd_ext_imguizmo
  tsd_ext_imoguizmo
  tsd_ext_imgui_sdl
  tsd_ext_tinyexr
  stb_image
)

project_include_directories(
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)
