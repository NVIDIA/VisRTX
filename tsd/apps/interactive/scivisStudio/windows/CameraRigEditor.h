// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"
#include "tsd/ui/imgui/windows/Window.h"

#include <string>

namespace tsd::scivis_studio {

struct CameraRigEditor : public tsd::ui::imgui::Window
{
  CameraRigEditor(
      tsd::ui::imgui::Application *app, ProjectContext *projectContext);
  ~CameraRigEditor() override;

  void buildUI() override;

 private:
  bool inputText(const char *label, std::string &value, size_t capacity = 512);
  void syncSelectionToActiveShot();
  void pollPendingFileIO();
  void buildUI_rigControls();
  void buildUI_ioError();
  bool buildUI_poseEditor(const char *label, ManipulatorState &state);
  void buildUI_keyframes(CameraRig &cameraRig);

  // The SDL file dialog is asynchronous: it writes the chosen path into the
  // target string from a later event-loop iteration, so the target must outlive
  // the call. We hand it a member and poll for the result on subsequent frames.
  enum class PendingFileIO
  {
    None,
    Import,
    Export
  };

  ProjectContext *m_projectContext{nullptr};
  int m_selectedRig{0};
  int m_selectedKeyframe{-1};
  ShotID m_lastActiveShotId;
  CameraRigID m_pendingDeleteRig;
  std::string m_ioError;
  PendingFileIO m_pendingFileIO{PendingFileIO::None};
  std::string m_pendingFilename;
  CameraRigID m_pendingExportRig;
};

} // namespace tsd::scivis_studio
