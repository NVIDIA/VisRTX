// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"
#include "tsd/ui/imgui/windows/Window.h"

#include <string>

namespace tsd::scivis_studio {

struct LightRigEditor : public tsd::ui::imgui::Window
{
  LightRigEditor(
      tsd::ui::imgui::Application *app, ProjectContext *projectContext);
  ~LightRigEditor() override;

  void buildUI() override;

 private:
  bool inputText(const char *label, std::string &value, size_t capacity = 512);
  // Buffered, reject-on-commit rig-name field: edits a scratch buffer and only
  // applies a valid, unique name on commit, surfacing an inline error otherwise.
  void buildUI_nameField(LightRig &rig);
  void syncSelectionToActiveShot();
  void pollPendingFileIO();
  void buildUI_lightList(LightRig &rig);
  void buildUI_addLight(LightRig &rig);
  void buildUI_ioError();

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
  int m_selectedLight{-1};
  ShotID m_lastActiveShotId;
  LightRigID m_lastActiveShotLightRigId;
  std::string m_renameLightName;
  LightRigID m_pendingDeleteRig;
  std::string m_ioError;
  // Rig-name-field edit state (keyed to the rig whose name is being edited).
  LightRigID m_nameBufferRig;
  std::string m_nameBuffer;
  std::string m_nameError;
  PendingFileIO m_pendingFileIO{PendingFileIO::None};
  std::string m_pendingFilename;
  LightRigID m_pendingExportRig;
};

} // namespace tsd::scivis_studio
