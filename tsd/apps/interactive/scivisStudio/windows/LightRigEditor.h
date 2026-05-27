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
  void buildUI_lightList(LightRig &rig);
  void buildUI_addLight(LightRig &rig);

  ProjectContext *m_projectContext{nullptr};
  int m_selectedRig{0};
  int m_selectedLight{0};
  std::string m_renameLightName;
  LightRigID m_pendingDeleteRig;
};

} // namespace tsd::scivis_studio
