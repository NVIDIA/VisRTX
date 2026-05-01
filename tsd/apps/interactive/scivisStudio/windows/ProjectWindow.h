// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"
#include "tsd/ui/imgui/windows/Window.h"

namespace tsd::scivis_studio {

struct ProjectWindow : public tsd::ui::imgui::Window
{
  ProjectWindow(
      tsd::ui::imgui::Application *app, ProjectContext *projectContext);
  ~ProjectWindow() override;

  void buildUI() override;

 private:
  ProjectContext *m_projectContext{nullptr};
};

} // namespace tsd::scivis_studio
