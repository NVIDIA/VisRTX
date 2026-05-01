// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"
#include "tsd/ui/imgui/windows/Window.h"

namespace tsd::scivis_studio {

struct CameraRigEditor : public tsd::ui::imgui::Window
{
  CameraRigEditor(
      tsd::ui::imgui::Application *app, ProjectContext *projectContext);
  ~CameraRigEditor() override;

  void buildUI() override;

 private:
  ProjectContext *m_projectContext{nullptr};
  int m_selectedKeyframe{-1};
};

} // namespace tsd::scivis_studio
