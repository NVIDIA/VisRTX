// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"
#include "tsd/ui/imgui/windows/Window.h"

namespace tsd::scivis_studio {

struct DatasetEditor : public tsd::ui::imgui::Window
{
  DatasetEditor(
      tsd::ui::imgui::Application *app, ProjectContext *projectContext);
  ~DatasetEditor() override;

  void buildUI() override;

 private:
  ProjectContext *m_projectContext{nullptr};
  int m_selectedDataset{0};
};

} // namespace tsd::scivis_studio
