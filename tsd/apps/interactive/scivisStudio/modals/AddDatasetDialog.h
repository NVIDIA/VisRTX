// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"
#include "tsd/ui/imgui/modals/Modal.h"

#include <array>

namespace tsd::scivis_studio {

struct AddDatasetDialog : public tsd::ui::imgui::Modal
{
  AddDatasetDialog(
      tsd::ui::imgui::Application *app, ProjectContext *projectContext);
  ~AddDatasetDialog() override;

 private:
  void buildUI() override;

  ProjectContext *m_projectContext{nullptr};
  std::array<char, 512> m_name{};
  std::array<char, 2048> m_sourcePath{};
  int m_selectedImporter{0};
};

} // namespace tsd::scivis_studio
