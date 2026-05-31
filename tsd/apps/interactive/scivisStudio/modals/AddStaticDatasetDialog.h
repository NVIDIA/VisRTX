// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"
#include "tsd/ui/imgui/modals/Modal.h"

#include <array>
#include <string>

namespace tsd::scivis_studio {

struct AddStaticDatasetDialog : public tsd::ui::imgui::Modal
{
  AddStaticDatasetDialog(
      tsd::ui::imgui::Application *app, ProjectContext *projectContext);
  ~AddStaticDatasetDialog() override;

 private:
  void buildUI() override;

  ProjectContext *m_projectContext{nullptr};
  std::array<char, 512> m_name{};
  std::array<char, 2048> m_sourcePath{};
  std::string m_browsedSourcePath;
  int m_selectedImporter{0};
};

} // namespace tsd::scivis_studio
