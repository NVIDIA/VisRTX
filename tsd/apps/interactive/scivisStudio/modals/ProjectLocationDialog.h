// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/ui/imgui/modals/Modal.h"

#include <array>
#include <filesystem>
#include <functional>
#include <string>

namespace tsd::scivis_studio {

enum class ProjectLocationMode
{
  OpenProject,
  SaveProjectAs
};

struct ProjectLocationDialog : public tsd::ui::imgui::Modal
{
  explicit ProjectLocationDialog(tsd::ui::imgui::Application *app);
  ~ProjectLocationDialog() override;

  void configure(ProjectLocationMode mode,
      std::function<void(const std::filesystem::path &)> onAccept);

 private:
  void buildUI() override;
  bool validate(const std::filesystem::path &path, std::string &error) const;

  ProjectLocationMode m_mode{ProjectLocationMode::OpenProject};
  std::function<void(const std::filesystem::path &)> m_onAccept;
  std::array<char, 2048> m_directory{};
  std::string m_browsedDirectory;
  std::string m_error;
};

} // namespace tsd::scivis_studio
