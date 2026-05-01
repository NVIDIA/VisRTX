// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ProjectLocationDialog.h"

#include "ProjectSerialization.h"

#include "imgui.h"

namespace tsd::scivis_studio {

ProjectLocationDialog::ProjectLocationDialog(tsd::ui::imgui::Application *app)
    : Modal(app, "Project Location")
{}

ProjectLocationDialog::~ProjectLocationDialog() = default;

void ProjectLocationDialog::configure(ProjectLocationMode mode,
    std::function<void(const std::filesystem::path &)> onAccept)
{
  m_mode = mode;
  m_onAccept = std::move(onAccept);
  m_error.clear();
}

bool ProjectLocationDialog::validate(
    const std::filesystem::path &path, std::string &error) const
{
  if (path.empty()) {
    error = "Enter a project directory.";
    return false;
  }

  const auto manifest = path / PROJECT_MANIFEST_FILENAME;
  if (m_mode == ProjectLocationMode::OpenProject) {
    auto result = validateProjectRoot(path);
    if (!result.ok)
      error = result.error;
    return result.ok;
  }

  if (std::filesystem::exists(manifest)) {
    error = "Target directory already contains project.tsd.";
    return false;
  }

  if (std::filesystem::exists(path) && !std::filesystem::is_directory(path)) {
    error = "Target path is not a directory.";
    return false;
  }

  return true;
}

void ProjectLocationDialog::buildUI()
{
  const char *title = "Open Project";
  const char *button = "Open";
  if (m_mode == ProjectLocationMode::NewProject) {
    title = "New Project";
    button = "Create";
  } else if (m_mode == ProjectLocationMode::SaveProjectAs) {
    title = "Save Project As";
    button = "Save";
  }

  ImGui::TextUnformatted(title);
  ImGui::InputText("Directory", m_directory.data(), m_directory.size());
  if (!m_error.empty())
    ImGui::TextColored(ImVec4(1.f, 0.35f, 0.25f, 1.f), "%s", m_error.c_str());

  ImGui::Spacing();
  if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    hide();
    return;
  }

  ImGui::SameLine();
  if (ImGui::Button(button)) {
    std::filesystem::path path(m_directory.data());
    if (!validate(path, m_error))
      return;
    hide();
    if (m_onAccept)
      m_onAccept(path);
  }
}

} // namespace tsd::scivis_studio
