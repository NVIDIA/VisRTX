// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ProjectWindow.h"

#include "imgui.h"

namespace tsd::scivis_studio {

ProjectWindow::ProjectWindow(
    tsd::ui::imgui::Application *app, ProjectContext *projectContext)
    : Window(app, "Project"), m_projectContext(projectContext)
{}

ProjectWindow::~ProjectWindow() = default;

void ProjectWindow::buildUI()
{
  if (!m_projectContext)
    return;

  auto &project = m_projectContext->project();
  ImGui::Text("Name: %s", project.name.c_str());
  ImGui::Text("Path: %s",
      project.projectDirectory.empty()
          ? "{unsaved}"
          : project.projectDirectory.string().c_str());
  ImGui::Text("Status: %s", project.dirty ? "dirty" : "clean");

  ImGui::SeparatorText("Datasets");
  if (project.datasets.empty())
    ImGui::TextDisabled("No datasets");
  for (const auto &dataset : project.datasets)
    ImGui::BulletText(
        "%s  [%s]",
        dataset.name.c_str(),
        dataset::toString(dataset.status));

  ImGui::SeparatorText("Shots");
  for (auto &shot : project.shots) {
    const bool selected = shot.id == project.activeShotId;
    if (ImGui::Selectable(shot.name.c_str(), selected)) {
      project.activeShotId = shot.id;
      m_projectContext->syncAnimationManagerToActiveShot();
      m_projectContext->applyActiveShot();
    }
  }
}

} // namespace tsd::scivis_studio
