// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CameraPoses.h"
#include "../AppCore.h"
// imgui
#include <misc/cpp/imgui_stdlib.h>

namespace tsd_viewer {

CameraPoses::CameraPoses(AppCore *core, const char *name) : Window(core, name)
{}

void CameraPoses::buildUI()
{
  if (ImGui::Button("save current"))
    m_core->addCurrentViewToCameraPoses();

  ImGui::Separator();

  int i = 0;
  int toRemove = -1;
  for (auto &p : m_core->view.poses) {
    ImGui::PushID(&p);
    ImGui::InputText("|", &p.name);
    ImGui::SameLine();
    if (ImGui::Button("set"))
      m_core->setCameraPose(p);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("set as current view");
    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();
    if (ImGui::Button("X"))
      toRemove = i;
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("delete this pose");
    ImGui::PopID();
    i++;
  }

  if (toRemove >= 0)
    m_core->view.poses.erase(m_core->view.poses.begin() + toRemove);
}

} // namespace tsd_viewer
