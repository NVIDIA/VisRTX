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
  if (ImGui::Button("add new"))
    m_core->addCurrentViewToCameraPoses();

  ImGui::Separator();

  int i = 0;
  int toRemove = -1;

  const ImGuiTableFlags flags = ImGuiTableFlags_RowBg
      | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV;

  if (ImGui::BeginTable("parameters", 4, flags)) {
    for (auto &p : m_core->view.poses) {
      ImGui::PushID(&p);

      ImGui::TableNextRow();

      ImGui::TableSetColumnIndex(0);
      ImGui::SetNextItemWidth(-1.f);
      ImGui::InputText("##", &p.name);

      ImGui::TableSetColumnIndex(1);
      if (ImGui::Button(">"))
        m_core->setCameraPose(p);
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("set as current view");

      ImGui::TableSetColumnIndex(2);
      if (ImGui::Button("+")) {
        m_core->updateExistingCameraPoseFromView(p);
        tsd::logStatus("camera pose '%s' updated", p.name.c_str());
      }
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("update this pose from current view");

      ImGui::TableSetColumnIndex(3);
      if (ImGui::Button("-"))
        toRemove = i;
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("delete this pose");
      ImGui::PopID();
      i++;
    }

    ImGui::EndTable();
  }

#if 0
  for (auto &p : m_core->view.poses) {
    ImGui::PushID(&p);
    ImGui::SetNextItemWidth(400.f);
    ImGui::InputText("|", &p.name);
    ImGui::SameLine();
    if (ImGui::Button("0"))
      m_core->setCameraPose(p);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("set as current view");
    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();
    if (ImGui::Button("+"))
      m_core->setCameraPose(p);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("update this pose from view");
    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();
    if (ImGui::Button("x"))
      toRemove = i;
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("delete this pose");
    ImGui::PopID();
    i++;
  }
#endif

  if (toRemove >= 0)
    m_core->view.poses.erase(m_core->view.poses.begin() + toRemove);
}

} // namespace tsd_viewer
