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
  ImGui::Text("Add:");
  ImGui::SameLine();

  if (ImGui::Button("current view"))
    m_core->addCurrentViewToCameraPoses();
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("insert new view using the current camera view");

  ImGui::SameLine();
  if (ImGui::Button("turntable views"))
    ImGui::OpenPopup("CameraPoses_turntablePopupMenu");

  ImGui::SameLine();
  ImGui::Text(" | ");
  ImGui::SameLine();

  if (ImGui::Button("clear"))
    ImGui::OpenPopup("CameraPoses_confirmPopupMenu");

  ImGui::Separator();

  int i = 0;
  int toRemove = -1;

  const ImGuiTableFlags flags = ImGuiTableFlags_RowBg
      | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV;

  if (ImGui::BeginTable("camera poses", 4, flags)) {
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

  if (toRemove >= 0)
    m_core->view.poses.erase(m_core->view.poses.begin() + toRemove);

  buildUI_turntablePopupMenu();
  buildUI_confirmPopupMenu();
}

void CameraPoses::buildUI_turntablePopupMenu()
{
  if (ImGui::BeginPopup("CameraPoses_turntablePopupMenu")) {
    ImGui::InputFloat3("azimuths", &m_turntableAzimuths.x, "%.3f");
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("{min, max, step size}");

    ImGui::InputFloat3("elevations", &m_turntableElevations.x, "%.3f");
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("{min, max, step size}");

    ImGui::InputFloat3("center", &m_turntableCenter.x, "%.3f");
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("view center");

    ImGui::InputFloat("distance", &m_turntableDistance, 0.01f, 0.1f, "%.3f");
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("view distance from center");

    if (ImGui::Button("ok")) {
      m_core->addTurntableCameraPoses(m_turntableAzimuths,
          m_turntableElevations,
          m_turntableCenter,
          m_turntableDistance);
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("cancel"))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
}

void CameraPoses::buildUI_confirmPopupMenu()
{
  if (ImGui::BeginPopup("CameraPoses_confirmPopupMenu")) {
    ImGui::Text("are you sure?");
    if (ImGui::Button("yes")) {
      m_core->removeAllPoses();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("cancel"))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
}

} // namespace tsd_viewer
