// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AppSettingsDialog.h"

namespace tsd_viewer {

AppSettingsDialog::AppSettingsDialog(AppCore *core) : Modal(core, "AppSettings")
{}

void AppSettingsDialog::buildUI()
{
  bool doUpdate = false;

  doUpdate |= ImGui::DragFloat(
      "font size", &m_core->windows.fontScale, 0.01f, 0.5f, 4.f);

  doUpdate |= ImGui::DragFloat(
      "rounding", &m_core->windows.uiRounding, 0.01f, 0.f, 12.f);

  if (doUpdate)
    applySettings();

  ImGui::NewLine();

  if (ImGui::Button("close") || ImGui::IsKeyDown(ImGuiKey_Escape))
    this->hide();
}

void AppSettingsDialog::applySettings()
{
  ImGuiIO &io = ImGui::GetIO();
  io.FontGlobalScale = m_core->windows.fontScale;

  ImGuiStyle &style = ImGui::GetStyle();
  style.WindowRounding = m_core->windows.uiRounding;
  style.ChildRounding = m_core->windows.uiRounding;
  style.FrameRounding = m_core->windows.uiRounding;
  style.ScrollbarRounding = m_core->windows.uiRounding;
  style.GrabRounding = m_core->windows.uiRounding;
  style.PopupRounding = m_core->windows.uiRounding;
}

} // namespace tsd_viewer
