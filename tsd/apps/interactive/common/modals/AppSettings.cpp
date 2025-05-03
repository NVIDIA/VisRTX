// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AppSettings.h"

namespace tsd_viewer {

AppSettings::AppSettings(AppCore *core) : Modal(core, "AppSettings")
{
  ImGuiIO &io = ImGui::GetIO();
  m_fontScale = io.FontGlobalScale;
}

void AppSettings::buildUI()
{
  bool doUpdate = false;

  doUpdate |= ImGui::DragFloat("font size", &m_fontScale, 0.01f, 0.5f, 4.f);

  if (doUpdate)
    update();

  ImGui::NewLine();

  if (ImGui::Button("close") || ImGui::IsKeyDown(ImGuiKey_Escape))
    this->hide();
}

void AppSettings::update()
{
  ImGuiIO &io = ImGui::GetIO();
  io.FontGlobalScale = m_fontScale;
}

} // namespace tsd_viewer
