// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AppSettings.h"

namespace tsd_viewer {

AppSettings::AppSettings() : Modal("AppSettings")
{
  update();
}

void AppSettings::buildUI()
{
  bool doUpdate = false;

  doUpdate |= ImGui::DragFloat("font size", &m_fontScale, 0.01f, 1.f, 4.f);

  if (doUpdate)
    update();
}

void AppSettings::update()
{
  ImGuiIO &io = ImGui::GetIO();
  io.FontGlobalScale = m_fontScale;
}

} // namespace tsd_viewer
