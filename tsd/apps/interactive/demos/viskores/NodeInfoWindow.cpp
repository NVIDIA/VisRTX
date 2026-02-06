// Copyright 2023-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "NodeInfoWindow.h"

namespace tsd::viskores_graph {

NodeInfoWindow::NodeInfoWindow(tsd::ui::imgui::Application *app)
    : tsd::ui::imgui::Window(app, "Node Info")
{}

void NodeInfoWindow::setText(std::string text)
{
  m_text = text;
}

void NodeInfoWindow::buildUI()
{
  ImGui::TextWrapped("%s", m_text.c_str());
}

} // namespace tsd::viskores_graph
