// Copyright 2023-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_ui_imgui
#include <tsd/ui/imgui/windows/Window.h>

namespace tsd::viskores_graph {

class NodeInfoWindow : public tsd::ui::imgui::Window
{
 public:
  NodeInfoWindow(tsd::ui::imgui::Application *app);
  ~NodeInfoWindow() override = default;

  void setText(std::string text);

  void buildUI() override;

 private:
  std::string m_text{"<no summary>"};
};

} // namespace tsd::viskores_graph
