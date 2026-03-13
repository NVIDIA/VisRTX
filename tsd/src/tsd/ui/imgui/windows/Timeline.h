// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_app
#include "tsd/app/Context.h"
// tsd_ui_imgui
#include "tsd/ui/imgui/windows/Window.h"
// std
#include <set>
#include <string>

namespace tsd::ui::imgui {

struct Timeline : public Window
{
  Timeline(Application *app, const char *name = "Timeline");

  void buildUI() override;

 private:
  void buildUI_transport();
  void buildUI_canvas();

  // Transport state //
  bool m_playing{false};
  bool m_loop{true};

  // Canvas state //
  float m_pixelsPerFrame{8.f};
  float m_canvasScrollX{0.f};
  std::set<size_t> m_selectedTracks;
};

} // namespace tsd::ui::imgui
