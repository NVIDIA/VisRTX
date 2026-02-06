// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_app
#include "tsd/app/Core.h"
// tsd_ui_imgui
#include "tsd/ui/imgui/windows/Window.h"
// std
#include <string>
#include <vector>

namespace tsd::ui::imgui {

struct Animations : public Window
{
  Animations(Application *app, const char *name = "Animations");

  void buildUI() override;

 private:
  void buildUI_animationControls();
  void buildUI_editAnimation(tsd::core::Animation *animation);

  // Data //

  bool m_playing{false};
};

} // namespace tsd::ui::imgui
