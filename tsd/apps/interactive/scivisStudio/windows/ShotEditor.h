// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"
#include "tsd/ui/imgui/windows/Window.h"

#include <functional>

namespace tsd::scivis_studio {

struct ShotEditor : public tsd::ui::imgui::Window
{
  ShotEditor(tsd::ui::imgui::Application *app,
      ProjectContext *projectContext,
      std::function<void()> onRender);
  ~ShotEditor() override;

  void buildUI() override;

 private:
  bool inputText(const char *label, std::string &value, size_t capacity = 512);

  ProjectContext *m_projectContext{nullptr};
  std::function<void()> m_onRender;
};

} // namespace tsd::scivis_studio
