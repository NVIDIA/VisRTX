// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_ui_imgui
#include <tsd/ui/imgui/windows/Window.h>
// tsd_core
#include <tsd/core/scene/Object.hpp>

namespace tsd::demo {

struct InstancingControls : public tsd::ui::imgui::Window
{
  InstancingControls(tsd::ui::imgui::Application *app,
      const char *name = "Instancing Controls");

  void buildUI() override;

 private:
  void createScene();
  void generateSpheres();
  void generateInstances();

  // Data //

  int m_numInstances{5000};
  float m_spacing{25.f};
  float m_particleRadius{0.5f};
  bool m_addSpheres{true};
  bool m_addInstances{true};
  tsd::core::Object *m_light{nullptr};
};

} // namespace tsd::demo
