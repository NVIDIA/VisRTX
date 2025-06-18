// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "windows/Window.h"
// tsd
#include "tsd/core/Object.hpp"

namespace tsd_viewer {

struct InstancingControls : public Window
{
  InstancingControls(AppCore *core, const char *name = "Instancing Controls");

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
  tsd::Object *m_light{nullptr};
};

} // namespace tsd_viewer
