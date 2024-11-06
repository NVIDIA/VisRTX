// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppContext.h"
// anari_viewer
#include "anari_viewer/windows/Window.h"

namespace tsd_viewer {

struct InstancingControls : public anari_viewer::windows::Window
{
  InstancingControls(
      AppContext *state, const char *name = "Instancing Controls");

  void buildUI() override;

 private:
  void createScene();
  void generateSpheres();
  void generateInstances();

  AppContext *m_context{nullptr};
  int m_numInstances{5000};
  float m_spacing{25.f};
  float m_particleRadius{0.5f};
  bool m_addSpheres{true};
  bool m_addInstances{true};
  tsd::Object *m_light{nullptr};
};

} // namespace tsd_viewer
