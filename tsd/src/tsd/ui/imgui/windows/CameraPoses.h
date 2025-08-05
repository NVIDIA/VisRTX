// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"
// tsd
#include "tsd/core/TSDMath.hpp"

namespace tsd::ui::imgui {

struct CameraPoses : public Window
{
  CameraPoses(Application *app, const char *name = "Camera Poses");
  void buildUI() override;

 private:
  void buildUI_turntablePopupMenu();
  void buildUI_confirmPopupMenu();

  tsd::math::float3 m_turntableCenter{0.f, 0.f, 0.f};
  tsd::math::float3 m_turntableAzimuths{0.f, 360.f, 20.f};
  tsd::math::float3 m_turntableElevations{0.f, 45.f, 10.f};
  float m_turntableDistance{1.f};
};

} // namespace tsd::ui::imgui
