// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"
// tsd
#include <atomic>
#include <future>
#include <mutex>
#include "tsd/app/Core.h"
#include "tsd/core/TSDMath.hpp"
#include "tsd/core/Timer.hpp"
#include "tsd/rendering/view/CameraPath.h"

namespace tsd::ui::imgui {

struct Viewport;

struct CameraPoses : public Window
{
  CameraPoses(
      Application *app, Viewport *viewport, const char *name = "Camera Poses");
  void buildUI() override;

 private:
  void buildUI_turntablePopupMenu();
  void buildUI_confirmPopupMenu();
  void buildUI_interpolationControls();
  void renderInterpolatedPath();

  tsd::math::float3 m_turntableCenter{0.f, 0.f, 0.f};
  tsd::math::float3 m_turntableAzimuths{0.f, 360.f, 20.f};
  tsd::math::float3 m_turntableElevations{0.f, 45.f, 10.f};
  float m_turntableDistance{1.f};

  bool m_updateViewport{true}; // Update viewport during rendering
  bool m_isRendering{false};
  bool m_cancelRequested{false};
  std::future<void> m_renderFuture;
  tsd::core::Timer m_renderTimer;
  int m_currentFrame{0};
  int m_totalFrames{0};
  std::atomic<bool> m_hasNewPose{false};
  tsd::rendering::CameraPose m_currentPose;
  std::mutex m_poseMutex;
  Viewport *m_viewport{nullptr};
};

} // namespace tsd::ui::imgui
