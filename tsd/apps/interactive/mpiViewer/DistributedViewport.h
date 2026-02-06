// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "anari_viewer/ui_anari.h"
// SDL
#include <SDL3/SDL.h>
// std
#include <array>
#include <limits>
// tsd_core
#include <tsd/core/scene/Object.hpp>
#include <tsd/core/scene/UpdateDelegate.hpp>
// tsd_app
#include <tsd/app/Core.h>
// tsd_rendering
#include <tsd/rendering/view/Manipulator.hpp>
// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/windows/Window.h>

#include "DistributedSceneController.h"

namespace tsd::mpi_viewer {

struct DistributedViewport : public tsd::ui::imgui::Window
{
  DistributedViewport(tsd::ui::imgui::Application *app,
      DistributedSceneController *dapp,
      const char *name = "Viewport");
  ~DistributedViewport();

  void buildUI() override;

  void setManipulator(tsd::rendering::Manipulator *m);
  void resetView(bool resetAzEl = true);

 private:
  void reshape(tsd::math::int2 newWindowSize);

  void updateCamera(bool force = false);
  void updateImage();

  void ui_handleInput();
  void ui_menuBar();
  void ui_overlay();
  void ui_timeControls();

  int windowFlags() const override; // anari_viewer::Window

  // Data /////////////////////////////////////////////////////////////////////

  DistributedSceneController *m_dapp{nullptr};

  tsd::math::float2 m_previousMouse{-1.f, -1.f};
  bool m_mouseRotating{false};
  bool m_manipulating{false};
  bool m_coreMenuVisible{false};
  bool m_saveNextFrame{false};
  int m_screenshotIndex{0};
  bool m_showOverlay{true};
  bool m_showTimeline{true};

  // ANARI objects //

  anari::DataType m_format{ANARI_UFIXED8_RGBA_SRGB};
  LocalState m_localState;

  // camera manipulator

  int m_arcballUp{1};
  tsd::rendering::Manipulator m_localArcball;
  tsd::rendering::Manipulator *m_arcball{nullptr};
  tsd::rendering::UpdateToken m_cameraToken{0};
  float m_fov{40.f};
  float m_apertureRadius{0.f};
  float m_focusDistance{1.f};

  // display

  SDL_Texture *m_framebufferTexture{nullptr};
  tsd::math::int2 m_viewportSize{1920, 1080};
  tsd::math::int2 m_renderSize{1920, 1080};
  float m_resolutionScale{1.f};

  float m_latestFL{1.f};
  float m_minFL{std::numeric_limits<float>::max()};
  float m_maxFL{-std::numeric_limits<float>::max()};

  std::string m_overlayWindowName;
  std::string m_coreMenuName;
};

} // namespace tsd::mpi_viewer
