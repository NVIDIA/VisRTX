// Copyright 2024-2025 NVIDIA Corporation
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

#include "ViewState.h"

namespace tsd::ptc {

struct DistributedViewport : public tsd::ui::imgui::Window
{
  DistributedViewport(tsd::ui::imgui::Application *app,
      RemoteAppStateWindow *win,
      const char *rendererSubtype,
      const char *name = "Viewport");
  ~DistributedViewport();

  void buildUI() override;

  void setWorld(anari::World world = nullptr, bool resetCameraView = true);
  void setManipulator(tsd::rendering::Manipulator *m);
  void resetView(bool resetAzEl = true);
  void setDevice(anari::Device d);

 private:
  void teardownDevice();
  void reshape(tsd::math::int2 newWindowSize);

  void updateFrame();
  void updateCamera(bool force = false);
  void updateImage();

  void writeRemoteData();

  void ui_handleInput();
  void ui_contextMenu();
  void ui_overlay();

  // Data /////////////////////////////////////////////////////////////////////

  RemoteAppStateWindow *m_win{nullptr};

  tsd::math::float2 m_previousMouse{-1.f, -1.f};
  bool m_mouseRotating{false};
  bool m_manipulating{false};
  bool m_coreMenuVisible{false};
  bool m_saveNextFrame{false};
  int m_screenshotIndex{0};

  bool m_showOverlay{true};

  float m_fov{40.f};

  // ANARI objects //

  std::string m_rendererSubtype;

  anari::DataType m_format{ANARI_UFIXED8_RGBA_SRGB};

  anari::Device m_device{nullptr};
  anari::Frame m_frame{nullptr};
  anari::World m_world{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Renderer m_renderer{nullptr};

  tsd::core::Object m_rendererObject{ANARI_RENDERER, "default"};

  struct RendererUpdateDelegate : public tsd::core::EmptyUpdateDelegate
  {
    void signalParameterUpdated(
        const tsd::core::Object *o, const tsd::core::Parameter *p) override;
    anari::Device d{nullptr};
    anari::Renderer r{nullptr};
    size_t *version{nullptr};
  } m_rud;

  // camera manipulator

  int m_arcballUp{1};
  tsd::rendering::Manipulator m_localArcball;
  tsd::rendering::Manipulator *m_arcball{nullptr};
  tsd::rendering::UpdateToken m_cameraToken{0};
  float m_apertureRadius{0.f};
  float m_focusDistance{1.f};

  // display

  SDL_Texture *m_framebufferTexture{nullptr};
  tsd::math::int2 m_viewportSize{1920, 1080};
#if 1
  tsd::math::int2 m_renderSize{1920, 1080};
#else
  tsd::math::int2 m_nextFrameRenderSize{1920, 1080};
  tsd::math::int2 m_currentFrameRenderSize{1920, 1080};
#endif
  float m_resolutionScale{1.f};

  float m_latestFL{1.f};
  float m_minFL{std::numeric_limits<float>::max()};
  float m_maxFL{-std::numeric_limits<float>::max()};

  std::string m_overlayWindowName;
  std::string m_coreMenuName;
};

} // namespace tsd::ptc
