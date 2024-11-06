// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "anari_viewer/ui_anari.h"
#include "anari_viewer/windows/Window.h"
// glfw
#include <GLFW/glfw3.h>
// anari
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/anari_cpp.hpp>
// std
#include <array>
#include <limits>
// tsd
#include "tsd/core/Object.hpp"
#include "tsd/core/UpdateDelegate.hpp"

#include "AppContext.h"
#include "ViewState.h"
#include "Manipulator.h"

namespace tsd_viewer {

struct DistributedViewport : public anari_viewer::windows::Window
{
  DistributedViewport(AppContext *state,
      RemoteAppStateWindow *win,
      const char *rendererSubtype,
      const char *name = "Viewport");
  ~DistributedViewport();

  void buildUI() override;

  void setWorld(anari::World world = nullptr, bool resetCameraView = true);

  void setManipulator(manipulators::Orbit *m);

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

  AppContext *m_context{nullptr};
  RemoteAppStateWindow *m_win{nullptr};

  tsd::math::float2 m_previousMouse{-1.f, -1.f};
  bool m_mouseRotating{false};
  bool m_manipulating{false};
  bool m_contextMenuVisible{false};
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

  tsd::Object m_rendererObject{ANARI_RENDERER, "default"};

  struct RendererUpdateDelegate : public tsd::EmptyUpdateDelegate
  {
    void signalParameterUpdated(
        const tsd::Object *o, const tsd::Parameter *p) override;
    anari::Device d{nullptr};
    anari::Renderer r{nullptr};
    size_t *version{nullptr};
  } m_rud;

  // camera manipulator

  int m_arcballUp{1};
  manipulators::Orbit m_localArcball;
  manipulators::Orbit *m_arcball{nullptr};
  manipulators::UpdateToken m_cameraToken{0};
  float m_apertureRadius{0.f};
  float m_focusDistance{1.f};

  // OpenGL + display

  GLuint m_framebufferTexture{0};
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
  std::string m_contextMenuName;
};

} // namespace tsd_viewer
