// Copyright 2024-2025 NVIDIA Corporation
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
#include <future>
#include <limits>
// tsd
#include "tsd/core/Object.hpp"
#include "tsd/core/UpdateDelegate.hpp"
// render_pipeline
#include "render_pipeline/RenderPipeline.h"

#include "../AppCore.h"
#include "../Manipulator.h"

namespace tsd_viewer {

struct Viewport : public anari_viewer::windows::Window
{
  Viewport(
      AppCore *state, manipulators::Orbit *m, const char *name = "Viewport");
  ~Viewport();

  void buildUI() override;
  void setManipulator(manipulators::Orbit *m);
  void resetView(bool resetAzEl = true);
  void setLibrary(const std::string &libName, bool doAsync = true);

 private:
  void teardownDevice();
  void reshape(tsd::math::int2 newWindowSize);
  void pick(tsd::math::int2 location, bool selectObject);
  void setSelectionVisibilityFilterEnabled(bool enabled);

  void updateFrame();
  void updateCamera(bool force = false);
  void updateImage();

  void echoCameraConfig();
  void ui_handleInput();
  void ui_picking();
  void ui_contextMenu();
  void ui_overlay();

  // Data /////////////////////////////////////////////////////////////////////

  float m_timeToLoadDevice{0.f};
  std::future<void> m_initFuture;
  bool m_deviceReadyToUse{false};
  AppCore *m_core{nullptr};
  std::string m_libName;
  tsd::RenderIndex *m_rIdx{nullptr};

  tsd::math::float2 m_previousMouse{-1.f, -1.f};
  float m_pickedDepth{0.f};
  bool m_mouseRotating{false};
  bool m_manipulating{false};
  bool m_coreMenuVisible{false};
  bool m_frameCancelled{false};
  bool m_saveNextFrame{false};
  bool m_echoCameraConfig{false};

  bool m_showOverlay{true};
  bool m_showCameraInfo{false};
  bool m_highlightSelection{true};
  bool m_showOnlySelected{false};
  int m_frameSamples{0};
  bool m_useOrthoCamera{false};

  bool m_visualizeDepth{false};
  float m_depthVisualMaximum{1.f};

  float m_fov{40.f};

  // ANARI objects //

  anari::DataType m_format{ANARI_UFIXED8_RGBA_SRGB};

  anari::Device m_device{nullptr};
  anari::Camera m_perspCamera{nullptr};
  anari::Camera m_orthoCamera{nullptr};

  std::vector<std::string> m_rendererNames;
  std::vector<anari::Renderer> m_renderers;
  std::vector<tsd::Object> m_rendererObjects;
  int m_currentRenderer{0};

  struct RendererUpdateDelegate : public tsd::EmptyUpdateDelegate
  {
    void signalParameterUpdated(
        const tsd::Object *o, const tsd::Parameter *p) override;
    anari::Device d{nullptr};
    anari::Renderer r{nullptr};
  } m_rud;

  // camera manipulator

  int m_arcballUp{1};
  manipulators::Orbit m_localArcball;
  manipulators::Orbit *m_arcball{nullptr};
  manipulators::UpdateToken m_cameraToken{0};
  float m_apertureRadius{0.f};
  float m_focusDistance{1.f};

  // OpenGL + display

  tsd::RenderPipeline m_pipeline;
  tsd::AnariRenderPass *m_anariPass{nullptr};
  tsd::VisualizeDepthPass *m_visualizeDepthPass{nullptr};
  tsd::OutlineRenderPass *m_outlinePass{nullptr};
  tsd::CopyToGLImagePass *m_outputPass{nullptr};

  tsd::math::int2 m_viewportSize{0, 0};
  tsd::math::int2 m_renderSize{0, 0};
  float m_resolutionScale{1.f};

  float m_latestFL{0.f};
  float m_latestAnariFL{0.f};
  float m_minFL{std::numeric_limits<float>::max()};
  float m_maxFL{-std::numeric_limits<float>::max()};

  std::string m_overlayWindowName;
  std::string m_coreMenuName;
};

} // namespace tsd_viewer
