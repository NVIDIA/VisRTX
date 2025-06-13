// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"

#include "anari_viewer/ui_anari.h"
// std
#include <array>
#include <future>
#include <limits>
// tsd
#include "tsd/core/Object.hpp"
#include "tsd/core/UpdateDelegate.hpp"
#include "tsd/rendering/RenderIndex.hpp"
#include "tsd/view/Manipulator.hpp"
// render_pipeline
#include "render_pipeline/RenderPipeline.h"

namespace tsd_viewer {

struct Viewport : public Window
{
  Viewport(AppCore *state,
      tsd::manipulators::Orbit *m,
      const char *name = "Viewport");
  ~Viewport();

  void buildUI() override;
  void setManipulator(tsd::manipulators::Orbit *m);
  void resetView(bool resetAzEl = true);
  void centerView();
  void setLibrary(const std::string &libName, bool doAsync = true);

 private:
  void saveSettings(tsd::serialization::DataNode &thisWindowRoot) override;
  void loadSettings(tsd::serialization::DataNode &thisWindowRoot) override;

  void loadANARIRendererParameters(anari::Device d);
  void updateAllRendererParameters(anari::Device d);

  void setupRenderPipeline();
  void teardownDevice();
  void reshape(tsd::math::int2 newWindowSize);
  void pick(tsd::math::int2 location, bool selectObject);
  void setSelectionVisibilityFilterEnabled(bool enabled);

  void updateFrame();
  void updateCamera(bool force = false);
  void updateImage();

  void echoCameraConfig();
  void ui_menubar();
  void ui_handleInput();
  bool ui_picking();
  void ui_overlay();

  int windowFlags() const override; // anari_viewer::Window

  // Data /////////////////////////////////////////////////////////////////////

  float m_timeToLoadDevice{0.f};
  std::future<void> m_initFuture;
  bool m_deviceReadyToUse{false};
  std::string m_libName;
  tsd::RenderIndex *m_rIdx{nullptr};

  tsd::math::float2 m_previousMouse{-1.f, -1.f};
  bool m_mouseRotating{false};
  bool m_manipulating{false};
  bool m_frameCancelled{false};
  bool m_saveNextFrame{false};
  bool m_echoCameraConfig{false};

  bool m_showOverlay{true};
  bool m_showCameraInfo{false};
  bool m_highlightSelection{true};
  bool m_showOnlySelected{false};
  int m_frameSamples{0};

  bool m_visualizeDepth{false};
  float m_depthVisualMaximum{1.f};

  float m_fov{40.f};

  // Picking state //

  bool m_selectObjectNextPick{false};
  tsd::math::int2 m_pickCoord{0, 0};
  float m_pickedDepth{0.f};

  // ANARI objects //

  anari::DataType m_format{ANARI_UFIXED8_RGBA_SRGB};

  anari::Extensions m_extensions{};
  anari::Device m_device{nullptr};
  anari::Camera m_currentCamera{nullptr};
  anari::Camera m_perspCamera{nullptr};
  anari::Camera m_orthoCamera{nullptr};
  anari::Camera m_omniCamera{nullptr};

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
  tsd::manipulators::Orbit m_localArcball;
  tsd::manipulators::Orbit *m_arcball{nullptr};
  tsd::manipulators::UpdateToken m_cameraToken{0};
  float m_apertureRadius{0.f};
  float m_focusDistance{1.f};

  // display

  tsd::RenderPipeline m_pipeline;
  tsd::AnariRenderPass *m_anariPass{nullptr};
  tsd::PickPass *m_pickPass{nullptr};
  tsd::VisualizeDepthPass *m_visualizeDepthPass{nullptr};
  tsd::OutlineRenderPass *m_outlinePass{nullptr};
  tsd::CopyToSDLTexturePass *m_outputPass{nullptr};

  tsd::math::int2 m_viewportSize{0, 0};
  tsd::math::int2 m_renderSize{0, 0};
  float m_resolutionScale{1.f};

  float m_latestFL{0.f};
  float m_latestAnariFL{0.f};
  float m_minFL{std::numeric_limits<float>::max()};
  float m_maxFL{-std::numeric_limits<float>::max()};

  std::string m_overlayWindowName;
};

} // namespace tsd_viewer
