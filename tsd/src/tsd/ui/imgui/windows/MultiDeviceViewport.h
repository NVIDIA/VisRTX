// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_ui_imgui
#include <tsd/ui/imgui/windows/Window.h>
// tsd_rendering
#include <tsd/rendering/pipeline/RenderPipeline.h>
#include <tsd/rendering/index/RenderIndexAllLayers.hpp>
#include <tsd/rendering/view/Manipulator.hpp>
// std
#include <memory>

namespace tsd::ui::imgui {

struct MultiDeviceViewport : public Window
{
  MultiDeviceViewport(Application *app,
      tsd::rendering::Manipulator *m,
      const char *name = "DP Viewport");
  ~MultiDeviceViewport();

  void buildUI() override;
  void setManipulator(tsd::rendering::Manipulator *m);
  void resetView(bool resetAzEl = true);
  void centerView();

  void setLibrary(const std::string &libName);

 private:
  void loadSettings(tsd::core::DataNode &thisWindowRoot) override;

  void getSceneBounds(tsd::math::float3 bounds[2]) const;
  tsd::rendering::RenderIndexAllLayers *getRenderIndex(size_t i = 0) const;
  void setupRenderPipeline(const std::vector<anari::Device> &devices);
  void reshape(tsd::math::int2 newWindowSize);
  void updateCamera(bool force = false);

  void loadANARIRendererParameters();
  void updateAllRendererParameters();

  void ui_menubar();
  void ui_handleInput();

  int windowFlags() const override; // anari_viewer::Window

  // ImGui input state //

  tsd::math::float2 m_previousMouse{-1.f, -1.f};
  bool m_mouseRotating{false};
  bool m_manipulating{false};

  // rendering //

  anari::DataType m_format{ANARI_UFIXED8_RGBA_SRGB};
  std::vector<anari::Camera> m_cameras;
  tsd::core::Object m_rendererObject;

  struct RendererUpdateDelegate : public tsd::core::EmptyUpdateDelegate
  {
    void signalParameterUpdated(
        const tsd::core::Object *o, const tsd::core::Parameter *p) override;
    std::vector<anari::Device> devices;
    std::vector<anari::Renderer> renderers;
  } m_rud;

  // camera manipulator //

  int m_arcballUp{1};
  tsd::rendering::Manipulator m_localArcball;
  tsd::rendering::Manipulator *m_arcball{nullptr};
  tsd::rendering::UpdateToken m_cameraToken{0};
  float m_fov{40.f};
  float m_apertureRadius{0.f};
  float m_focusDistance{1.f};

  // display //

  bool m_showAxes{true};

  tsd::rendering::RenderPipeline m_pipeline;
  tsd::rendering::MultiDeviceSceneRenderPass *m_anariPass{nullptr};
  tsd::rendering::AnariAxesRenderPass *m_axesPass{nullptr};
  tsd::rendering::CopyToSDLTexturePass *m_outputPass{nullptr};

  tsd::math::int2 m_viewportSize{0, 0};
  tsd::math::int2 m_renderSize{0, 0};
  float m_resolutionScale{1.f};
};

} // namespace tsd::ui::imgui