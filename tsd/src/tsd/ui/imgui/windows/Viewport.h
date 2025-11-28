// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"

#include "tsd/ui/imgui/tsd_ui_imgui.h"

// tsd_core
#include "tsd/core/scene/Object.hpp"
#include "tsd/core/scene/UpdateDelegate.hpp"
#include "tsd/core/scene/objects/Camera.hpp"

// tsd_rendering
#include "tsd/rendering/index/RenderIndex.hpp"
#include "tsd/rendering/pipeline/RenderPipeline.h"
#include "tsd/rendering/view/Manipulator.hpp"
#include "tsd/rendering/view/CameraUpdateDelegate.hpp"

// ImGuizmo
#include <ImGuizmo.h>

// std
#include <array>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <vector>
#include <string>

namespace tsd::ui::imgui {

using ViewportDeviceChangeCb = std::function<void(const std::string &)>;

struct Viewport : public Window
{
  Viewport(Application *app,
      tsd::rendering::Manipulator *m,
      const char *name = "Viewport");
  ~Viewport();

  void buildUI() override;
  void setManipulator(tsd::rendering::Manipulator *m);
  void resetView(bool resetAzEl = true);
  void centerView();
  void setLibrary(const std::string &libName, bool doAsync = true);
  void setDeviceChangeCb(ViewportDeviceChangeCb cb);
  void setExternalInstances(
      const anari::Instance *instances = nullptr, size_t count = 0);

  void setDatabaseCamera(tsd::core::CameraRef cam);
  void clearDatabaseCamera();
  void createCameraFromCurrentView();

 private:
  void saveSettings(tsd::core::DataNode &thisWindowRoot) override;
  void loadSettings(tsd::core::DataNode &thisWindowRoot) override;

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

  void applyCameraParameters(tsd::core::Camera *cam);

  void echoCameraConfig();
  void ui_menubar();
  void ui_handleInput();
  bool ui_picking();
  void ui_overlay();
  void ui_gizmo();
  bool canShowGizmo() const;

  int windowFlags() const override; // anari_viewer::Window

  // Data /////////////////////////////////////////////////////////////////////

  ViewportDeviceChangeCb m_deviceChangeCb;
  float m_timeToLoadDevice{0.f};
  std::future<void> m_initFuture;
  bool m_deviceReadyToUse{false};
  std::string m_libName;
  tsd::rendering::RenderIndex *m_rIdx{nullptr};

  tsd::math::float2 m_previousMouse{-1.f, -1.f};
  bool m_mouseRotating{false};
  bool m_manipulating{false};
  bool m_frameCancelled{false};
  bool m_echoCameraConfig{false};

  bool m_showOverlay{true};
  bool m_showCameraInfo{false};
  bool m_highlightSelection{true};
  bool m_showOnlySelected{false};
  int m_frameSamples{0};

  tsd::rendering::AOVType m_visualizeAOV{tsd::rendering::AOVType::NONE};
  bool m_showAxes{true};
  float m_depthVisualMinimum{0.f};
  float m_depthVisualMaximum{1.f};

  float m_fov{40.f};

  // Gizmo state //

  bool m_enableGizmo{true};
  ImGuizmo::OPERATION m_gizmoOperation{ImGuizmo::TRANSLATE};
  ImGuizmo::MODE m_gizmoMode{ImGuizmo::WORLD};

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
  std::vector<tsd::core::Object> m_rendererObjects;
  int m_currentRenderer{0};

  struct RendererUpdateDelegate : public tsd::core::EmptyUpdateDelegate
  {
    void signalParameterUpdated(
        const tsd::core::Object *o, const tsd::core::Parameter *p) override;
    anari::Device d{nullptr};
    anari::Renderer r{nullptr};
  } m_rud;

  // Camera manipulator //

  int m_arcballUp{1};
  tsd::rendering::Manipulator m_localArcball;
  tsd::rendering::Manipulator *m_arcball{nullptr};
  tsd::rendering::UpdateToken m_cameraToken{0};
  float m_apertureRadius{0.f};
  float m_focusDistance{1.f};

  // Database camera state //

  tsd::core::CameraRef m_selectedCamera;
  std::unique_ptr<tsd::core::CameraUpdateDelegate> m_cameraDelegate;
  std::vector<tsd::core::CameraRef> m_menuCameraRefs;

  // Display //

  tsd::rendering::RenderPipeline m_pipeline;
  tsd::rendering::AnariSceneRenderPass *m_anariPass{nullptr};
  tsd::rendering::PickPass *m_pickPass{nullptr};
  tsd::rendering::VisualizeAOVPass *m_visualizeAOVPass{nullptr};
  tsd::rendering::OutlineRenderPass *m_outlinePass{nullptr};
  tsd::rendering::AnariAxesRenderPass *m_axesPass{nullptr};
  tsd::rendering::CopyToSDLTexturePass *m_outputPass{nullptr};
  tsd::rendering::SaveToFilePass *m_saveToFilePass{nullptr};

  tsd::math::int2 m_viewportSize{0, 0};
  tsd::math::int2 m_renderSize{0, 0};
  float m_resolutionScale{1.f};

  float m_latestFL{0.f};
  float m_latestAnariFL{0.f};
  float m_minFL{std::numeric_limits<float>::max()};
  float m_maxFL{-std::numeric_limits<float>::max()};

  std::string m_overlayWindowName;
};

} // namespace tsd::ui::imgui
