// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "BaseViewport.h"

// tsd_core
#include "tsd/scene/objects/Camera.hpp"
// tsd_rendering
#include "tsd/rendering/index/RenderIndex.hpp"
#include "tsd/rendering/pipeline/ImagePipeline.h"
#include "tsd/rendering/view/Manipulator.hpp"
// anari
#include <anari/frontend/anari_enums.h>
// std
#include <functional>
#include <future>
#include <limits>
#include <string>

namespace tsd::ui::imgui {

using ViewportDeviceChangeCb = std::function<void(const std::string &)>;

struct Viewport : public BaseViewport
{
  Viewport(Application *app,
      tsd::rendering::Manipulator *m,
      const char *name = "Viewport");
  ~Viewport();

  void buildUI() override;
  void setLibrary(const std::string &libName, bool doAsync = true);
  void setLibraryToDefault();
  void setDeviceChangeCb(ViewportDeviceChangeCb cb);
  void setExternalInstances(
      const anari::Instance *instances = nullptr, size_t count = 0);
  void setCustomFrameParameter(const char *name, const tsd::core::Any &value);

 private:
  void saveSettings(tsd::core::DataNode &thisWindowRoot) override;
  void loadSettings(tsd::core::DataNode &thisWindowRoot) override;

  void imagePipeline_populate(tsd::rendering::ImagePipeline &p) override;

  void camera_resetView(bool resetAzEl = true) override;
  void camera_centerView() override;

  void renderer_resetParameterDefaults() override;

  void teardownDevice();
  void pick(tsd::math::int2 location, bool selectObject);
  void setSelectionVisibilityFilterEnabled(bool enabled);

  void updateFrame();
  void updateImage();
  void updateAxes();

  void ui_menubar();
  void ui_menubar_Device();
  void ui_menubar_Camera();
  void ui_menubar_Viewport();
  void ui_menubar_World();

  bool ui_picking();
  void ui_overlay();

  // Data /////////////////////////////////////////////////////////////////////

  size_t m_defragToken{0};

  ViewportDeviceChangeCb m_deviceChangeCb;
  float m_timeToLoadDevice{0.f};
  std::future<void> m_initFuture;
  std::string m_libName;
  tsd::rendering::RenderIndex *m_rIdx{nullptr};
  tsd::app::RenderIndexKind m_lastIndexKind{
      tsd::app::RenderIndexKind::ALL_LAYERS};

  bool m_refreshDeviceNextFrame{false};
  bool m_showOverlay{true};
  bool m_showAxes{true};
  bool m_highlightSelection{true};
  bool m_showOnlySelected{false};
  int m_frameSamples{0};

  tsd::rendering::AOVType m_visualizeAOV{tsd::rendering::AOVType::NONE};
  float m_depthVisualMinimum{0.f};
  float m_depthVisualMaximum{1.f};
  bool m_edgeInvert{false};
  anari::DataType m_colorFormat{ANARI_UFIXED8_RGBA_SRGB};

  // Picking state //

  bool m_selectObjectNextPick{false};
  tsd::math::int2 m_pickCoord{0, 0};
  float m_pickedDepth{0.f};

  // ANARI objects //

  anari::Device m_device{nullptr};
  tsd::scene::RendererAppRef m_prevRenderer;
  tsd::scene::CameraAppRef m_prevCamera;
  tsd::core::ObjectVersion m_lastCameraChange{};

  // Display //

  tsd::rendering::AnariSceneRenderPass *m_anariPass{nullptr};
  tsd::rendering::PickPass *m_pickPass{nullptr};
  tsd::rendering::VisualizeAOVPass *m_visualizeAOVPass{nullptr};
  tsd::rendering::OutlineRenderPass *m_outlinePass{nullptr};
  tsd::rendering::AnariAxesRenderPass *m_axesPass{nullptr};
  tsd::rendering::CopyToSDLTexturePass *m_outputPass{nullptr};
  tsd::rendering::SaveToFilePass *m_saveToFilePass{nullptr};

  float m_latestFL{0.f};
  float m_latestAnariFL{0.f};
  float m_minFL{std::numeric_limits<float>::max()};
  float m_maxFL{-std::numeric_limits<float>::max()};
};

} // namespace tsd::ui::imgui
