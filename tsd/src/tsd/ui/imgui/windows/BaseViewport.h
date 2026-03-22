// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"
// tsd_rendering
#include "tsd/rendering/pipeline/ImagePipeline.h"
// ImGuizmo
#include <ImGuizmo.h>

namespace tsd::ui::imgui {

/*
 * This is a base class for viewports in the TSD UI. It provides common
 * functionality for managing default cameras, renderers, and an image pipeline,
 * as well as a common UI for interacting with these things. It is intended to
 * be subclassed for specific viewport types (eg. a viewport that shows the main
 * scene, a viewport that shows a render preview, etc.) that can customize the
 * UI and rendering as needed.
 */
struct BaseViewport : public Window
{
  BaseViewport(Application *app, const char *name);
  virtual ~BaseViewport() override;

  void buildUI() override;
  void setManipulator(tsd::rendering::Manipulator *m);

 protected:
  void saveSettings(tsd::core::DataNode &thisWindowRoot) override;
  void loadSettings(tsd::core::DataNode &thisWindowRoot) override;

  /////////////////////////////////////////////////////////////////////////////
  // The following represents the internal API for child classes to use common
  // functionality -- methods are organized by theme using a prefix.
  void viewport_setActive(bool active); // global toggle for display/input/etc.
  bool viewport_isActive() const;
  virtual void viewport_reshape(tsd::math::int2 newWindowSize);

  virtual void imagePipeline_populate(tsd::rendering::ImagePipeline &p) = 0;
  void imagePipeline_setup();
  bool imagePipeline_isSetup() const;
  void imagePipeline_setDimensions(uint32_t width, uint32_t height);
  void imagePipeline_render();
  void imagePipeline_teardown();
  const tsd::rendering::ImagePipeline &imagePipeline() const;

  void camera_update(bool force = false);
  void camera_setCurrent(tsd::scene::CameraAppRef c);
  virtual void camera_resetView(bool resetAzEl = true) = 0;
  virtual void camera_centerView() = 0;

  virtual void renderer_resetParameterDefaults() = 0;

  bool gizmo_canShow() const;

  void ui_handleInput();
  void ui_gizmo();
  void ui_menubar_Renderer();
  void ui_menubar_Camera();
  void ui_menubar_TransformManipulator();
  /////////////////////////////////////////////////////////////////////////////

  struct CameraState
  {
    tsd::scene::CameraAppRef current;
    tsd::rendering::Manipulator localArcball;
    tsd::rendering::Manipulator *arcball{nullptr};
    tsd::rendering::UpdateToken arcballToken{0};
  } m_camera;

  struct ViewportState
  {
    bool active{false};
    tsd::math::int2 size{0, 0};
    tsd::math::int2 renderSize{0, 0};
    float resolutionScale{1.f};
  } m_viewport;

  struct RendererState
  {
    std::vector<tsd::scene::RendererAppRef> objects;
    tsd::scene::RendererAppRef current;
  } m_renderers;

 private:
  int windowFlags() const override; // anari_viewer::Window

  tsd::rendering::ImagePipeline m_pipeline;

  struct InputState
  {
    tsd::math::float2 previousMouse{-1.f, -1.f};
    bool mouseRotating{false};
    bool manipulating{false};
  } m_input;

  struct GizmoState
  {
    bool active{true};
    ImGuizmo::OPERATION operation{ImGuizmo::OPERATION::TRANSLATE};
    ImGuizmo::MODE mode{ImGuizmo::MODE::WORLD};
  } m_gizmo;
};

} // namespace tsd::ui::imgui
