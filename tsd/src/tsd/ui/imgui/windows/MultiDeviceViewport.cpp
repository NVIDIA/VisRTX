// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "MultiDeviceViewport.h"
// tsd_core
#include <tsd/core/Logging.hpp>
// tsd_ui_imgui
#include <tsd/ui/imgui/Application.h>
#include <tsd/ui/imgui/tsd_ui_imgui.h>
// tsd_rendering
#include <tsd/rendering/view/ManipulatorToAnari.hpp>

namespace tsd::ui::imgui {

MultiDeviceViewport::MultiDeviceViewport(
    Application *app, tsd::rendering::Manipulator *m, const char *name)
    : Window(app, name), m_arcball(m)
{}

MultiDeviceViewport::~MultiDeviceViewport()
{
  for (size_t i = 0; i < m_cameras.size(); i++) {
    auto *ri = getRenderIndex(i);
    auto d = ri->device();
    auto c = m_cameras[i];
    auto r = m_rud.renderers[i];

    anari::release(d, c);
    anari::release(d, r);
  }

  m_rud.renderers.clear();
  m_rud.devices.clear();
}

void MultiDeviceViewport::buildUI()
{
  ImVec2 _viewportSize = ImGui::GetContentRegionAvail();
  tsd::math::int2 viewportSize(_viewportSize.x, _viewportSize.y);

  if (m_viewportSize != viewportSize)
    reshape(viewportSize);

  m_pipeline.render();

  ImGui::Image((ImTextureID)m_outputPass->getTexture(),
      ImGui::GetContentRegionAvail(),
      ImVec2(0, 1),
      ImVec2(1, 0));

  updateCamera();
  ui_handleInput();
  ui_menubar();
}

void MultiDeviceViewport::setManipulator(tsd::rendering::Manipulator *m)
{
  m_arcball = m ? m : &m_localArcball;
}

void MultiDeviceViewport::resetView(bool resetAzEl)
{
  tsd::math::float3 bounds[2];
  getSceneBounds(bounds);
  auto center = 0.5f * (bounds[0] + bounds[1]);
  auto diag = bounds[1] - bounds[0];
  auto azel = resetAzEl ? tsd::math::float2(0.f, 20.f) : m_arcball->azel();
  m_arcball->setConfig(center, 1.25f * linalg::length(diag), azel);
  m_cameraToken = 0;
}

void MultiDeviceViewport::centerView()
{
  tsd::math::float3 bounds[2];
  getSceneBounds(bounds);
  m_arcball->setCenter(0.5f * (bounds[0] + bounds[1]));
  m_cameraToken = 0;
}

void MultiDeviceViewport::setLibrary(const std::string &libName)
{
  auto *core = appCore();
  auto &adm = core->anari;
  auto &scene = core->tsd.scene;

  auto library =
      anari::loadLibrary(libName.c_str(), tsd::app::anariStatusFunc, core);
  if (!library) {
    tsd::core::logError(
        "[multi-viewport] failed to load ANARI library '%s'", libName.c_str());
    return;
  }

  tsd::core::logStatus(
      "[multi-viewport] loaded ANARI library '%s'", libName.c_str());

  std::vector<anari::Device> devices;

  tsd::core::logStatus(
      "[multi-viewport] creating %zu devices...", scene.numberOfLayers());

  for (size_t i = 0; i < scene.numberOfLayers(); i++) {
    auto d = anari::newDevice(library, "default");
    devices.push_back(d);
  }

  tsd::core::logStatus("[multi-viewport] tethering devices...");

  if (devices.size() > 1) {
    for (size_t i = 0; i < devices.size(); i++) {
      auto d = devices[i];
      anari::setParameter(d, d, "dataGroupID", int(i));
      anari::setParameter(d, d, "tetherCount", int(devices.size()));
      anari::setParameter(d, d, "tetherIndex", int(i));
      if (i > 0)
        anari::setParameter(d, d, "tetherDevice", devices[0]);
      anari::commitParameters(d, d);
    }
  }

  tsd::core::logStatus("[multi-viewport] setting up render pipeline...");

  setupRenderPipeline(devices);

  tsd::core::logStatus("[multi-viewport] creating cameras and renderers...");

  for (size_t i = 0; i < scene.numberOfLayers(); i++) {
    auto *l = scene.layer(i);
    auto d = devices[i];
    auto c = anari::newObject<anari::Camera>(d, "perspective");
    auto r = anari::newObject<anari::Renderer>(d, "default");

    auto *ri = (tsd::rendering::RenderIndexAllLayers *)adm.acquireRenderIndex(
        scene, d);
    ri->setIncludedLayers({l});

    m_cameras.push_back(c);
    m_rud.devices.push_back(d);
    m_rud.renderers.push_back(r);

    m_anariPass->setCamera(i, c);
    m_anariPass->setRenderer(i, r);
    m_anariPass->setWorld(i, ri->world());
  }

  loadANARIRendererParameters();
  updateAllRendererParameters();

  static bool firstFrame = true;
  if (firstFrame && core->commandLine.loadedFromStateFile)
    firstFrame = false;

  if (firstFrame || m_arcball->distance() == tsd::math::inf) {
    resetView(true);
    if (core->view.poses.empty()) {
      tsd::core::logStatus("[multi-viewport] adding 'default' camera pose");
      core->addCurrentViewToCameraPoses("default");
    }
    firstFrame = false;
  }

  tsd::core::logStatus("[multi-viewport] ...viewport setup complete");

  anari::unloadLibrary(library);
}

void MultiDeviceViewport::loadSettings(tsd::core::DataNode &root)
{
  Window::loadSettings(root);

  // Viewport settings //

  root["fov"].getValue(ANARI_FLOAT32, &m_fov);
  root["resolutionScale"].getValue(ANARI_FLOAT32, &m_resolutionScale);
  root["showAxes"].getValue(ANARI_BOOL, &m_showAxes);

  // Camera //

  if (auto *c = root.child("camera"); c != nullptr) {
    tsd::math::float3 at(0.f);
    float distance = 0.f;
    tsd::math::float2 azel(0.f);
    int axis = 0;

    auto &camera = *c;
    camera["at"].getValue(ANARI_FLOAT32_VEC3, &at);
    camera["distance"].getValue(ANARI_FLOAT32, &distance);
    camera["azel"].getValue(ANARI_FLOAT32_VEC2, &azel);
    camera["up"].getValue(ANARI_INT32, &axis);

    camera["apertureRadius"].getValue(ANARI_FLOAT32, &m_apertureRadius);
    camera["focusDistance"].getValue(ANARI_FLOAT32, &m_focusDistance);

    m_arcball->setAxis(tsd::rendering::UpAxis(axis));
    m_arcball->setConfig(at, distance, azel);
  }
}

tsd::rendering::RenderIndexAllLayers *MultiDeviceViewport::getRenderIndex(
    size_t i) const
{
  auto &delegate = appCore()->anari.getUpdateDelegate();
  return (tsd::rendering::RenderIndexAllLayers *)delegate.get(i);
}

void MultiDeviceViewport::getSceneBounds(tsd::math::float3 boundsOut[2]) const
{
  tsd::math::float3 bounds[2] = {{-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}};

#if 0
  auto *ridx = getRenderIndex();
  auto d = ridx->device();
  auto w = ridx->world();

  if (!anariGetProperty(d,
          w,
          "bounds",
          ANARI_FLOAT32_BOX3,
          &bounds[0],
          sizeof(bounds),
          ANARI_WAIT)) {
    tsd::core::logWarning("[multi-viewport] ANARIWorld returned no bounds!");
  }
#endif

  std::memcpy(boundsOut, bounds, sizeof(tsd::math::float3) * 2);
}

void MultiDeviceViewport::setupRenderPipeline(
    const std::vector<anari::Device> &devices)
{
  m_pipeline.clear();

  m_anariPass =
      m_pipeline.emplace_back<tsd::rendering::MultiDeviceSceneRenderPass>(
          devices);

  {
    auto &adm = appCore()->anari;
    auto d = adm.loadDevice("helide");
    auto e = adm.loadDeviceExtensions("helide");

    m_axesPass =
        m_pipeline.emplace_back<tsd::rendering::AnariAxesRenderPass>(d, *e);
    m_axesPass->setEnabled(m_showAxes);

    anari::release(d, d);
  }

  m_outputPass = m_pipeline.emplace_back<tsd::rendering::CopyToSDLTexturePass>(
      m_app->sdlRenderer());

  reshape(m_viewportSize);
}

void MultiDeviceViewport::reshape(tsd::math::int2 newSize)
{
  if (newSize.x <= 0 || newSize.y <= 0)
    return;

  m_viewportSize = newSize;
  m_renderSize =
      tsd::math::int2(tsd::math::float2(m_viewportSize) * m_resolutionScale);

  m_pipeline.setDimensions(m_renderSize.x, m_renderSize.y);

  updateCamera(true);
}

void MultiDeviceViewport::updateCamera(bool force)
{
  if ((!force && !m_arcball->hasChanged(m_cameraToken)))
    return;

  for (size_t i = 0; i < m_cameras.size(); i++) {
    auto *ri = getRenderIndex(i);
    auto d = ri->device();
    auto c = m_cameras[i];

    tsd::rendering::updateCameraParametersPerspective(d, c, *m_arcball);
    anari::setParameter(
        d, c, "aspect", m_viewportSize.x / float(m_viewportSize.y));
    anari::setParameter(d, c, "apertureRadius", m_apertureRadius);
    anari::setParameter(d, c, "focusDistance", m_focusDistance);

    anari::setParameter(d, c, "fovy", anari::radians(m_fov));
    anari::commitParameters(d, c);
  }

  m_axesPass->setView(m_arcball->dir(), m_arcball->up());
}

void MultiDeviceViewport::loadANARIRendererParameters()
{
  auto d = m_rud.devices[0];
  for (auto &name : tsd::core::getANARIObjectSubtypes(d, ANARI_RENDERER)) {
    auto &o = m_rendererObject;
    o = tsd::core::parseANARIObjectInfo(d, ANARI_RENDERER, name.c_str());
    o.setName(name.c_str());
    o.setUpdateDelegate(&m_rud);
    break;
  }
}

void MultiDeviceViewport::updateAllRendererParameters()
{
  for (size_t i = 0; i < m_rud.devices.size(); i++) {
    auto &ro = m_rendererObject;
    auto d = m_rud.devices[i];
    auto r = m_rud.renderers[i];
    ro.updateAllANARIParameters(d, r);
    anari::commitParameters(d, r);
  }
}

void MultiDeviceViewport::ui_menubar()
{
  if (ImGui::BeginMenuBar()) {
    // Renderer //

    if (ImGui::BeginMenu("Renderer")) {
      ImGui::Text("Parameters:");
      ImGui::Indent(INDENT_AMOUNT);

      tsd::ui::buildUI_object(m_rendererObject, appCore()->tsd.scene, false);

      ImGui::Unindent(INDENT_AMOUNT);
      ImGui::Separator();
      ImGui::Separator();
      ImGui::Indent(INDENT_AMOUNT);

      if (ImGui::BeginMenu("reset to defaults?")) {
        if (ImGui::MenuItem("yes")) {
          loadANARIRendererParameters();
          updateAllRendererParameters();
        }
        ImGui::EndMenu();
      }

      ImGui::Unindent(INDENT_AMOUNT);
      ImGui::EndMenu();
    }

    // Camera //

    if (ImGui::BeginMenu("Camera")) {
      if (ImGui::Combo("up", &m_arcballUp, "+x\0+y\0+z\0-x\0-y\0-z\0\0")) {
        m_arcball->setAxis(static_cast<tsd::rendering::UpAxis>(m_arcballUp));
        resetView();
      }

      ImGui::Separator();

      ImGui::Text("Perspective Parameters:");

      ImGui::Indent(INDENT_AMOUNT);
      if (ImGui::SliderFloat("fov", &m_fov, 0.1f, 180.f))
        updateCamera(true);

      {
        ImGui::Text("Depth of Field:");
        ImGui::Indent(INDENT_AMOUNT);
        if (ImGui::DragFloat("aperture", &m_apertureRadius, 0.01f, 0.f, 1.f))
          updateCamera(true);

        if (ImGui::DragFloat(
                "focus distance", &m_focusDistance, 0.1f, 0.f, 1e20f))
          updateCamera(true);

        ImGui::Unindent(INDENT_AMOUNT);
      }
      ImGui::Unindent(INDENT_AMOUNT);

      ImGui::Separator();

      ImGui::Text("Reset View:");
      ImGui::Indent(INDENT_AMOUNT);
      if (ImGui::MenuItem("center"))
        centerView();
      if (ImGui::MenuItem("dist"))
        resetView(false);
      if (ImGui::MenuItem("angle + dist + center"))
        resetView(true);
      ImGui::Unindent(INDENT_AMOUNT);

      ImGui::EndMenu();
    }

    // Viewport //

    if (ImGui::BeginMenu("Viewport")) {
      {
        ImGui::Text("Format:");
        ImGui::Indent(INDENT_AMOUNT);
        const anari::DataType format = m_format;
        if (ImGui::RadioButton(
                "UFIXED8_RGBA_SRGB", m_format == ANARI_UFIXED8_RGBA_SRGB))
          m_format = ANARI_UFIXED8_RGBA_SRGB;
        if (ImGui::RadioButton("UFIXED8_VEC4", m_format == ANARI_UFIXED8_VEC4))
          m_format = ANARI_UFIXED8_VEC4;
        if (ImGui::RadioButton("FLOAT32_VEC4", m_format == ANARI_FLOAT32_VEC4))
          m_format = ANARI_FLOAT32_VEC4;

        if (format != m_format)
          m_anariPass->setColorFormat(m_format);
        ImGui::Unindent(INDENT_AMOUNT);
      }

      ImGui::Separator();

      {
        ImGui::Text("Render Resolution:");
        ImGui::Indent(INDENT_AMOUNT);

        const float current = m_resolutionScale;
        if (ImGui::RadioButton("100%", current == 1.f))
          m_resolutionScale = 1.f;
        if (ImGui::RadioButton("75%", current == 0.75f))
          m_resolutionScale = 0.75f;
        if (ImGui::RadioButton("50%", current == 0.5f))
          m_resolutionScale = 0.5f;
        if (ImGui::RadioButton("25%", current == 0.25f))
          m_resolutionScale = 0.25f;
        if (ImGui::RadioButton("12.5%", current == 0.125f))
          m_resolutionScale = 0.125f;

        if (current != m_resolutionScale)
          reshape(m_viewportSize);

        ImGui::Unindent(INDENT_AMOUNT);
      }

      ImGui::EndMenu();
    }

    ImGui::EndMenuBar();
  }
}

void MultiDeviceViewport::ui_handleInput()
{
  if (!ImGui::IsWindowFocused())
    return;

  ImGuiIO &io = ImGui::GetIO();

  const bool dolly = ImGui::IsMouseDown(ImGuiMouseButton_Right)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && ImGui::IsKeyDown(ImGuiKey_LeftShift));
  const bool pan = ImGui::IsMouseDown(ImGuiMouseButton_Middle)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && ImGui::IsKeyDown(ImGuiKey_LeftAlt));
  const bool orbit = ImGui::IsMouseDown(ImGuiMouseButton_Left);

  const bool anyMovement = dolly || pan || orbit;

  if (!anyMovement) {
    m_manipulating = false;
    m_previousMouse = tsd::math::float2(-1);
  } else if (ImGui::IsItemHovered() && !m_manipulating)
    m_manipulating = true;

  if (m_mouseRotating && !orbit)
    m_mouseRotating = false;

  if (m_manipulating) {
    tsd::math::float2 position;
    std::memcpy(&position, &io.MousePos, sizeof(position));

    const tsd::math::float2 mouse(position.x, position.y);

    if (anyMovement && m_previousMouse != tsd::math::float2(-1)) {
      const tsd::math::float2 prev = m_previousMouse;

      const tsd::math::float2 mouseFrom =
          prev * 2.f / tsd::math::float2(m_viewportSize);
      const tsd::math::float2 mouseTo =
          mouse * 2.f / tsd::math::float2(m_viewportSize);

      const tsd::math::float2 mouseDelta = mouseTo - mouseFrom;

      if (mouseDelta != tsd::math::float2(0.f)) {
        if (orbit && !(pan || dolly)) {
          if (!m_mouseRotating) {
            m_arcball->startNewRotation();
            m_mouseRotating = true;
          }

          m_arcball->rotate(mouseDelta);
        } else if (dolly)
          m_arcball->zoom(mouseDelta.y);
        else if (pan)
          m_arcball->pan(mouseDelta);
      }
    }

    m_previousMouse = mouse;
  }
}

int MultiDeviceViewport::windowFlags() const
{
  return ImGuiWindowFlags_MenuBar;
}

void MultiDeviceViewport::RendererUpdateDelegate::signalParameterUpdated(
    const tsd::core::Object *o, const tsd::core::Parameter *p)
{
  for (size_t i = 0; i < devices.size(); i++) {
    auto d = devices[i];
    auto r = renderers[i];
    o->updateANARIParameter(d, r, *p, p->name().c_str());
    anari::commitParameters(d, r);
  }
}

} // namespace tsd::ui::imgui
