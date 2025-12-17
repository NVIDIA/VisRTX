// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Viewport.h"
// tsd_ui_imgui
#include "imgui.h"
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/tsd_ui_imgui.h"
// tsd_core
#include "tsd/core/Logging.hpp"
#include "tsd/core/scene/objects/Camera.hpp"
// tsd_rendering
#include "tsd/rendering/view/ManipulatorToAnari.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"

// std
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

// stb
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace tsd::ui::imgui {

Viewport::Viewport(
    Application *app, tsd::rendering::Manipulator *m, const char *name)
    : Window(app, name)
{
  setManipulator(m);
  m_overlayWindowName = "overlay_";
  m_overlayWindowName += name;
  setLibrary("");
}

Viewport::~Viewport()
{
  teardownDevice();
}

void Viewport::buildUI()
{
  bool deviceReady = m_device != nullptr;
  if (deviceReady && !m_deviceReadyToUse)
    setupRenderPipeline();

  m_deviceReadyToUse = deviceReady;

  ImVec2 _viewportSize = ImGui::GetContentRegionAvail();
  tsd::math::int2 viewportSize(_viewportSize.x, _viewportSize.y);

  if (m_viewportSize != viewportSize)
    reshape(viewportSize);

  updateImage();
  updateCamera();

  ImGui::BeginDisabled(!m_deviceReadyToUse);

  if (m_outputPass) {
    ImGui::Image((ImTextureID)m_outputPass->getTexture(),
        ImGui::GetContentRegionAvail(),
        ImVec2(0, 1),
        ImVec2(1, 0));
  }

  if (m_showOverlay)
    ui_overlay();

  ImGui::EndDisabled();

  ui_gizmo();
  ui_handleInput();
  bool didPick = ui_picking(); // Needs to happen before ui_menubar
  ui_menubar();

  if (m_anariPass && !didPick)
    m_anariPass->setEnableIDs(appCore()->objectIsSelected());

  if (m_rIdx && (m_rIdx->isFlat() != appCore()->anari.useFlatRenderIndex())) {
    tsd::core::logWarning("instancing setting changed: resetting viewport");
    auto lib = m_libName;
    setLibrary(""); // clear old library
    setLibrary(lib);
  }
}

void Viewport::setManipulator(tsd::rendering::Manipulator *m)
{
  m_arcball = m ? m : &m_localArcball;
}

void Viewport::resetView(bool resetAzEl)
{
  if (!m_device)
    return;

  tsd::math::float3 bounds[2] = {{-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}};
  if (!anariGetProperty(m_device,
          m_rIdx->world(),
          "bounds",
          ANARI_FLOAT32_BOX3,
          &bounds[0],
          sizeof(bounds),
          ANARI_WAIT)) {
    tsd::core::logWarning("[viewport] ANARIWorld returned no bounds!");
  }

  auto center = 0.5f * (bounds[0] + bounds[1]);
  auto diag = bounds[1] - bounds[0];

  auto azel = resetAzEl ? tsd::math::float2(0.f, 20.f) : m_arcball->azel();
  m_arcball->setConfig(center, 1.25f * linalg::length(diag), azel);
  m_cameraToken = 0;
}

void Viewport::centerView()
{
  if (!m_device)
    return;

  tsd::math::float3 bounds[2] = {{-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}};
  if (!anariGetProperty(m_device,
          m_rIdx->world(),
          "bounds",
          ANARI_FLOAT32_BOX3,
          &bounds[0],
          sizeof(bounds),
          ANARI_WAIT)) {
    tsd::core::logWarning("[viewport] ANARIWorld returned no bounds!");
  }

  m_arcball->setCenter(0.5f * (bounds[0] + bounds[1]));
  m_cameraToken = 0;
}

void Viewport::setLibrary(const std::string &libName, bool doAsync)
{
  teardownDevice();

  if (!libName.empty() && libName != "{none}") {
    tsd::core::logStatus(
        "[viewport] *** setting viewport to use ANARI device '%s' ***",
        libName.c_str());
  }

  auto updateLibrary = [&, libName = libName]() {
    auto &adm = appCore()->anari;
    auto &scene = appCore()->tsd.scene;

    auto start = std::chrono::steady_clock::now();
    auto d = adm.loadDevice(libName);
    m_rud.d = d;
    m_libName = libName;

    m_frameSamples = 0;
    m_latestFL = 0.f;
    m_minFL = std::numeric_limits<float>::infinity();
    m_maxFL = -std::numeric_limits<float>::infinity();

    if (d) {
      if (auto *exts = adm.loadDeviceExtensions(libName); exts != nullptr)
        m_extensions = *exts;
      else
        m_extensions = {};

      tsd::core::logStatus("[viewport] getting renderer params...");

      m_currentRenderer = 0;
      loadANARIRendererParameters(d);
      updateAllRendererParameters(d);

      m_perspCamera = anari::newObject<anari::Camera>(d, "perspective");
      m_currentCamera = m_perspCamera;
      if (m_extensions.ANARI_KHR_CAMERA_ORTHOGRAPHIC)
        m_orthoCamera = anari::newObject<anari::Camera>(d, "orthographic");
      if (m_extensions.ANARI_KHR_CAMERA_OMNIDIRECTIONAL)
        m_omniCamera = anari::newObject<anari::Camera>(d, "omnidirectional");

      tsd::core::logStatus("[viewport] populating render index...");

      m_rIdx = adm.acquireRenderIndex(scene, d);
      setSelectionVisibilityFilterEnabled(m_showOnlySelected);

      tsd::core::logStatus("[viewport] getting scene bounds...");

      // NOTE(jda) - Setting the device on this viewport is what triggers active
      //             rendering in the UI thread, so this must be done here and
      //             no earlier. Also note that resetView() below will need this
      //             device also to be set.
      m_device = d;

      static bool firstFrame = true;
      if (firstFrame && appCore()->commandLine.loadedFromStateFile)
        firstFrame = false;

      if (firstFrame || m_arcball->distance() == tsd::math::inf) {
        resetView(true);
        if (appCore()->view.poses.empty()) {
          tsd::core::logStatus("[viewport] adding 'default' camera pose");
          appCore()->addCurrentViewToCameraPoses("default");
        }
        firstFrame = false;
      }

      tsd::core::logStatus("[viewport] ...device load complete");
    }

    auto end = std::chrono::steady_clock::now();
    m_timeToLoadDevice = std::chrono::duration<float>(end - start).count();

    if (m_deviceChangeCb)
      m_deviceChangeCb(m_libName);
  };

  if (doAsync)
    m_initFuture = std::async(updateLibrary);
  else
    updateLibrary();
}

void Viewport::setDeviceChangeCb(ViewportDeviceChangeCb cb)
{
  m_deviceChangeCb = std::move(cb);
}

void Viewport::setExternalInstances(
    const anari::Instance *instances, size_t count)
{
  if (m_rIdx)
    m_rIdx->setExternalInstances(instances, count);
}

void Viewport::setCustomFrameParameter(
    const char *name, const tsd::core::Any &value)
{
  if (!m_anariPass) {
    tsd::core::logWarning(
        "[viewport] cannot set custom frame parameter '%s': no frame yet",
        name);
    return;
  }

  auto d = m_anariPass->getDevice();
  auto f = m_anariPass->getFrame();
  anari::setParameter(d, f, name, value.type(), value.data());
  anari::commitParameters(d, f);
}

void Viewport::saveSettings(tsd::core::DataNode &root)
{
  root.reset(); // clear all previous values, if they exist

  // Viewport settings //

  root["echoCameraConfig"] = m_echoCameraConfig;
  root["showOverlay"] = m_showOverlay;
  root["showCameraInfo"] = m_showCameraInfo;
  root["showOnlySelected"] = m_showOnlySelected;
  root["highlightSelection"] = m_highlightSelection;
  root["showOnlySelected"] = m_showOnlySelected;
  root["visualizeAOV"] = static_cast<int>(m_visualizeAOV);
  root["depthVisualMinimum"] = m_depthVisualMinimum;
  root["depthVisualMaximum"] = m_depthVisualMaximum;
  root["fov"] = m_fov;
  root["resolutionScale"] = m_resolutionScale;
  root["showAxes"] = m_showAxes;

  // Gizmo settings //

  root["enableGizmo"] = m_enableGizmo;
  root["gizmoOperation"] = static_cast<int>(m_gizmoOperation);
  root["gizmoMode"] = static_cast<int>(m_gizmoMode);

  root["anariLibrary"] = m_libName;

  // Camera //

  auto &camera = root["camera"];
  camera["at"] = m_arcball->at();
  camera["distance"] = m_arcball->distance();
  camera["azel"] = m_arcball->azel();
  camera["up"] = int(m_arcball->axis());
  camera["apertureRadius"] = m_apertureRadius;
  camera["focusDistance"] = m_focusDistance;

  // Database Camera //

  if (m_selectedCamera) {
    root["selectedCamera"] = static_cast<uint64_t>(m_selectedCamera.index());
  }

  // Renderer settings //

  auto &renderers = root["renderers"];
  for (auto &ro : m_rendererObjects)
    tsd::io::objectToNode(ro, renderers[ro.name()]);

  // Base window settings //

  Window::saveSettings(root);
}

void Viewport::loadSettings(tsd::core::DataNode &root)
{
  Window::loadSettings(root);

  // Viewport settings //

  root["echoCameraConfig"].getValue(ANARI_BOOL, &m_echoCameraConfig);
  root["showOverlay"].getValue(ANARI_BOOL, &m_showOverlay);
  root["showCameraInfo"].getValue(ANARI_BOOL, &m_showCameraInfo);
  root["showOnlySelected"].getValue(ANARI_BOOL, &m_showOnlySelected);
  root["highlightSelection"].getValue(ANARI_BOOL, &m_highlightSelection);
  root["showOnlySelected"].getValue(ANARI_BOOL, &m_showOnlySelected);
  int aovType = static_cast<int>(m_visualizeAOV);
  root["visualizeAOV"].getValue(ANARI_INT32, &aovType);
  m_visualizeAOV = static_cast<tsd::rendering::AOVType>(aovType);
  root["depthVisualMinimum"].getValue(ANARI_FLOAT32, &m_depthVisualMinimum);
  root["depthVisualMaximum"].getValue(ANARI_FLOAT32, &m_depthVisualMaximum);
  root["fov"].getValue(ANARI_FLOAT32, &m_fov);
  root["resolutionScale"].getValue(ANARI_FLOAT32, &m_resolutionScale);
  root["showAxes"].getValue(ANARI_BOOL, &m_showAxes);

  // Gizmo settings //

  root["enableGizmo"].getValue(ANARI_BOOL, &m_enableGizmo);
  int gizmoOp = static_cast<int>(m_gizmoOperation);
  root["gizmoOperation"].getValue(ANARI_INT32, &gizmoOp);
  m_gizmoOperation = static_cast<ImGuizmo::OPERATION>(gizmoOp);
  int gizmoMode = static_cast<int>(m_gizmoMode);
  root["gizmoMode"].getValue(ANARI_INT32, &gizmoMode);
  m_gizmoMode = static_cast<ImGuizmo::MODE>(gizmoMode);

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

  // Database Camera //

  if (auto *c = root.child("selectedCamera"); c) {
    uint64_t idx = 0;
    c->getValue(ANARI_UINT64, &idx);
    m_selectedCamera = appCore()->tsd.scene.getObject<tsd::core::Camera>(idx);
  }

  // Setup library //

  auto *core = appCore();
  if (core->commandLine.useDefaultRenderer) {
    std::string libraryName;
    root["anariLibrary"].getValue(ANARI_STRING, &libraryName);
    setLibrary(libraryName);
  }

  // Renderer settings //

  root["renderers"].foreach_child([&](auto &node) {
    for (auto &ro : m_rendererObjects) {
      if (ro.subtype() == node.name()) {
        tsd::io::nodeToObject(node, ro);
        return;
      }
    }
  });

  updateAllRendererParameters(m_device);
}

void Viewport::loadANARIRendererParameters(anari::Device d)
{
  m_rendererObjects.clear();
  for (auto &r : m_renderers)
    anari::release(d, r);
  m_renderers.clear();

  for (auto &name : tsd::core::getANARIObjectSubtypes(d, ANARI_RENDERER)) {
    auto ar = anari::newObject<anari::Renderer>(d, name.c_str());
    auto o = tsd::core::parseANARIObjectInfo(d, ANARI_RENDERER, name.c_str());
    o.setName(name.c_str());
    o.setUpdateDelegate(&m_rud);
    m_rendererObjects.push_back(std::move(o));
    m_renderers.push_back(ar);
  }
}

void Viewport::updateAllRendererParameters(anari::Device d)
{
  for (size_t i = 0; i < m_rendererObjects.size(); i++) {
    auto &ro = m_rendererObjects[i];
    auto ar = m_renderers[i];
    ro.updateAllANARIParameters(d, ar);
    anari::commitParameters(d, ar);
  }
}

void Viewport::setupRenderPipeline()
{
  tsd::core::logStatus("[viewport] initialized scene for '%s' device in %.2fs",
      m_libName.c_str(),
      m_timeToLoadDevice);

  m_anariPass =
      m_pipeline.emplace_back<tsd::rendering::AnariSceneRenderPass>(m_device);

  m_saveToFilePass = m_pipeline.emplace_back<tsd::rendering::SaveToFilePass>();
  m_saveToFilePass->setEnabled(false);
  m_saveToFilePass->setSingleShotMode(true);

  m_pickPass = m_pipeline.emplace_back<tsd::rendering::PickPass>();
  m_pickPass->setEnabled(false);
  m_pickPass->setPickOperation([&](tsd::rendering::RenderBuffers &b) {
    // Get depth //

    auto [width, height] = m_pickPass->getDimensions();

    auto l = linalg::clamp(m_pickCoord,
        tsd::math::int2(0, 0),
        tsd::math::int2(width - 1, height - 1));
    l.x = width - l.x;
    l.y = height - l.y;
    const auto i = l.y * width + l.x;

    m_pickedDepth = b.depth ? b.depth[i] : 1e30f;

    if (!m_selectObjectNextPick) {
      // Do object selection //
      auto mPos = ImGui::GetMousePos();
      auto wMin = ImGui::GetItemRectMin();
      auto pixel = m_pickCoord;
      pixel.x = int(mPos[0] - wMin[0]);
      pixel.y = m_viewportSize.y - int(mPos[1] - wMin[1]);

      const float aspect = m_viewportSize.x / float(m_viewportSize.y);
      anari::math::float2 imgPlaneSize;
      imgPlaneSize.y = 2.f * tanf(0.5f * anari::radians(m_fov));
      imgPlaneSize.x = imgPlaneSize.y * aspect;

      const auto d = m_arcball->dir();
      const auto u = m_arcball->up();

      const auto dir_du =
          anari::math::normalize(anari::math::cross(d, u)) * imgPlaneSize.x;
      const auto dir_dv = anari::math::normalize(anari::math::cross(dir_du, d))
          * imgPlaneSize.y;
      const auto dir_00 = d - .5f * dir_du - .5f * dir_dv;

      const auto screen = anari::math::float2(
          1.f / m_viewportSize.x * pixel.x, (1.f / m_viewportSize.y * pixel.y));

      const auto dir = anari::math::normalize(
          dir_00 + screen.x * dir_du + screen.y * dir_dv);

      const auto p = m_arcball->eye();
      const auto c = p + m_pickedDepth * dir;

      tsd::core::logStatus(
          "[viewport] pick [%i, %i] {%f, %f} depth %f / %f| {%f, %f, %f}",
          int(pixel.x),
          int(pixel.y),
          screen.x,
          screen.y,
          m_pickedDepth,
          m_arcball->distance(),
          c.x,
          c.y,
          c.z);

      m_arcball->setCenter(c);
    } else {
      // Do object selection //

      uint32_t id = b.objectId ? b.objectId[i] : ~0u;
      if (id != ~0u) {
        tsd::core::logStatus("[viewport] picked object '%u' @ (%i, %i) | z: %f",
            id,
            l.x,
            l.y,
            m_pickedDepth);
      }

      anari::DataType objectType = ANARI_SURFACE;
      if (id != ~0u && id & 0x80000000u) {
        objectType = ANARI_VOLUME;
        id &= 0x7FFFFFFF;
      }

      appCore()->setSelectedObject(
          id == ~0u ? nullptr : appCore()->tsd.scene.getObject(objectType, id));
    }

    m_pickPass->setEnabled(false);
  });

  m_visualizeAOVPass =
      m_pipeline.emplace_back<tsd::rendering::VisualizeAOVPass>();
  m_visualizeAOVPass->setEnabled(false);

  m_outlinePass = m_pipeline.emplace_back<tsd::rendering::OutlineRenderPass>();

  m_axesPass = m_pipeline.emplace_back<tsd::rendering::AnariAxesRenderPass>(
      m_device, m_extensions);
  m_axesPass->setEnabled(m_showAxes);

  m_outputPass = m_pipeline.emplace_back<tsd::rendering::CopyToSDLTexturePass>(
      m_app->sdlRenderer());

  reshape(m_viewportSize);
}

void Viewport::teardownDevice()
{
  if (m_initFuture.valid())
    m_initFuture.get();

  if (!m_deviceReadyToUse)
    return;

  m_pipeline.clear();
  m_anariPass = nullptr;
  m_outlinePass = nullptr;
  m_outputPass = nullptr;
  m_saveToFilePass = nullptr;

  appCore()->anari.releaseRenderIndex(m_device);
  m_rIdx = nullptr;
  m_libName.clear();

  anari::release(m_device, m_perspCamera);
  anari::release(m_device, m_orthoCamera);
  anari::release(m_device, m_omniCamera);
  for (auto &r : m_renderers)
    anari::release(m_device, r);
  anari::release(m_device, m_device);

  m_perspCamera = nullptr;
  m_orthoCamera = nullptr;
  m_omniCamera = nullptr;
  m_renderers.clear();
  m_rendererObjects.clear();
  m_device = nullptr;

  m_deviceReadyToUse = false;
}

void Viewport::reshape(tsd::math::int2 newSize)
{
  if (newSize.x <= 0 || newSize.y <= 0)
    return;

  m_viewportSize = newSize;
  m_renderSize =
      tsd::math::int2(tsd::math::float2(m_viewportSize) * m_resolutionScale);

  m_pipeline.setDimensions(m_renderSize.x, m_renderSize.y);

  updateFrame();
  updateCamera(true);
}

void Viewport::pick(tsd::math::int2 l, bool selectObject)
{
  m_selectObjectNextPick = selectObject;
  m_pickCoord = l;
  m_pickPass->setEnabled(true);
  m_anariPass->setEnableIDs(true);
}

void Viewport::setSelectionVisibilityFilterEnabled(bool enabled)
{
  if (!enabled)
    m_rIdx->setFilterFunction({});
  else {
    m_rIdx->setFilterFunction([&](const tsd::core::Object *obj) {
      return !appCore()->tsd.selectedObject
          || obj == appCore()->tsd.selectedObject;
    });
  }
}

void Viewport::updateFrame()
{
  if (!m_anariPass)
    return;

  m_rud.r = m_renderers[m_currentRenderer];
  m_anariPass->setCamera(m_currentCamera);
  m_anariPass->setRenderer(m_rud.r);
  m_anariPass->setWorld(m_rIdx->world());
}

void Viewport::updateCamera(bool force)
{
  if (!m_anariPass)
    return;

  // Check if camera changed, might it be database camera or manipulator one.
  bool isDbCamera = m_selectedCamera && m_cameraDelegate;

  // Before proceeding, check if the camera still does exist
  if (isDbCamera && !m_selectedCamera->self()) {
    tsd::core::logWarning(
        "[viewport] selected camera no longer exists, reverting to manipulator camera");
    clearDatabaseCamera();
    isDbCamera = false;
  }

  if (!force
      && !(isDbCamera ? m_cameraDelegate->hasChanged(m_cameraToken)
                      : m_arcball->hasChanged(m_cameraToken)))
    return;

  // Get compass information
  tsd::math::float3 axesDir;
  tsd::math::float3 axesUp;
  if (isDbCamera) {
    applyCameraParameters(&*m_selectedCamera);
    axesDir = m_selectedCamera->parameterValueAs<tsd::math::float3>("direction")
                  .value_or(tsd::math::float3(0.0f, 0.0f, -1.0f));
    axesUp =
        m_selectedCamera->parameterValueAs<tsd::math::float3>("up").value_or(
            tsd::math::float3(0.0f, 1.0f, 0.0f));
  } else {
    // perspective camera //
    tsd::rendering::updateCameraParametersPerspective(
        m_device, m_perspCamera, *m_arcball);
    anari::setParameter(m_device,
        m_perspCamera,
        "aspect",
        m_viewportSize.x / float(m_viewportSize.y));
    anari::setParameter(
        m_device, m_perspCamera, "apertureRadius", m_apertureRadius);
    anari::setParameter(
        m_device, m_perspCamera, "focusDistance", m_focusDistance);

    anari::setParameter(m_device, m_perspCamera, "fovy", anari::radians(m_fov));
    anari::commitParameters(m_device, m_perspCamera);

    // orthographic camera //

    if (m_orthoCamera) {
      tsd::rendering::updateCameraParametersOrthographic(
          m_device, m_orthoCamera, *m_arcball);
      anari::setParameter(m_device,
          m_orthoCamera,
          "aspect",
          m_viewportSize.x / float(m_viewportSize.y));
      anari::commitParameters(m_device, m_orthoCamera);
    }

    // omnidirectional camera //

    if (m_omniCamera) {
      tsd::rendering::updateCameraParametersPerspective( // also works for omni
          m_device,
          m_omniCamera,
          *m_arcball);
      anari::commitParameters(m_device, m_omniCamera);
    }
    if (m_echoCameraConfig)
      echoCameraConfig();
    axesUp = m_arcball->up();
    axesDir = m_arcball->dir();
  }

  m_axesPass->setView(axesDir, axesUp);
}

void Viewport::updateImage()
{
  if (!m_deviceReadyToUse)
    return;

  auto frame = m_anariPass->getFrame();
  anari::getProperty(
      m_device, frame, "numSamples", m_frameSamples, ANARI_NO_WAIT);

  const auto &tsd_ctx = appCore()->tsd;
  const auto *selectedObject = tsd_ctx.selectedObject;
  const bool doHighlight = !m_showOnlySelected && m_highlightSelection
      && selectedObject
      && (selectedObject->type() == ANARI_SURFACE
          || selectedObject->type() == ANARI_VOLUME);
  auto id = uint32_t(~0u);
  if (doHighlight) {
    id = selectedObject->index();
    if (selectedObject->type() == ANARI_VOLUME)
      id |= 0x80000000u;
  }
  m_outlinePass->setOutlineId(id);

  auto start = std::chrono::steady_clock::now();
  m_pipeline.render();
  auto end = std::chrono::steady_clock::now();
  m_latestFL = std::chrono::duration<float>(end - start).count() * 1000;

  float duration = 0.f;
  anari::getProperty(m_device, frame, "duration", duration, ANARI_NO_WAIT);

  m_latestAnariFL = duration * 1000;
  m_minFL = std::min(m_minFL, m_latestAnariFL);
  m_maxFL = std::max(m_maxFL, m_latestAnariFL);
}

void Viewport::applyCameraParameters(tsd::core::Camera *cam)
{
  if (!cam || !m_device || !m_currentCamera)
    return;

  auto d = m_device;
  auto c = m_currentCamera; // ANARI camera
  anari::setParameter(
      d, m_currentCamera, "aspect", m_viewportSize.x / float(m_viewportSize.y));
  cam->updateAllANARIParameters(d, m_currentCamera);
  anari::commitParameters(d, c);
}

void Viewport::setDatabaseCamera(tsd::core::CameraRef cam)
{
  // Detach previous delegate if any
  if (m_cameraDelegate)
    m_cameraDelegate->detach();
  m_cameraDelegate.reset();

  m_selectedCamera = cam;
  m_cameraToken = 0;
  // Wire new delegate
  if (m_selectedCamera) {
    m_cameraDelegate = std::make_unique<tsd::core::CameraUpdateDelegate>(
        m_selectedCamera.data());
  }
  updateCamera(true);
  tsd::core::logStatus(
      "Viewport using database camera '%s'", cam->name().c_str());
}

void Viewport::clearDatabaseCamera()
{
  // Detach delegate if any
  if (m_cameraDelegate)
    m_cameraDelegate->detach();
  m_cameraDelegate.reset();
  m_selectedCamera = {};
  m_cameraToken = 0;
  updateCamera(true);
  tsd::core::logStatus("Viewport using manipulator");
}

void Viewport::createCameraFromCurrentView()
{
  auto &scene = appCore()->tsd.scene;

  tsd::core::CameraRef cam;

  if (m_selectedCamera) {
    // If a database camera is selected, copy it
    auto sourceCam = m_selectedCamera;

    // Create new camera with same subtype
    cam = scene.createObject<tsd::core::Camera>(sourceCam->subtype());

    // Copy all parameters
    for (size_t i = 0; i < sourceCam->numParameters(); i++) {
      const auto &srcParam = sourceCam->parameterAt(i);
      const char *paramName = sourceCam->parameterNameAt(i);
      cam->parameter(paramName)->setValue(srcParam.value());
    }
  } else {
    // No database camera selected, create from manipulator state

    // Determine camera type from current ANARI camera
    tsd::core::Token subtype = tsd::core::tokens::camera::perspective;
    if (m_currentCamera == m_orthoCamera)
      subtype = tsd::core::tokens::camera::orthographic;
    else if (m_currentCamera == m_omniCamera)
      subtype = tsd::core::tokens::camera::omnidirectional;

    // Create camera object
    cam = scene.createObject<tsd::core::Camera>(subtype);

    // Set parameters from manipulator
    auto eye = m_arcball->eye();
    auto dir = tsd::math::normalize(m_arcball->at() - eye);
    auto up = m_arcball->up();

    cam->parameter("position")->setValue(eye);
    cam->parameter("direction")->setValue(dir);
    cam->parameter("up")->setValue(up);

    // Set type-specific params
    if (subtype == tsd::core::tokens::camera::perspective) {
      cam->parameter("fovy")->setValue(tsd::math::radians(m_fov));
      cam->parameter("apertureRadius")->setValue(m_apertureRadius);
      cam->parameter("focusDistance")->setValue(m_focusDistance);
    } else if (subtype == tsd::core::tokens::camera::orthographic) {
      // Nothing to set here
    }
  }

  // Set name
  std::string name = "ViewCamera_" + std::to_string(cam.index());
  cam->setName(name.c_str());

  // Auto-select it
  setDatabaseCamera(cam);

  tsd::core::logStatus("Created camera '%s' from current view", name.c_str());
}

void Viewport::echoCameraConfig()
{
  const auto p = m_arcball->eye();
  const auto d = m_arcball->dir();
  const auto u = m_arcball->up();

  tsd::core::logStatus("Camera:");
  tsd::core::logStatus("  pos: %f, %f, %f", p.x, p.y, p.z);
  tsd::core::logStatus("  dir: %f, %f, %f", d.x, d.y, d.z);
  tsd::core::logStatus("   up: %f, %f, %f", u.x, u.y, u.z);
}

void Viewport::ui_menubar()
{
  if (ImGui::BeginMenuBar()) {
    // Device //

    if (ImGui::BeginMenu("Device")) {
      for (auto &libName : appCore()->commandLine.libraryList) {
        const bool isThisLibrary = m_libName == libName;
        if (ImGui::RadioButton(libName.c_str(), isThisLibrary))
          setLibrary(libName);
      }
      ImGui::Separator();
      if (ImGui::MenuItem("reload device")) {
        auto lib = m_libName; // setLibrary() clears m_libName
        setLibrary(lib);
      }
      ImGui::EndMenu();
    }

    ImGui::BeginDisabled(!m_device);

    // Renderer //

    if (ImGui::BeginMenu("Renderer")) {
      if (m_rendererObjects.size() > 1) {
        ImGui::Text("Subtype:");
        ImGui::Indent(INDENT_AMOUNT);
        for (int i = 0; i < m_rendererObjects.size(); i++) {
          const char *rName = m_rendererObjects[i].name().c_str();
          if (ImGui::RadioButton(rName, &m_currentRenderer, i))
            updateFrame();
        }
        ImGui::Unindent(INDENT_AMOUNT);
      }

      ImGui::Separator();

      if (!m_rendererObjects.empty()) {
        ImGui::Text("Parameters:");
        ImGui::Indent(INDENT_AMOUNT);

        tsd::ui::buildUI_object(
            m_rendererObjects[m_currentRenderer], appCore()->tsd.scene, true);

        ImGui::Unindent(INDENT_AMOUNT);
        ImGui::Separator();
        ImGui::Separator();
        ImGui::Indent(INDENT_AMOUNT);

        if (ImGui::BeginMenu("reset to defaults?")) {
          if (ImGui::MenuItem("yes")) {
            loadANARIRendererParameters(m_device);
            updateAllRendererParameters(m_device);
            updateFrame();
          }
          ImGui::EndMenu();
        }

        ImGui::Unindent(INDENT_AMOUNT);
      }
      ImGui::EndMenu();
    }

    // Camera //

    if (ImGui::BeginMenu("Camera")) {
      {
        ImGui::Text("Subtype:");
        ImGui::Indent(INDENT_AMOUNT);

        bool changeType = false;
        if (ImGui::RadioButton(
                "perspective", m_currentCamera == m_perspCamera)) {
          m_currentCamera = m_perspCamera;
          changeType = true;
        }

        ImGui::BeginDisabled(!m_orthoCamera);
        if (ImGui::RadioButton("orthographic",
                m_orthoCamera && m_currentCamera == m_orthoCamera)) {
          m_currentCamera = m_orthoCamera;
          changeType = true;
        }
        ImGui::EndDisabled();

        ImGui::BeginDisabled(!m_omniCamera);
        if (ImGui::RadioButton("omnidirectional",
                m_omniCamera && m_currentCamera == m_omniCamera)) {
          m_currentCamera = m_omniCamera;
          changeType = true;
        }
        ImGui::EndDisabled();

        if (changeType)
          updateFrame();

        ImGui::Unindent(INDENT_AMOUNT);
      }

      if (ImGui::Combo("up", &m_arcballUp, "+x\0+y\0+z\0-x\0-y\0-z\0\0")) {
        m_arcball->setAxis(static_cast<tsd::rendering::UpAxis>(m_arcballUp));
        resetView();
      }

      ImGui::Separator();

      ImGui::BeginDisabled(m_currentCamera != m_perspCamera);

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
      ImGui::EndDisabled();

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

      ImGui::Separator();

      if (ImGui::Checkbox("echo config", &m_echoCameraConfig)
          && m_echoCameraConfig)
        echoCameraConfig();

      ImGui::Separator();

      // Database Camera Selection
      ImGui::Text("Database Camera:");
      ImGui::Indent(INDENT_AMOUNT);

      // Build camera list
      std::vector<std::string> cameraNames = {"<Manipulator>"};
      m_menuCameraRefs.resize(1);
      m_menuCameraRefs[0] = {};
      int currentSelection = 0;

      const auto &cameraDB = appCore()->tsd.scene.objectDB().camera;
      tsd::core::foreach_item_const(cameraDB, [&](const auto *cam) {
        if (cam) {
          cameraNames.push_back(cam->name());
          m_menuCameraRefs.push_back(cam->self());
          if (m_selectedCamera == cam->self()) {
            currentSelection = static_cast<int>(cameraNames.size() - 1);
          }
        }
      });

      if (ImGui::Combo(
              "Select",
              &currentSelection,
              [](void *data, int idx, const char **out) {
                auto *names = (std::vector<std::string> *)data;
                *out = (*names)[idx].c_str();
                return true;
              },
              &cameraNames,
              static_cast<int>(cameraNames.size()))) {
        if (currentSelection == 0) {
          clearDatabaseCamera();
        } else {
          setDatabaseCamera(m_menuCameraRefs[currentSelection]);
        }
      }

      if (ImGui::Button("Create from Current View")) {
        createCameraFromCurrentView();
      }

      ImGui::Unindent(INDENT_AMOUNT);

      ImGui::EndMenu();
    }

    // Gizmo //

    if (ImGui::BeginMenu("Transform Manipulator")) {
      ImGui::Checkbox("Enable Manipulator", &m_enableGizmo);

      ImGui::Separator();
      ImGui::Text("Operation:");
      ImGui::Indent(INDENT_AMOUNT);
      const auto &gOp = m_gizmoOperation;
      if (ImGui::RadioButton("(w) Translate", gOp == ImGuizmo::TRANSLATE))
        m_gizmoOperation = ImGuizmo::TRANSLATE;
      if (ImGui::RadioButton("(e) Scale", gOp == ImGuizmo::SCALE))
        m_gizmoOperation = ImGuizmo::SCALE;
      if (ImGui::RadioButton("(r) Rotate", gOp == ImGuizmo::ROTATE))
        m_gizmoOperation = ImGuizmo::ROTATE;
      ImGui::Unindent(INDENT_AMOUNT);

      ImGui::Separator();
      ImGui::Text("Mode:");
      ImGui::Indent(INDENT_AMOUNT);
      if (ImGui::RadioButton("Local", m_gizmoMode == ImGuizmo::LOCAL))
        m_gizmoMode = ImGuizmo::LOCAL;
      if (ImGui::RadioButton("World", m_gizmoMode == ImGuizmo::WORLD))
        m_gizmoMode = ImGuizmo::WORLD;
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

      ImGui::Separator();

      const char *aovItems[] = {"default", "depth", "albedo", "normal"};
      if (int aov = int(m_visualizeAOV); ImGui::Combo(
              "visualize AOV", &aov, aovItems, IM_ARRAYSIZE(aovItems))) {
        if (aov != int(m_visualizeAOV)) {
          m_visualizeAOV = static_cast<tsd::rendering::AOVType>(aov);
          m_visualizeAOVPass->setAOVType(m_visualizeAOV);
          m_anariPass->setEnableAlbedo(
              m_visualizeAOV == tsd::rendering::AOVType::ALBEDO);
          m_anariPass->setEnableNormals(
              m_visualizeAOV == tsd::rendering::AOVType::NORMAL);
        }
      }

      ImGui::BeginDisabled(m_visualizeAOV != tsd::rendering::AOVType::DEPTH);
      bool depthRangeChanged = false;
      depthRangeChanged |= ImGui::DragFloat("depth minimum",
          &m_depthVisualMinimum,
          0.1f,
          0.f,
          m_depthVisualMaximum);
      depthRangeChanged |= ImGui::DragFloat("depth maximum",
          &m_depthVisualMaximum,
          0.1f,
          m_depthVisualMinimum,
          1e20f);
      if (depthRangeChanged)
        m_visualizeAOVPass->setDepthRange(
            m_depthVisualMinimum, m_depthVisualMaximum);
      ImGui::EndDisabled();

      ImGui::Separator();

      ImGui::BeginDisabled(m_showOnlySelected);
      ImGui::Checkbox("highlight selected", &m_highlightSelection);
      ImGui::EndDisabled();

      if (ImGui::Checkbox("only show selected", &m_showOnlySelected))
        setSelectionVisibilityFilterEnabled(m_showOnlySelected);

      ImGui::Separator();

      if (ImGui::Checkbox("show axes", &m_showAxes))
        m_axesPass->setEnabled(m_showAxes);

      ImGui::Separator();

      ImGui::Checkbox("show info overlay", &m_showOverlay);
      if (ImGui::MenuItem("reset stats")) {
        m_minFL = m_latestFL;
        m_maxFL = m_latestFL;
      }

      ImGui::Separator();

      if (ImGui::MenuItem("take screenshot")) {
        // Generate timestamped filename
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch())
            % 1000;

        std::stringstream ss;
        ss << "screenshot_"
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << "_"
           << std::setfill('0') << std::setw(3) << ms.count() << ".png";

        // Ensure the screenshot is saved in the current working directory
        std::filesystem::path workingDir = std::filesystem::current_path();
        std::filesystem::path filename = workingDir / ss.str();

        m_saveToFilePass->setFilename(filename.string());
        m_saveToFilePass->setEnabled(true);
      }

      ImGui::EndMenu();
    }

    // World //

    if (ImGui::BeginMenu("World")) {
      if (ImGui::MenuItem("print bounds")) {
        tsd::math::float3 bounds[2];

        anariGetProperty(m_device,
            m_rIdx->world(),
            "bounds",
            ANARI_FLOAT32_BOX3,
            &bounds[0],
            sizeof(bounds),
            ANARI_WAIT);

        tsd::core::logStatus(
            "[viewport] current world bounds {%f, %f, %f} x {%f, %f, %f}",
            bounds[0].x,
            bounds[0].y,
            bounds[0].z,
            bounds[1].x,
            bounds[1].y,
            bounds[1].z);
      }

      ImGui::EndMenu();
    }

    ImGui::EndDisabled();

    ImGui::EndMenuBar();
  }
}

void Viewport::ui_handleInput()
{
  // No device
  if (!m_deviceReadyToUse)
    return;

  // Handle gizmo keyboard shortcuts. Handle those before checking for
  // window focus so they can act globally.
  // When a new manipulator mode is selected, we default to world mode.
  // Otherwise, toggle between local and global modes.
  if (ImGui::IsKeyPressed(ImGuiKey_Q, false)
      || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    m_enableGizmo = false;
  } else if (ImGui::IsKeyPressed(ImGuiKey_W, false)) {
    if (m_enableGizmo && m_gizmoOperation == ImGuizmo::TRANSLATE) {
      m_gizmoMode =
          (m_gizmoMode == ImGuizmo::WORLD) ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
    } else {
      m_enableGizmo = true;
      m_gizmoOperation = ImGuizmo::TRANSLATE;
      m_gizmoMode = ImGuizmo::WORLD;
    }
  } else if (ImGui::IsKeyPressed(ImGuiKey_E, false)) {
    if (m_enableGizmo && m_gizmoOperation == ImGuizmo::SCALE) {
      m_gizmoMode =
          (m_gizmoMode == ImGuizmo::WORLD) ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
    } else {
      m_enableGizmo = true;
      m_gizmoOperation = ImGuizmo::SCALE;
      m_gizmoMode = ImGuizmo::WORLD;
    }
  } else if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
    if (m_enableGizmo && m_gizmoOperation == ImGuizmo::ROTATE) {
      m_gizmoMode =
          (m_gizmoMode == ImGuizmo::WORLD) ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
    } else {
      m_enableGizmo = true;
      m_gizmoOperation = ImGuizmo::ROTATE;
      m_gizmoMode = ImGuizmo::WORLD;
    }
  }

  // Enforce global Gizmo state so that it actually stops tracking
  // interactions when disabled.
  ImGuizmo::Enable(m_enableGizmo);

  // Block arcball input and picking when ImGuizmo is being used
  if (ImGuizmo::IsUsing())
    return;

  // Do not bother with events if the window is not hovered
  // or no interaction is ongoing.
  // We'll use that hovering status to check for starting an
  // event below.
  if (!ImGui::IsWindowHovered() && !m_manipulating)
    return;

  // Block arcball input when a database camera is selected
  if (m_selectedCamera)
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
  } else if (ImGui::IsItemHovered() && !m_manipulating) {
    m_manipulating = true;
    ImGui::SetWindowFocus(); // ensure we keep focus while manipulating
  }

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

bool Viewport::ui_picking()
{
  const ImGuiIO &io = ImGui::GetIO();

  // Pick view center //

  const bool shouldPickCenter = m_currentCamera == m_perspCamera
      && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)
      && ImGui::IsKeyDown(ImGuiKey_LeftShift);
  if (shouldPickCenter && ImGui::IsWindowHovered()) {
    auto mPos = ImGui::GetMousePos();
    auto wMin = ImGui::GetItemRectMin();
    auto pixel = tsd::math::int2(
        tsd::math::float2(
            m_viewportSize.x - (mPos[0] - wMin[0]), mPos[1] - wMin[1])
        * m_resolutionScale);
    pick(pixel, false);
    return true;
  }

  // Pick object //

  const bool shouldPickObject =
      ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
  if (shouldPickObject && ImGui::IsWindowHovered()) {
    auto mPos = ImGui::GetMousePos();
    auto wMin = ImGui::GetItemRectMin();
    auto pixel = tsd::math::float2(
                     m_viewportSize.x - (mPos[0] - wMin[0]), mPos[1] - wMin[1])
        * m_resolutionScale;
    pick(tsd::math::int2(pixel), true);
    return true;
  }

  return false;
}

void Viewport::ui_overlay()
{
  ImGuiIO &io = ImGui::GetIO();
  ImVec2 windowPos = ImGui::GetWindowPos();
  windowPos.x += 10;
  windowPos.y += 63 * io.FontGlobalScale;

  ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration
      | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize
      | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
      | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;

  ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always);

  if (ImGui::Begin(m_overlayWindowName.c_str(), nullptr, window_flags)) {
    ImGui::Text("  device: %s", m_libName.c_str());

    // Camera indicator
    if (m_selectedCamera) {
      ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f),
          "  camera: %s",
          m_selectedCamera->name().c_str());
    } else {
      ImGui::Text("  camera: <Manipulator>");
    }

    ImGui::Text("Viewport: %i x %i", m_viewportSize.x, m_viewportSize.y);
    ImGui::Text("  render: %i x %i", m_renderSize.x, m_renderSize.y);
    ImGui::Text(" samples: %i", m_frameSamples);

    ImGui::Text(" display: %.2fms", m_latestFL);
    ImGui::Text("   ANARI: %.2fms", m_latestAnariFL);
    ImGui::Text("   (min): %.2fms", m_minFL);
    ImGui::Text("   (max): %.2fms", m_maxFL);

    ImGui::Separator();

    ImGui::Checkbox("camera config", &m_showCameraInfo);
    if (m_showCameraInfo) {
      auto at = m_arcball->at();
      auto azel = m_arcball->azel();
      auto dist = m_arcball->distance();
      auto fixedDist = m_arcball->fixedDistance();

      bool update = ImGui::SliderFloat("az", &azel.x, 0.f, 360.f);
      update |= ImGui::SliderFloat("el", &azel.y, 0.f, 360.f);
      update |= ImGui::DragFloat("dist", &dist);
      update |= ImGui::DragFloat3("at", &at.x);
      ImGui::BeginDisabled(m_currentCamera != m_orthoCamera);
      update |= ImGui::DragFloat("near", &fixedDist);
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("near plane distance for orthographic camera");
      ImGui::EndDisabled();

      if (update) {
        m_arcball->setConfig(at, dist, azel);
        m_arcball->setFixedDistance(fixedDist);
      }
    }

    ImGui::End();
  }
}

bool Viewport::canShowGizmo() const
{
  if (!m_enableGizmo || !m_deviceReadyToUse)
    return false;
  if (m_selectedCamera)
    return false; // No gizmo with database camera

  // Check if we have a selected node with a transform
  if (appCore()->tsd.selectedNode) {
    auto &node = *appCore()->tsd.selectedNode;
    return node->isTransform();
  }

  return false;
}

void Viewport::ui_gizmo()
{
  if (!canShowGizmo())
    return;

  auto computeWorldTransform = [](tsd::core::LayerNodeRef node) -> math::mat4 {
    auto world = math::IDENTITY_MAT4;
    for (; node; node = node->parent())
      world = mul((*node)->getTransform(), world);

    return world;
  };

  auto selectedNodeRef = appCore()->tsd.selectedNode;
  auto parentNodeRef = selectedNodeRef->parent();

  auto localTransform = (*selectedNodeRef)->getTransform();
  auto parentWorldTransform = computeWorldTransform(parentNodeRef);
  auto worldTransform = mul(parentWorldTransform, localTransform);

  ImGuizmo::SetOrthographic(m_currentCamera == m_orthoCamera);
  ImGuizmo::BeginFrame();

  // Setup ImGuizmo with window and relative viewport information
  ImVec2 viewportPos = ImGui::GetCursorScreenPos();
  ImVec2 viewportSize = ImGui::GetContentRegionAvail();
  ImVec2 imageMin = ImGui::GetItemRectMin();
  ImVec2 imageMax = ImGui::GetItemRectMax();
  ImVec2 imageSize(imageMax.x - imageMin.x, imageMax.y - imageMin.y);
  ImGuizmo::SetRect(imageMin.x, imageMin.y, imageSize.x, imageSize.y);

  // Build view matrix and projection matrices from manipulator
  // Not sure if we can get those more directly...
  const auto eye = m_arcball->eye();
  const auto at = m_arcball->at();
  const auto up = m_arcball->up();
  const auto view = linalg::lookat_matrix(eye, at, up);

  const float aspect = m_viewportSize.x / float(m_viewportSize.y);
  const float fovRadians = math::radians(m_fov);
  math::mat4 proj;

  // Try and get some legroom for ImGuizmo get precision on depth.
  // We don't know the extent of scene, so try and estimate a good enough near
  // plane position based on the distance to the select object
  const auto selectedObjectPos =
      math::float3(
          worldTransform[3][0], worldTransform[3][1], worldTransform[3][2])
      - eye;
  const float distanceToSelectedObject =
      dot(selectedObjectPos, normalize(at - eye));

  float near = std::max(1e-8f, distanceToSelectedObject * 1e-2f);
  float far = std::max(1e-6f, distanceToSelectedObject * 1e2f);

  if (m_currentCamera == m_perspCamera) {
    float oneOverTanFov = 1.0f / tan(fovRadians / 2.0f);
    proj = math::mat4{
      {oneOverTanFov / aspect, 0.0f, 0.0f, 0.0f},
      {0.0f, oneOverTanFov, 0.0f, 0.0f},
      {0.0f, 0.0f, -(far + near) / (far - near), -1.0f},
      {0.0f, 0.0f, -2.0f * far * near / (far - near), 0.0f},
    };
  } else if (m_currentCamera == m_orthoCamera) {
    // The 0.75 factor is to match updateCameraParametersOrthographic
    const float height = m_arcball->distance() * 0.75f;
    const float halfHeight = height * 0.5f;
    const float halfWidth = halfHeight * aspect;
    const float left = -halfWidth;
    const float right = halfWidth;
    const float bottom = -halfHeight;
    const float top = halfHeight;

    proj = math::mat4{{2.0f / (right - left), 0.0f, 0.0f, 0.0f},
        {0.0f, 2.0f / (top - bottom), 0.0f, 0.0f},
        {0.0f, 0.0f, -2.0f / (far - near), 0.0f},
        {-(right + left) / (right - left),
            -(top + bottom) / (top - bottom),
            -(far + near) / (far - near),
            1.0f}};
  } else {
    // No support for omnidirectional camera, bail out.
    return;
  }

  // Draw and manipulate the gizmo
  ImGuizmo::SetDrawlist();
  if (ImGuizmo::Manipulate(&view[0].x,
          &proj[0].x,
          m_gizmoOperation,
          m_gizmoMode,
          &worldTransform[0].x)) {
    auto invParent = linalg::inverse(parentWorldTransform);
    localTransform = mul(invParent, worldTransform);
    (*selectedNodeRef)->setAsTransform(localTransform);
    appCore()->tsd.scene.signalLayerChange(selectedNodeRef->container());
  }
}

int Viewport::windowFlags() const
{
  return ImGuiWindowFlags_MenuBar;
}

void Viewport::RendererUpdateDelegate::signalParameterUpdated(
    const tsd::core::Object *o, const tsd::core::Parameter *p)
{
  if (d && r) {
    o->updateANARIParameter(d, r, *p, p->name().c_str());
    anari::commitParameters(d, r);
  }
}

} // namespace tsd::ui::imgui
