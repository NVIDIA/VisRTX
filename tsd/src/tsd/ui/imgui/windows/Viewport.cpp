// Copyright 2024-2026 NVIDIA Corporation
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
#include "tsd/rendering/view/ManipulatorToTSD.hpp"
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

  ui_menubar();

  ImGui::BeginDisabled(!m_deviceReadyToUse);

  if (m_outputPass) {
    ImGui::Image((ImTextureID)m_outputPass->getTexture(),
        ImGui::GetContentRegionAvail(),
        ImVec2(0, 1),
        ImVec2(1, 0));
  }

  ui_gizmo();
  ui_handleInput();
  bool didPick = ui_picking(); // Needs to happen before ui_menubar

  // Render the overlay after input handling so it does not interfere.
  if (m_showOverlay)
    ui_overlay();

  ImGui::EndDisabled();

  if (m_anariPass && !didPick) {
    bool needIDs = appCore()->getFirstSelected().valid()
        || m_visualizeAOV == tsd::rendering::AOVType::EDGES
        || m_visualizeAOV == tsd::rendering::AOVType::OBJECT_ID;
    m_anariPass->setEnableIDs(needIDs);
  }

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
  auto axis = m_arcball->axis();
  auto azel = resetAzEl ? tsd::math::float2(0.f, 20.f) : m_arcball->azel();
  m_arcball->setConfig(m_rIdx->computeDefaultView());
  m_arcball->setAzel(azel);
  m_arcball->setAxis(axis);
  m_cameraToken = 0;
}

void Viewport::centerView()
{
  if (!m_device)
    return;
  auto axis = m_arcball->axis();
  auto azel = m_arcball->azel();
  auto dist = m_arcball->distance();
  auto fixedDist = m_arcball->fixedDistance();
  m_arcball->setConfig(m_rIdx->computeDefaultView());
  m_arcball->setAzel(azel);
  m_arcball->setDistance(dist);
  m_arcball->setFixedDistance(fixedDist);
  m_arcball->setAxis(axis);
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
    m_libName = libName;

    m_frameSamples = 0;
    m_latestFL = 0.f;
    m_minFL = std::numeric_limits<float>::infinity();
    m_maxFL = -std::numeric_limits<float>::infinity();

    if (d) {
      tsd::core::logStatus("[viewport] setting up renderer objects...");

      m_rendererObjects = scene.renderersOfDevice(libName);
      if (m_rendererObjects.empty())
        m_rendererObjects = scene.createStandardRenderers(libName, d);
      m_currentRenderer = m_rendererObjects[0];

      tsd::core::logStatus("[viewport] populating render index...");

      m_rIdx = adm.acquireRenderIndex(scene, libName, d);
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

      if (!m_currentCamera)
        m_currentCamera = scene.defaultCamera();

      rendering::updateManipulatorFromCamera(*m_arcball, *m_currentCamera);

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

void Viewport::setLibraryToDefault()
{
  if (appCore()->commandLine.loadedFromStateFile)
    return;

  setLibrary(m_app->commandLineOptions()->useDefaultRenderer
          ? appCore()->anari.libraryList()[0]
          : "");
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

  root["anariLibrary"] = m_libName;

  // Viewport settings //

  root["showOverlay"] = m_showOverlay;
  root["showCameraInfo"] = m_showCameraInfo;
  root["showOnlySelected"] = m_showOnlySelected;
  root["highlightSelection"] = m_highlightSelection;
  root["showOnlySelected"] = m_showOnlySelected;
  root["visualizeAOV"] = static_cast<int>(m_visualizeAOV);
  root["depthVisualMinimum"] = m_depthVisualMinimum;
  root["depthVisualMaximum"] = m_depthVisualMaximum;
  root["edgeThreshold"] = m_edgeThreshold;
  root["edgeInvert"] = m_edgeInvert;
  root["resolutionScale"] = m_resolutionScale;
  root["showAxes"] = m_showAxes;

  // Gizmo settings //

  root["enableGizmo"] = m_enableGizmo;
  root["gizmoOperation"] = static_cast<int>(m_gizmoOperation);
  root["gizmoMode"] = static_cast<int>(m_gizmoMode);

  // Database Camera //

  if (m_currentCamera)
    root["currentCamera"] = static_cast<uint64_t>(m_currentCamera->index());

  // Base window settings //

  Window::saveSettings(root);
}

void Viewport::loadSettings(tsd::core::DataNode &root)
{
  Window::loadSettings(root);

  // Viewport settings //

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
  root["edgeThreshold"].getValue(ANARI_FLOAT32, &m_edgeThreshold);
  root["edgeInvert"].getValue(ANARI_BOOL, &m_edgeInvert);
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

  // Database Camera //

  if (auto *c = root.child("currentCamera"); c) {
    uint64_t idx = 0;
    c->getValue(ANARI_UINT64, &idx);
    m_currentCamera = appCore()->tsd.scene.getObject<tsd::core::Camera>(idx);
  }

  // Setup library //

  auto *core = appCore();
  if (m_app->commandLineOptions()->useDefaultRenderer) {
    std::string libraryName;
    root["anariLibrary"].getValue(ANARI_STRING, &libraryName);
    setLibrary(libraryName);
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

      auto fov = m_currentCamera->parameterValueAs<float>("fovy").value_or(
          math::radians(40.f));
      imgPlaneSize.y = 2.f * tanf(0.5f * fov);
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

      auto *obj = (id == ~0u) ? nullptr
                              : appCore()->tsd.scene.getObject(objectType, id);
      appCore()->setSelected(obj);
    }

    m_pickPass->setEnabled(false);
  });

  m_visualizeAOVPass =
      m_pipeline.emplace_back<tsd::rendering::VisualizeAOVPass>();
  m_visualizeAOVPass->setEnabled(false);
  m_visualizeAOVPass->setEdgeThreshold(m_edgeThreshold);
  m_visualizeAOVPass->setEdgeInvert(m_edgeInvert);

  m_outlinePass = m_pipeline.emplace_back<tsd::rendering::OutlineRenderPass>();

  anari::Extensions extensions{};
  auto &adm = appCore()->anari;
  if (auto *exts = adm.loadDeviceExtensions(m_libName); exts != nullptr)
    extensions = *exts;

  m_axesPass = m_pipeline.emplace_back<tsd::rendering::AnariAxesRenderPass>(
      m_device, extensions);
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

  anari::release(m_device, m_device);

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
    m_rIdx->setFilterFunction([this](const tsd::core::Object *obj) {
      auto selectedNode = appCore()->getFirstSelected();
      if (!selectedNode.valid())
        return true;
      auto *selectedObject = (*selectedNode)->getObject();
      return !selectedObject || obj == selectedObject;
    });
  }
}

void Viewport::updateFrame()
{
  if (!m_anariPass)
    return;

  if (!m_currentCamera)
    m_currentCamera = appCore()->tsd.scene.defaultCamera();

  m_anariPass->setWorld(m_rIdx->world());
  if (m_currentCamera)
    m_anariPass->setCamera(m_rIdx->camera(m_currentCamera->index()));
  if (m_currentRenderer)
    m_anariPass->setRenderer(m_rIdx->renderer(m_currentRenderer->index()));
}

void Viewport::updateCamera(bool force)
{
  if (!m_anariPass)
    return;

  if (!m_currentCamera)
    m_currentCamera = appCore()->tsd.scene.defaultCamera();

  if (!force && !m_arcball->hasChanged(m_cameraToken))
    return;

  // Get compass information
  tsd::rendering::updateCameraObject(*m_currentCamera, *m_arcball);

  auto axesDir =
      m_currentCamera->parameterValueAs<tsd::math::float3>("direction")
          .value_or(tsd::math::float3(0.0f, 0.0f, -1.0f));
  auto axesUp =
      m_currentCamera->parameterValueAs<tsd::math::float3>("up").value_or(
          tsd::math::float3(0.0f, 1.0f, 0.0f));

  m_axesPass->setView(axesDir, axesUp);
}

void Viewport::updateImage()
{
  if (!m_deviceReadyToUse)
    return;

  auto frame = m_anariPass->getFrame();
  anari::getProperty(
      m_device, frame, "numSamples", m_frameSamples, ANARI_NO_WAIT);

  auto selectedNode = appCore()->getFirstSelected();
  const auto *selectedObject =
      selectedNode.valid() ? (*selectedNode)->getObject() : nullptr;
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

void Viewport::ui_menubar()
{
  if (ImGui::BeginMenuBar()) {
    ui_menubar_Device();
    ImGui::BeginDisabled(!m_device);
    ui_menubar_Renderer();
    ui_menubar_Camera();
    ui_menubar_TransformManipulator();
    ui_menubar_Viewport();
    ui_menubar_World();
    ImGui::EndDisabled();
    ImGui::EndMenuBar();
  }
}

void Viewport::ui_menubar_Device()
{
  if (ImGui::BeginMenu("Device")) {
    const auto &libraryList = appCore()->anari.libraryList();
    for (auto &libName : libraryList) {
      const bool isThisLibrary = m_libName == libName;
      if (ImGui::RadioButton(libName.c_str(), isThisLibrary))
        setLibrary(libName);
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Reload Current Device")) {
      auto lib = m_libName; // setLibrary() clears m_libName
      setLibrary(lib);
    }
    ImGui::EndMenu();
  }
}

void Viewport::ui_menubar_Renderer()
{
  if (ImGui::BeginMenu("Renderer")) {
    if (m_rendererObjects.size() > 1) {
      ImGui::Text("Subtype:");
      ImGui::Indent(INDENT_AMOUNT);
      for (int i = 0; i < m_rendererObjects.size(); i++) {
        auto ro = m_rendererObjects[i];
        const char *rName = ro->subtype().c_str();
        if (ImGui::RadioButton(rName, m_currentRenderer == ro)) {
          m_currentRenderer = ro;
          updateFrame();
        }
      }
      ImGui::Unindent(INDENT_AMOUNT);
    }

    ImGui::Separator();

    if (!m_rendererObjects.empty()) {
      ImGui::Text("Parameters:");
      ImGui::Indent(INDENT_AMOUNT);

      tsd::ui::buildUI_object(*m_currentRenderer, appCore()->tsd.scene, true);

      ImGui::Unindent(INDENT_AMOUNT);
      ImGui::Separator();
      ImGui::Separator();
      ImGui::Indent(INDENT_AMOUNT);

      if (ImGui::BeginMenu("Reset to Defaults?")) {
        if (ImGui::MenuItem("Yes")) {
          m_currentRenderer->removeAllParameters();
          m_currentRenderer->setCommonParameterDefaults();
          tsd::core::parseANARIObjectInfo(*m_currentRenderer,
              m_device,
              ANARI_RENDERER,
              m_currentRenderer->subtype().c_str());
        }
        ImGui::EndMenu();
      }

      ImGui::Unindent(INDENT_AMOUNT);
    }
    ImGui::EndMenu();
  }
}

void Viewport::ui_menubar_Camera()
{
  if (ImGui::BeginMenu("Camera")) {
    auto &scene = appCore()->tsd.scene;

    ImGui::Text("Manipulator:");
    {
      ImGui::Indent(INDENT_AMOUNT);

      if (ImGui::Combo("Up", &m_arcballUp, "+x\0+y\0+z\0-x\0-y\0-z\0\0")) {
        m_arcball->setAxis(static_cast<tsd::rendering::UpAxis>(m_arcballUp));
        resetView();
      }

      auto at = m_arcball->at();
      auto azel = m_arcball->azel();
      auto dist = m_arcball->distance();
      auto fixedDist = m_arcball->fixedDistance();

      bool update = ImGui::SliderFloat("Azimuth", &azel.x, 0.f, 360.f);
      update |= ImGui::SliderFloat("Elevation", &azel.y, 0.f, 360.f);
      update |= ImGui::DragFloat("Distance", &dist);
      update |= ImGui::DragFloat3("At", &at.x);
      ImGui::BeginDisabled(
          m_currentCamera->subtype() != core::tokens::camera::orthographic);
      update |= ImGui::DragFloat("Near", &fixedDist);
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("near plane distance for orthographic camera");
      ImGui::EndDisabled();

      if (update) {
        m_arcball->setConfig(at, dist, azel);
        m_arcball->setFixedDistance(fixedDist);
      }

      ImGui::Separator();

      if (ImGui::BeginMenu("Reset View")) {
        if (ImGui::MenuItem("Center"))
          centerView();
        if (ImGui::MenuItem("Distance"))
          resetView(false);
        if (ImGui::MenuItem("Angle + Distance + Center"))
          resetView(true);
        ImGui::EndMenu();
      }

      ImGui::Unindent(INDENT_AMOUNT);
    }

    ImGui::Separator();

    ImGui::Text("Current Camera:");
    {
      ImGui::Indent(INDENT_AMOUNT);

      if (ImGui::BeginMenu("Select Camera")) {
        if (ImGui::BeginMenu("New")) {
          tsd::core::CameraRef newCam;
          if (ImGui::MenuItem("Perspective")) {
            newCam = scene.createObject<tsd::core::Camera>(
                tsd::core::tokens::camera::perspective);
          }
          if (ImGui::MenuItem("Orthographic")) {
            newCam = scene.createObject<tsd::core::Camera>(
                tsd::core::tokens::camera::orthographic);
          }
          if (ImGui::MenuItem("Omnidirectional")) {
            newCam = scene.createObject<tsd::core::Camera>(
                tsd::core::tokens::camera::omnidirectional);
          }

          if (newCam) {
            m_currentCamera = newCam;
            newCam->setName("camera" + std::to_string(newCam->index()));
            updateCamera(true);
            updateFrame();
          }

          ImGui::EndMenu();
        }

        ImGui::Separator();

        auto t = ANARI_CAMERA;
        if (auto i = tsd::ui::buildUI_objects_menulist(scene, t);
            i != TSD_INVALID_INDEX) {
          m_currentCamera = scene.getObject<tsd::core::Camera>(i);
          tsd::rendering::updateManipulatorFromCamera(
              *m_arcball, *m_currentCamera);
          updateFrame();
        }

        ImGui::EndMenu();
      }

      ImGui::Separator();
      tsd::ui::buildUI_object(*m_currentCamera, scene, true);
      ImGui::Unindent(INDENT_AMOUNT);
    }

    ImGui::EndMenu();
  }
}

void Viewport::ui_menubar_TransformManipulator()
{
  if (ImGui::BeginMenu("Transform Manipulator")) {
    ImGui::Checkbox("Enabled", &m_enableGizmo);

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
}

void Viewport::ui_menubar_Viewport()
{
  if (ImGui::BeginMenu("Viewport")) {
    {
      ImGui::Text("Format:");
      ImGui::Indent(INDENT_AMOUNT);
      anari::DataType format = m_anariPass->getColorFormat();
      if (ImGui::RadioButton(
              "UFIXED8_RGBA_SRGB", format == ANARI_UFIXED8_RGBA_SRGB))
        format = ANARI_UFIXED8_RGBA_SRGB;
      if (ImGui::RadioButton("UFIXED8_VEC4", format == ANARI_UFIXED8_VEC4))
        format = ANARI_UFIXED8_VEC4;
      if (ImGui::RadioButton("FLOAT32_VEC4", format == ANARI_FLOAT32_VEC4))
        format = ANARI_FLOAT32_VEC4;

      if (format != m_anariPass->getColorFormat())
        m_anariPass->setColorFormat(format);
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

    {
      ImGui::Text("AOV Visualization:");
      ImGui::Indent(INDENT_AMOUNT);

      const char *aovItems[] = {"default",
          "depth",
          "albedo",
          "normal",
          "edges",
          "object ID",
          "primitive ID",
          "instance ID"};
      if (int aov = int(m_visualizeAOV);
          ImGui::Combo("AOV", &aov, aovItems, IM_ARRAYSIZE(aovItems))) {
        if (aov != int(m_visualizeAOV)) {
          m_visualizeAOV = static_cast<tsd::rendering::AOVType>(aov);
          m_visualizeAOVPass->setAOVType(m_visualizeAOV);
          m_anariPass->setEnableAlbedo(
              m_visualizeAOV == tsd::rendering::AOVType::ALBEDO);
          m_anariPass->setEnableNormals(
              m_visualizeAOV == tsd::rendering::AOVType::NORMAL);
          m_anariPass->setEnableIDs(
              m_visualizeAOV == tsd::rendering::AOVType::EDGES
              || m_visualizeAOV == tsd::rendering::AOVType::OBJECT_ID);
          m_anariPass->setEnablePrimitiveId(
              m_visualizeAOV == tsd::rendering::AOVType::PRIMITIVE_ID);
          m_anariPass->setEnableInstanceId(
              m_visualizeAOV == tsd::rendering::AOVType::INSTANCE_ID);
        }
      }

      ImGui::BeginDisabled(m_visualizeAOV != tsd::rendering::AOVType::DEPTH);
      bool depthRangeChanged = false;
      depthRangeChanged |= ImGui::DragFloat("Depth Minimum",
          &m_depthVisualMinimum,
          0.1f,
          0.f,
          m_depthVisualMaximum);
      depthRangeChanged |= ImGui::DragFloat("Depth Maximum",
          &m_depthVisualMaximum,
          0.1f,
          m_depthVisualMinimum,
          1e20f);
      if (depthRangeChanged)
        m_visualizeAOVPass->setDepthRange(
            m_depthVisualMinimum, m_depthVisualMaximum);
      ImGui::EndDisabled();

      ImGui::BeginDisabled(m_visualizeAOV != tsd::rendering::AOVType::EDGES);
      bool edgeSettingsChanged = false;
      edgeSettingsChanged |=
          ImGui::DragFloat("Edge Threshold", &m_edgeThreshold, 0.01f, 0.f, 1.f);
      edgeSettingsChanged |= ImGui::Checkbox("Invert Edges", &m_edgeInvert);
      if (edgeSettingsChanged) {
        m_visualizeAOVPass->setEdgeThreshold(m_edgeThreshold);
        m_visualizeAOVPass->setEdgeInvert(m_edgeInvert);
      }
      ImGui::EndDisabled();

      ImGui::Unindent(INDENT_AMOUNT);
    }

    ImGui::Separator();

    {
      ImGui::Text("Display:");
      ImGui::Indent(INDENT_AMOUNT);

      ImGui::BeginDisabled(m_showOnlySelected);
      ImGui::Checkbox("Highlight Selected", &m_highlightSelection);
      ImGui::EndDisabled();

      if (ImGui::Checkbox("Only Show Selected", &m_showOnlySelected))
        setSelectionVisibilityFilterEnabled(m_showOnlySelected);

      ImGui::Unindent(INDENT_AMOUNT);
    }

    ImGui::Separator();

    {
      ImGui::Text("Overlay:");
      ImGui::Indent(INDENT_AMOUNT);

      if (ImGui::Checkbox("Axes", &m_showAxes))
        m_axesPass->setEnabled(m_showAxes);

      ImGui::Checkbox("Info Window", &m_showOverlay);
      if (ImGui::MenuItem("Reset Timing Stats")) {
        m_minFL = m_latestFL;
        m_maxFL = m_latestFL;
      }
      ImGui::Unindent(INDENT_AMOUNT);
    }

    ImGui::Separator();

    if (ImGui::MenuItem("Take Screenshot")) {
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
}

void Viewport::ui_menubar_World()
{
  if (ImGui::BeginMenu("World")) {
    if (ImGui::MenuItem("Print Bounds")) {
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

  if (!m_currentCamera)
    return false;

  // Pick view center //

  const bool shouldPickCenter =
      m_currentCamera->subtype() == core::tokens::camera::perspective
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
  ImVec2 contentStart = ImGui::GetCursorStartPos();
  ImGui::SetCursorPos(ImVec2(contentStart[0] + 2.0f, contentStart[1] + 2.0f));

  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0f, 0.0f, 0.0f, 0.7f));

  ImGuiChildFlags childFlags = ImGuiChildFlags_Border
      | ImGuiChildFlags_AutoResizeX | ImGuiChildFlags_AutoResizeY;
  ImGuiWindowFlags childWindowFlags =
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

  // Render overlay as a child window within the viewport.
  // This ensures it's properly occluded when other windows are on top.
  if (ImGui::BeginChild(
          "##viewportOverlay", ImVec2(0, 0), childFlags, childWindowFlags)) {
    ImGui::Text("  device: %s", m_libName.c_str());
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.f),
        "renderer: %s",
        m_currentRenderer ? m_currentRenderer->subtype().c_str() : "---");

    // Camera indicator
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
        "  camera: %s",
        m_currentCamera ? m_currentCamera->name().c_str() : "---");

    ImGui::Separator();

    ImGui::Text("viewport: %i x %i", m_viewportSize.x, m_viewportSize.y);
    ImGui::Text("  render: %i x %i", m_renderSize.x, m_renderSize.y);

    ImGui::Separator();

    ImGui::Text(" samples: %i", m_frameSamples);
    ImGui::Text(" display: %.2fms", m_latestFL);
    ImGui::Text("   ANARI: %.2fms", m_latestAnariFL);
    ImGui::Text("   (min): %.2fms", m_minFL);
    ImGui::Text("   (max): %.2fms", m_maxFL);
  }
  ImGui::EndChild();

  ImGui::PopStyleColor();
}

bool Viewport::canShowGizmo() const
{
  if (!m_enableGizmo || !m_deviceReadyToUse)
    return false;

  // Check if we have a selected node with a transform
  auto selectedNode = appCore()->getFirstSelected();
  if (selectedNode.valid()) {
    return (*selectedNode)->isTransform();
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

  auto selectedNodeRef = appCore()->getFirstSelected();
  auto parentNodeRef = selectedNodeRef->parent();

  auto localTransform = (*selectedNodeRef)->getTransform();
  auto parentWorldTransform = computeWorldTransform(parentNodeRef);
  auto worldTransform = mul(parentWorldTransform, localTransform);

  ImGuizmo::SetOrthographic(
      m_currentCamera->subtype() == core::tokens::camera::orthographic);
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

  if (m_currentCamera->subtype() == core::tokens::camera::perspective) {
    const float fovRadians =
        m_currentCamera->parameterValueAs<float>("fovy").value_or(
            math::radians(40.f));
    float oneOverTanFov = 1.0f / tan(fovRadians / 2.0f);
    proj = math::mat4{
        {oneOverTanFov / aspect, 0.0f, 0.0f, 0.0f},
        {0.0f, oneOverTanFov, 0.0f, 0.0f},
        {0.0f, 0.0f, -(far + near) / (far - near), -1.0f},
        {0.0f, 0.0f, -2.0f * far * near / (far - near), 0.0f},
    };
  } else if (m_currentCamera->subtype() == core::tokens::camera::orthographic) {
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
  return ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoScrollbar;
}

} // namespace tsd::ui::imgui
