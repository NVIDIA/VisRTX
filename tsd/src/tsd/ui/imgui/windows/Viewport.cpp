// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Viewport.h"
// tsd_ui_imgui
#include "imgui.h"
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/tsd_ui_imgui.h"
// tsd_core
#include "tsd/core/Logging.hpp"
#include "tsd/scene/objects/Camera.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"
// tsd_rendering
#include "tsd/rendering/view/ManipulatorToTSD.hpp"
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

namespace tsd::ui::imgui {

Viewport::Viewport(
    Application *app, tsd::rendering::Manipulator *m, const char *name)
    : BaseViewport(app, name)
{
  m_viewport.resolutionScale = 0.75f;
  BaseViewport::setManipulator(m);
  m_defragToken = appContext()->tsd.scene.addDefragCallback(
      [this](const auto &) { m_refreshDeviceNextFrame = true; });
}

Viewport::~Viewport()
{
  teardownDevice();
  appContext()->tsd.scene.removeDefragCallback(m_defragToken);
}

void Viewport::buildUI()
{
  const bool setupPipeline = BaseViewport::viewport_isActive()
      && !BaseViewport::imagePipeline_isSetup();
  if (setupPipeline)
    BaseViewport::imagePipeline_setup();

  if (m_refreshDeviceNextFrame) {
    if (!m_libName.empty()) {
      auto lib = m_libName; // setLibrary() clears m_libName
      setLibrary(lib);
    }
    m_refreshDeviceNextFrame = false;
  }

  BaseViewport::buildUI();

  if (m_prevRenderer != m_renderers.current
      || m_prevCamera != m_camera.current) {
    updateFrame();
  }

  updateImage();
  BaseViewport::camera_update();

  ui_menubar();

  ImGui::BeginDisabled(!BaseViewport::imagePipeline_isSetup());

  if (m_outputPass) {
    ImGui::Image((ImTextureID)m_outputPass->getTexture(),
        ImGui::GetContentRegionAvail(),
        ImVec2(0, 1),
        ImVec2(1, 0));
  }

  BaseViewport::ui_gizmo();
  const bool widgetActive = BaseViewport::ui_orientationWidget();
  if (!widgetActive)
    BaseViewport::ui_handleInput();
  bool didPick = ui_picking(); // Needs to happen before ui_menubar

  // Render the overlay after input handling so it does not interfere.
  if (m_showOverlay)
    ui_overlay();

  BaseViewport::ui_animationSlider();

  ImGui::EndDisabled();

  if (m_anariPass && !didPick) {
    bool needIDs = appContext()->getFirstSelected().valid()
        || m_visualizeAOV == tsd::rendering::AOVType::EDGES
        || m_visualizeAOV == tsd::rendering::AOVType::OBJECT_ID;
    m_anariPass->setEnableIDs(needIDs);
  }

  if (m_rIdx) {
    auto kind = appContext()->anari.renderIndexKind();
    if (kind != m_lastIndexKind) {
      tsd::core::logWarning("render index setting changed: resetting viewport");
      m_lastIndexKind = kind;
      m_refreshDeviceNextFrame = true;
    }
  }
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
    auto &adm = appContext()->anari;
    auto &scene = appContext()->tsd.scene;

    auto start = std::chrono::steady_clock::now();
    auto d = adm.loadDevice(libName);
    m_libName = libName;

    m_frameSamples = 0;
    m_latestFL = 0.f;
    m_minFL = std::numeric_limits<float>::infinity();
    m_maxFL = -std::numeric_limits<float>::infinity();

    if (d) {
      tsd::core::logStatus("[viewport] setting up renderer objects...");

      m_renderers.objects = scene.renderersOfDevice(libName);
      if (m_renderers.objects.empty())
        m_renderers.objects = scene.createStandardRenderers(libName, d);
      m_renderers.current = m_renderers.objects[0];

      tsd::core::logStatus("[viewport] populating render index...");

      m_rIdx = adm.acquireRenderIndex(scene, libName, d);
      setSelectionVisibilityFilterEnabled(m_showOnlySelected);

      tsd::core::logStatus("[viewport] getting scene bounds...");

      m_device = d;
      viewport_setActive(true);

      static bool firstFrame = true;
      if (firstFrame && appContext()->commandLine.loadedFromStateFile)
        firstFrame = false;

      if (!m_camera.current)
        BaseViewport::camera_setCurrent(scene.defaultCamera());

      rendering::updateManipulatorFromCamera(
          *m_camera.arcball, *m_camera.current);

      if (firstFrame || m_camera.arcball->distance() == tsd::math::inf) {
        camera_resetView(true);
        if (appContext()->view.poses.empty()) {
          tsd::core::logStatus("[viewport] adding 'default' camera pose");
          appContext()->addCurrentViewToCameraPoses("default");
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
  if (appContext()->commandLine.loadedFromStateFile)
    return;

  setLibrary(m_app->commandLineOptions()->useDefaultRenderer
          ? appContext()->anari.libraryList()[0]
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

  auto f = m_anariPass->getFrame();
  anari::setParameter(m_device, f, name, value.type(), value.data());
  anari::commitParameters(m_device, f);
}

void Viewport::saveSettings(tsd::core::DataNode &root)
{
  root["anariLibrary"] = m_libName;

  // Viewport settings //

  root["showOverlay"] = m_showOverlay;
  root["showOnlySelected"] = m_showOnlySelected;
  root["highlightSelection"] = m_highlightSelection;
  root["showOnlySelected"] = m_showOnlySelected;
  root["visualizeAOV"] = static_cast<int>(m_visualizeAOV);
  root["depthVisualMinimum"] = m_depthVisualMinimum;
  root["depthVisualMaximum"] = m_depthVisualMaximum;
  root["edgeInvert"] = m_edgeInvert;
  root["autoExposureEnabled"] = m_autoExposureEnabled;
  root["toneMapExposure"] = m_toneMapExposure;
  root["toneMapGamma"] = m_toneMapGamma;
  root["toneMapOperator"] = static_cast<int>(m_toneMapOperator);

  // Database Camera //

  if (m_camera.current)
    root["currentCamera"] = static_cast<uint64_t>(m_camera.current->index());

  // BaseViewport settings //

  BaseViewport::saveSettings(root);
}

void Viewport::loadSettings(tsd::core::DataNode &root)
{
  BaseViewport::loadSettings(root);

  // Viewport settings //

  root["showOverlay"].getValue(ANARI_BOOL, &m_showOverlay);
  root["showOnlySelected"].getValue(ANARI_BOOL, &m_showOnlySelected);
  root["highlightSelection"].getValue(ANARI_BOOL, &m_highlightSelection);
  root["showOnlySelected"].getValue(ANARI_BOOL, &m_showOnlySelected);
  int aovType = static_cast<int>(m_visualizeAOV);
  root["visualizeAOV"].getValue(ANARI_INT32, &aovType);
  m_visualizeAOV = static_cast<tsd::rendering::AOVType>(aovType);
  root["depthVisualMinimum"].getValue(ANARI_FLOAT32, &m_depthVisualMinimum);
  root["depthVisualMaximum"].getValue(ANARI_FLOAT32, &m_depthVisualMaximum);
  root["edgeInvert"].getValue(ANARI_BOOL, &m_edgeInvert);
  root["autoExposureEnabled"].getValue(ANARI_BOOL, &m_autoExposureEnabled);
  root["toneMapExposure"].getValue(ANARI_FLOAT32, &m_toneMapExposure);
  root["toneMapGamma"].getValue(ANARI_FLOAT32, &m_toneMapGamma);
  int toneMapOperator = static_cast<int>(m_toneMapOperator);
  root["toneMapOperator"].getValue(ANARI_INT32, &toneMapOperator);
  m_toneMapOperator =
      static_cast<tsd::rendering::ToneMapOperator>(toneMapOperator);

  // Database Camera //

  if (auto *c = root.child("currentCamera"); c) {
    uint64_t idx = 0;
    c->getValue(ANARI_UINT64, &idx);
    m_camera.current =
        appContext()->tsd.scene.getObject<tsd::scene::Camera>(idx);
  }

  // Setup library //

  auto *ctx = appContext();
  if (m_app->commandLineOptions()->useDefaultRenderer) {
    std::string libraryName;
    root["anariLibrary"].getValue(ANARI_STRING, &libraryName);
    setLibrary(libraryName);
  }
}

void Viewport::imagePipeline_populate(tsd::rendering::ImagePipeline &p)
{
  tsd::core::logStatus("[viewport] initialized scene for '%s' device in %.2fs",
      m_libName.c_str(),
      m_timeToLoadDevice);

  m_anariPass = p.emplace_back<tsd::rendering::AnariSceneRenderPass>(m_device);

  m_saveToFilePass = p.emplace_back<tsd::rendering::SaveToFilePass>();
  m_saveToFilePass->setEnabled(false);
  m_saveToFilePass->setSingleShotMode(true);

  m_pickPass = p.emplace_back<tsd::rendering::PickPass>();
  m_pickPass->setEnabled(false);
  m_pickPass->setPickOperation([&](tsd::rendering::ImageBuffers &b) {
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
      pixel.y = m_viewport.size.y - int(mPos[1] - wMin[1]);

      const float aspect = m_viewport.size.x / float(m_viewport.size.y);
      anari::math::float2 imgPlaneSize;

      auto fov = m_camera.current->parameterValueAs<float>("fovy").value_or(
          math::radians(40.f));
      imgPlaneSize.y = 2.f * tanf(0.5f * fov);
      imgPlaneSize.x = imgPlaneSize.y * aspect;

      const auto d = m_camera.arcball->dir();
      const auto u = m_camera.arcball->up();

      const auto dir_du =
          anari::math::normalize(anari::math::cross(d, u)) * imgPlaneSize.x;
      const auto dir_dv = anari::math::normalize(anari::math::cross(dir_du, d))
          * imgPlaneSize.y;
      const auto dir_00 = d - .5f * dir_du - .5f * dir_dv;

      const auto screen = anari::math::float2(1.f / m_viewport.size.x * pixel.x,
          (1.f / m_viewport.size.y * pixel.y));

      const auto dir = anari::math::normalize(
          dir_00 + screen.x * dir_du + screen.y * dir_dv);

      const auto p = m_camera.arcball->eye();
      const auto c = p + m_pickedDepth * dir;

      tsd::core::logStatus(
          "[viewport] pick [%i, %i] {%f, %f} depth %f / %f| {%f, %f, %f}",
          int(pixel.x),
          int(pixel.y),
          screen.x,
          screen.y,
          m_pickedDepth,
          m_camera.arcball->distance(),
          c.x,
          c.y,
          c.z);

      m_camera.arcball->setCenter(c);
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

      auto *obj = (id == ~0u)
          ? nullptr
          : appContext()->tsd.scene.getObject(objectType, id);
      appContext()->setSelected(obj);
    }

    m_pickPass->setEnabled(false);
  });

  m_autoExposurePass = p.emplace_back<tsd::rendering::AutoExposurePass>();

  m_toneMapPass = p.emplace_back<tsd::rendering::ToneMapPass>();
  m_toneMapPass->setOperator(m_toneMapOperator);
  m_toneMapPass->setAutoExposureEnabled(m_autoExposureEnabled);
  m_toneMapPass->setExposure(m_toneMapExposure);

  m_outputTransformPass = p.emplace_back<tsd::rendering::OutputTransformPass>();
  m_outputTransformPass->setGamma(m_toneMapGamma);

  m_visualizeAOVPass = p.emplace_back<tsd::rendering::VisualizeAOVPass>();
  m_visualizeAOVPass->setEnabled(false);
  m_visualizeAOVPass->setEdgeInvert(m_edgeInvert);
  updateDisplayPassState();

  m_outlinePass = p.emplace_back<tsd::rendering::OutlineRenderPass>();

  m_outputPass = p.emplace_back<tsd::rendering::CopyToSDLTexturePass>(
      m_app->sdlRenderer());
}

void Viewport::camera_resetView(bool resetAzEl)
{
  if (!BaseViewport::viewport_isActive())
    return;
  auto axis = m_camera.arcball->axis();
  auto azel =
      resetAzEl ? tsd::math::float2(0.f, 20.f) : m_camera.arcball->azel();
  m_camera.arcball->setConfig(m_rIdx->computeDefaultView());
  m_camera.arcball->setAzel(azel);
  m_camera.arcball->setAxis(axis);
  m_camera.arcballToken = 0;
}

void Viewport::camera_centerView()
{
  if (!BaseViewport::viewport_isActive())
    return;
  auto axis = m_camera.arcball->axis();
  auto azel = m_camera.arcball->azel();
  auto dist = m_camera.arcball->distance();
  auto fixedDist = m_camera.arcball->fixedDistance();
  m_camera.arcball->setConfig(m_rIdx->computeDefaultView());
  m_camera.arcball->setAzel(azel);
  m_camera.arcball->setDistance(dist);
  m_camera.arcball->setFixedDistance(fixedDist);
  m_camera.arcball->setAxis(axis);
  m_camera.arcballToken = 0;
}

void Viewport::renderer_resetParameterDefaults()
{
  if (!m_device || !m_renderers.current)
    return;

  m_renderers.current->removeAllParameters();
  m_renderers.current->setCommonParameterDefaults();
  tsd::scene::parseANARIObjectInfo(*m_renderers.current,
      m_device,
      ANARI_RENDERER,
      m_renderers.current->subtype().c_str());
}

void Viewport::teardownDevice()
{
  if (m_initFuture.valid())
    m_initFuture.get();

  if (!BaseViewport::imagePipeline_isSetup())
    return;

  BaseViewport::imagePipeline_teardown();

  m_anariPass = nullptr;
  m_pickPass = nullptr;
  m_visualizeAOVPass = nullptr;
  m_autoExposurePass = nullptr;
  m_toneMapPass = nullptr;
  m_outputTransformPass = nullptr;
  m_outlinePass = nullptr;
  m_outputPass = nullptr;
  m_saveToFilePass = nullptr;

  appContext()->anari.releaseRenderIndex(m_device);
  m_rIdx = nullptr;
  m_libName.clear();

  m_camera.current = {};
  m_prevCamera = {};

  anari::release(m_device, m_device);

  m_renderers.objects.clear();
  m_renderers.current = nullptr;
  m_prevRenderer = {};

  m_device = nullptr;

  BaseViewport::viewport_setActive(false);
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
    m_rIdx->setFilterFunction([this](const tsd::scene::Object *obj) {
      auto selectedNode = appContext()->getFirstSelected();
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

  if (!m_camera.current)
    m_camera.current = appContext()->tsd.scene.defaultCamera();

  m_anariPass->setWorld(m_rIdx->world());
  if (m_camera.current)
    m_anariPass->setCamera(m_rIdx->camera(m_camera.current->index()));
  if (m_renderers.current) {
    m_anariPass->setRenderer(m_rIdx->renderer(m_renderers.current->index()));
    m_prevRenderer = m_renderers.current;
  }
}

void Viewport::updateImage()
{
  if (!m_anariPass)
    return;

  auto frame = m_anariPass->getFrame();
  anari::getProperty(
      m_device, frame, "numSamples", m_frameSamples, ANARI_NO_WAIT);

  auto selectedNode = appContext()->getFirstSelected();
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
  BaseViewport::imagePipeline_render();
  if (m_autoExposurePass)
    m_currentAutoExposure = m_autoExposurePass->currentExposure();
  auto end = std::chrono::steady_clock::now();
  m_latestFL = std::chrono::duration<float>(end - start).count() * 1000;

  float duration = 0.f;
  anari::getProperty(m_device, frame, "duration", duration, ANARI_NO_WAIT);

  m_latestAnariFL = duration * 1000;
  m_minFL = std::min(m_minFL, m_latestAnariFL);
  m_maxFL = std::max(m_maxFL, m_latestAnariFL);
}

void Viewport::updateDisplayPassState()
{
  if (!m_toneMapPass || !m_outputTransformPass)
    return;

  const bool showBeauty = m_visualizeAOV == tsd::rendering::AOVType::NONE;
  if (m_autoExposurePass) {
    m_autoExposurePass->setEnabled(showBeauty && m_autoExposureEnabled);
    m_autoExposurePass->setHDREnabled(
        showBeauty && m_colorFormat == ANARI_FLOAT32_VEC4);
  }
  m_toneMapPass->setEnabled(showBeauty);
  m_outputTransformPass->setEnabled(showBeauty);
  m_toneMapPass->setAutoExposureEnabled(showBeauty && m_autoExposureEnabled);
  m_toneMapPass->setHDREnabled(
      showBeauty && m_colorFormat == ANARI_FLOAT32_VEC4);
  m_outputTransformPass->setColorFormat(m_colorFormat);
}

void Viewport::ui_menubar()
{
  if (ImGui::BeginMenuBar()) {
    ui_menubar_Device();
    ImGui::BeginDisabled(!viewport_isActive());
    BaseViewport::ui_menubar_Renderer();
    BaseViewport::ui_menubar_Camera();
    BaseViewport::ui_menubar_TransformManipulator();
    ui_menubar_Viewport();
    ui_menubar_World();
    ImGui::EndDisabled();
    ImGui::EndMenuBar();
  }
}

void Viewport::ui_menubar_Device()
{
  if (ImGui::BeginMenu("Device")) {
    const auto &libraryList = appContext()->anari.libraryList();
    for (auto &libName : libraryList) {
      const bool isThisLibrary = m_libName == libName;
      if (ImGui::RadioButton(libName.c_str(), isThisLibrary))
        setLibrary(libName);
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Reload Current Device"))
      m_refreshDeviceNextFrame = true;
    ImGui::EndMenu();
  }
}

void Viewport::ui_menubar_Viewport()
{
  if (ImGui::BeginMenu("Viewport")) {
    {
      ImGui::Text("Format:");
      ImGui::Indent(INDENT_AMOUNT);
      anari::DataType format = m_colorFormat;
      if (ImGui::RadioButton(
              "UFIXED8_RGBA_SRGB", format == ANARI_UFIXED8_RGBA_SRGB))
        format = ANARI_UFIXED8_RGBA_SRGB;
      if (ImGui::RadioButton("UFIXED8_VEC4", format == ANARI_UFIXED8_VEC4))
        format = ANARI_UFIXED8_VEC4;
      if (ImGui::RadioButton("FLOAT32_VEC4", format == ANARI_FLOAT32_VEC4))
        format = ANARI_FLOAT32_VEC4;

      if (format != m_colorFormat) {
        m_anariPass->setColorFormat(format);
        m_colorFormat = format;
        updateDisplayPassState();
      }
      ImGui::Unindent(INDENT_AMOUNT);
    }

    ImGui::Separator();

    {
      ImGui::Text("Render Resolution:");
      ImGui::Indent(INDENT_AMOUNT);

      const float current = m_viewport.resolutionScale;
      if (ImGui::RadioButton("100%", current == 1.f))
        m_viewport.resolutionScale = 1.f;
      if (ImGui::RadioButton("75%", current == 0.75f))
        m_viewport.resolutionScale = 0.75f;
      if (ImGui::RadioButton("50%", current == 0.5f))
        m_viewport.resolutionScale = 0.5f;
      if (ImGui::RadioButton("25%", current == 0.25f))
        m_viewport.resolutionScale = 0.25f;
      if (ImGui::RadioButton("12.5%", current == 0.125f))
        m_viewport.resolutionScale = 0.125f;

      if (ImGui::BeginMenu("Custom")) {
        ImGui::DragFloat("##customRes",
            &m_viewport.resolutionScale,
            0.01f,
            0.1f,
            2.f,
            "%.2f");
        ImGui::EndMenu();
      }

      if (current != m_viewport.resolutionScale) {
        if (m_viewport.resolutionScale < 0.05f)
          m_viewport.resolutionScale = 0.05f;
        viewport_reshape(m_viewport.size);
      }

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
          updateDisplayPassState();
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
      edgeSettingsChanged |= ImGui::Checkbox("Invert Edges", &m_edgeInvert);
      if (edgeSettingsChanged) {
        m_visualizeAOVPass->setEdgeInvert(m_edgeInvert);
      }
      ImGui::EndDisabled();

      ImGui::Unindent(INDENT_AMOUNT);
    }

    ImGui::Separator();

    {
      ImGui::Text("Exposure:");
      ImGui::Indent(INDENT_AMOUNT);

      ImGui::BeginDisabled(m_colorFormat != ANARI_FLOAT32_VEC4
          || m_visualizeAOV != tsd::rendering::AOVType::NONE);

      if (ImGui::Checkbox("Auto Exposure", &m_autoExposureEnabled))
        updateDisplayPassState();

      if (m_autoExposureEnabled) {
        if (ImGui::DragFloat(
                "Compensation", &m_toneMapExposure, 0.05f, -10.f, 10.f))
          m_toneMapPass->setExposure(m_toneMapExposure);
        ImGui::Text("Current EV: %.2f", m_currentAutoExposure);
      } else if (ImGui::DragFloat(
                     "Exposure", &m_toneMapExposure, 0.05f, -10.f, 10.f)) {
        m_toneMapPass->setExposure(m_toneMapExposure);
      }

      ImGui::EndDisabled();
      ImGui::Unindent(INDENT_AMOUNT);
    }

    ImGui::Separator();

    {
      ImGui::Text("Tonemapping:");
      ImGui::Indent(INDENT_AMOUNT);

      ImGui::BeginDisabled(m_colorFormat != ANARI_FLOAT32_VEC4
          || m_visualizeAOV != tsd::rendering::AOVType::NONE);

      const char *toneMapItems[] = {"None",
          "Reinhard",
          "ACES Filmic",
          "Hable",
          "Khronos PBR Neutral",
          "AgX"};
      if (int op = int(m_toneMapOperator); ImGui::Combo(
              "Operator", &op, toneMapItems, IM_ARRAYSIZE(toneMapItems))) {
        m_toneMapOperator = static_cast<tsd::rendering::ToneMapOperator>(op);
        m_toneMapPass->setOperator(m_toneMapOperator);
      }

      ImGui::EndDisabled();
      ImGui::Unindent(INDENT_AMOUNT);
    }

    ImGui::Separator();

    {
      ImGui::Text("Output Transform:");
      ImGui::Indent(INDENT_AMOUNT);

      ImGui::BeginDisabled(m_colorFormat == ANARI_UFIXED8_RGBA_SRGB
          || m_visualizeAOV != tsd::rendering::AOVType::NONE);

      if (ImGui::DragFloat("Gamma", &m_toneMapGamma, 0.01f, 0.1f, 5.f))
        m_outputTransformPass->setGamma(m_toneMapGamma);

      ImGui::EndDisabled();
      ImGui::Unindent(INDENT_AMOUNT);
    }

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

      ImGui::Checkbox("Axes", &m_showOrientationWidget);
      ImGui::Checkbox("Animation Time Slider", &m_showAnimationSlider);
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

bool Viewport::ui_picking()
{
  const ImGuiIO &io = ImGui::GetIO();

  if (!m_camera.current)
    return false;

  // Pick view center //

  const bool shouldPickCenter =
      m_camera.current->subtype() == scene::tokens::camera::perspective
      && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)
      && ImGui::IsKeyDown(ImGuiKey_LeftShift);
  if (shouldPickCenter && ImGui::IsWindowHovered()) {
    auto mPos = ImGui::GetMousePos();
    auto wMin = ImGui::GetItemRectMin();
    auto pixel = tsd::math::int2(
        tsd::math::float2(
            m_viewport.size.x - (mPos[0] - wMin[0]), mPos[1] - wMin[1])
        * m_viewport.resolutionScale);
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
                     m_viewport.size.x - (mPos[0] - wMin[0]), mPos[1] - wMin[1])
        * m_viewport.resolutionScale;
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
        m_renderers.current ? m_renderers.current->subtype().c_str() : "---");

    // Camera indicator
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
        "  camera: %s",
        m_camera.current ? m_camera.current->name().c_str() : "---");

    ImGui::Separator();

    ImGui::Text("viewport: %i x %i", m_viewport.size.x, m_viewport.size.y);
    ImGui::Text(
        "  render: %i x %i", m_viewport.renderSize.x, m_viewport.renderSize.y);

    ImGui::Separator();

    ImGui::Text(" samples: %i", m_frameSamples);
    ImGui::Text(" display: %.2fms", m_latestFL);
    ImGui::Text("   ANARI: %.2fms", m_latestAnariFL);
    ImGui::Text("   (min): %.2fms", m_minFL);
    ImGui::Text("   (max): %.2fms", m_maxFL);

    const auto &passTimings = imagePipeline().getPassTimings();
    if (!passTimings.empty()) {
      ImGui::Separator();
      ImGui::Text("passes:");
      for (const auto &timing : passTimings)
        ImGui::Text("  %s: %.2fms", timing.name, timing.milliseconds);
    }
  }
  ImGui::EndChild();

  ImGui::PopStyleColor();
}

} // namespace tsd::ui::imgui
