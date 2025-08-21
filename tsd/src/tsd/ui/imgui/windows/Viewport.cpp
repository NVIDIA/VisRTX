// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Viewport.h"
#include "Log.h"
// tsd_ui_imgui
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/tsd_ui_imgui.h"
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
#include <sstream>
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
  if (m_initFuture.valid())
    m_initFuture.get();
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

  bool didPick = ui_picking();

  ui_handleInput();
  ui_menubar();

  if (m_anariPass && !didPick)
    m_anariPass->setEnableIDs(appCore()->objectIsSelected());

  if (m_rIdx && (m_rIdx->isFlat() != appCore()->anari.useFlatRenderIndex())) {
    tsd::core::logWarning("instancing setting changed: resetting viewport");
    auto lib = m_libName;
    setLibrary("", false); // clear old library
    setLibrary(lib, false);
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
    auto &ctx = appCore()->tsd.ctx;

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

      m_rIdx = adm.acquireRenderIndex(ctx, d);
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
  };

  if (doAsync)
    m_initFuture = std::async(updateLibrary);
  else
    updateLibrary();
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
  root["depthVisualMaximum"] = m_depthVisualMaximum;
  root["fov"] = m_fov;
  root["resolutionScale"] = m_resolutionScale;
  root["showAxes"] = m_showAxes;

  root["anariLibrary"] = m_libName;

  // Camera //

  auto &camera = root["camera"];
  camera["at"] = m_arcball->at();
  camera["distance"] = m_arcball->distance();
  camera["azel"] = m_arcball->azel();
  camera["up"] = int(m_arcball->axis());
  camera["apertureRadius"] = m_apertureRadius;
  camera["focusDistance"] = m_focusDistance;

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
  root["depthVisualMaximum"].getValue(ANARI_FLOAT32, &m_depthVisualMaximum);
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

  // Setup library //

  std::string libraryName;
  root["anariLibrary"].getValue(ANARI_STRING, &libraryName);
  setLibrary(libraryName, false);

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
          id == ~0u ? nullptr : appCore()->tsd.ctx.getObject(objectType, id));
    }

    m_pickPass->setEnabled(false);
  });
  m_visualizeDepthPass =
      m_pipeline.emplace_back<tsd::rendering::VisualizeDepthPass>();
  m_visualizeDepthPass->setEnabled(false);
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
  if (!m_deviceReadyToUse)
    return;

  m_pipeline.clear();
  m_anariPass = nullptr;
  m_outlinePass = nullptr;
  m_outputPass = nullptr;

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

  if ((!force && !m_arcball->hasChanged(m_cameraToken)))
    return;

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

  m_axesPass->setView(m_arcball->dir(), m_arcball->up());
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

  // Handle screenshot saving
  if (m_saveNextFrame) {
    // Wait for the frame to be ready
    anari::wait(m_device, frame);

    // Map the color channel to get the frame buffer data
    auto fb = anari::map<uint32_t>(m_device, frame, "channel.color");

    if (fb.data) {
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

      // Use STB's built-in vertical flip functionality for simplicity
      stbi_flip_vertically_on_write(1);

      // Save the screenshot with full alpha channel preservation
      stbi_write_png(
          filename.string().c_str(), fb.width, fb.height, 4, fb.data, 4 * fb.width);

      // Reset the flip setting to avoid affecting other code
      stbi_flip_vertically_on_write(0);

      tsd::core::logStatus(
          "Screenshot saved to '%s'", filename.string().c_str());
    } else {
      tsd::core::logWarning("Failed to map frame buffer for screenshot");
    }

    // Unmap the frame buffer
    anari::unmap(m_device, frame, "channel.color");

    m_saveNextFrame = false;
  }
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
          setLibrary(libName, false);
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
            m_rendererObjects[m_currentRenderer], appCore()->tsd.ctx, false);

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

      if (ImGui::Checkbox("visualize depth", &m_visualizeDepth))
        m_visualizeDepthPass->setEnabled(m_visualizeDepth);

      ImGui::BeginDisabled(!m_visualizeDepth);
      if (ImGui::DragFloat("maximum", &m_depthVisualMaximum, 1.f, 1e-3f, 1e20f))
        m_visualizeDepthPass->setMaxDepth(m_depthVisualMaximum);
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

      if (ImGui::MenuItem("take screenshot"))
        m_saveNextFrame = true;

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
  if (!m_deviceReadyToUse || !ImGui::IsWindowFocused())
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
    ImGui::Text("Viewport: %i x %i", m_viewportSize.x, m_viewportSize.y);
    ImGui::Text("  render: %i x %i", m_renderSize.x, m_renderSize.y);
    ImGui::Text(" samples: %i", m_frameSamples);

    ImGui::Text(" display: %.2fms", m_latestFL);
    ImGui::Text("   ANARI: %.2fms", m_latestAnariFL);
    ImGui::Text("   (min): %.2fms", m_minFL);
    ImGui::Text("   (max): %.2fms", m_maxFL);

    ImGui::Separator();

    ImGui::Checkbox("camera setup", &m_showCameraInfo);
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
