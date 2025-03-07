// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Viewport.h"
#include "Log.h"
// std
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
// stb_image
#include "tsd_stb/stb_image_write.h"
// tsd
#include "../tsd_ui.h"

namespace tsd_viewer {

constexpr float inf = std::numeric_limits<float>::infinity();
static std::atomic_bool g_firstFrame = true;
int g_screenshotIndex = 0;

// Viewport definitions ///////////////////////////////////////////////////////

Viewport::Viewport(AppCore *ctx, manipulators::Orbit *m, const char *name)
    : anari_viewer::windows::Window(name, true), m_core(ctx)
{
  stbi_flip_vertically_on_write(1);

  setManipulator(m);

  m_overlayWindowName = "overlay_";
  m_overlayWindowName += name;

  m_coreMenuName = "vpContextMenu_";
  m_coreMenuName += name;

  // GL //

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
  if (deviceReady && !m_deviceReadyToUse) {
    tsd::logStatus("[viewport] initialized scene for '%s' device in %.2fs",
        m_libName.c_str(),
        m_timeToLoadDevice);
    m_anariPass = m_pipeline.emplace_back<tsd::AnariRenderPass>(m_device);
    m_visualizeDepthPass = m_pipeline.emplace_back<tsd::VisualizeDepthPass>();
    m_visualizeDepthPass->setEnabled(false);
    m_outlinePass = m_pipeline.emplace_back<tsd::OutlineRenderPass>();
    m_outputPass = m_pipeline.emplace_back<tsd::CopyToGLImagePass>();
    reshape(m_viewportSize);
  }

  m_deviceReadyToUse = deviceReady;

  ImVec2 _viewportSize = ImGui::GetContentRegionAvail();
  tsd::math::int2 viewportSize(_viewportSize.x, _viewportSize.y);

  if (m_viewportSize != viewportSize)
    reshape(viewportSize);

  updateImage();
  updateCamera();

  ImGui::BeginDisabled(!m_deviceReadyToUse);

  if (m_outputPass) {
    ImGui::Image((void *)(intptr_t)m_outputPass->getGLTexture(),
        ImGui::GetContentRegionAvail(),
        ImVec2(0, 1),
        ImVec2(1, 0));
  }

  if (m_showOverlay)
    ui_overlay();

  ImGui::EndDisabled();

  ui_picking();
  ui_contextMenu();

  if (!m_coreMenuVisible)
    ui_handleInput();
}

void Viewport::setManipulator(manipulators::Orbit *m)
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
    tsd::logWarning("[viewport] ANARIWorld returned no bounds!");
  }

  auto center = 0.5f * (bounds[0] + bounds[1]);
  auto diag = bounds[1] - bounds[0];

  auto azel = resetAzEl ? tsd::math::float2(0.f, 20.f) : m_arcball->azel();
  m_arcball->setConfig(center, 1.25f * linalg::length(diag), azel);
  m_cameraToken = 0;
}

void Viewport::setLibrary(const std::string &libName, bool doAsync)
{
  teardownDevice();

  if (!libName.empty() && libName != "{none}") {
    tsd::logStatus(
        "[viewport] *** setting viewport to use ANARI device '%s' ***",
        libName.c_str());
  }

  auto updateLibrary = [&, libName = libName]() {
    auto start = std::chrono::steady_clock::now();
    auto d = m_core->loadDevice(libName);
    m_rud.d = d;
    m_libName = libName;

    m_frameSamples = 0;
    m_latestFL = 0.f;
    m_minFL = std::numeric_limits<float>::infinity();
    m_maxFL = -std::numeric_limits<float>::infinity();

    if (d) {
      anari::retain(d, d);

      tsd::logStatus("[viewport] getting renderer params...");

      const char **r_subtypes = anariGetObjectSubtypes(d, ANARI_RENDERER);

      std::vector<std::string> rendererNames;
      if (r_subtypes != nullptr) {
        for (int i = 0; r_subtypes[i] != nullptr; i++)
          rendererNames.push_back(r_subtypes[i]);
      } else
        rendererNames.emplace_back("default");
      m_currentRenderer = 0;

      for (auto &name : rendererNames) {
        auto ar = anari::newObject<anari::Renderer>(d, name.c_str());
        auto o = tsd::ui::parseANARIObject(d, ANARI_RENDERER, name.c_str());
        o.setName(name.c_str());
        o.setUpdateDelegate(&m_rud);
        o.updateAllANARIParameters(d, ar);
        anari::commitParameters(d, ar);
        m_rendererObjects.push_back(std::move(o));
        m_renderers.push_back(ar);
      }

      m_perspCamera = anari::newObject<anari::Camera>(d, "perspective");
      m_orthoCamera = anari::newObject<anari::Camera>(d, "orthographic");
      m_omniCamera = anari::newObject<anari::Camera>(d, "omnidirectional");

      tsd::logStatus("[viewport] populating render index...");

      m_rIdx = m_core->acquireRenderIndex(d);
      setSelectionVisibilityFilterEnabled(m_showOnlySelected);

      tsd::logStatus("[viewport] getting scene bounds...");

      m_device = d;

      if (g_firstFrame || m_arcball->distance() == inf) {
        resetView(true);
        g_firstFrame = false;
      } else {
        // NOTE(jda) - this *should* cause a commit buffer flush
        tsd::math::float3 bounds[2];
        anariGetProperty(m_device,
            m_rIdx->world(),
            "bounds",
            ANARI_FLOAT32_BOX3,
            &bounds[0],
            sizeof(bounds),
            ANARI_WAIT);
      }

      tsd::logStatus("[viewport] ...device load complete");
    }

    auto end = std::chrono::steady_clock::now();
    m_timeToLoadDevice = std::chrono::duration<float>(end - start).count();
  };

  if (doAsync)
    m_initFuture = std::async(updateLibrary);
  else
    updateLibrary();
}

void Viewport::teardownDevice()
{
  if (!m_deviceReadyToUse)
    return;

  m_pipeline.clear();
  m_anariPass = nullptr;
  m_outlinePass = nullptr;
  m_outputPass = nullptr;

  m_core->releaseRenderIndex(m_device);
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
  auto frame = m_anariPass->getFrame();

  // Get depth //

  auto depth = anari::map<float>(m_device, frame, "channel.depth");

  l = linalg::clamp(l,
      tsd::math::int2(0, 0),
      tsd::math::int2(depth.width - 1, depth.height - 1));
  l.x = depth.width - l.x;
  l.y = depth.height - l.y;
  auto i = l.y * depth.width + l.x;

  m_pickedDepth = depth.pixelType == ANARI_FLOAT32
      ? depth.data[i]
      : std::numeric_limits<float>::max();
  anari::unmap(m_device, frame, "channel.depth");

  if (!selectObject)
    return;

  // Do object selection //

  uint32_t id = ~0u;

  auto fb = anari::map<uint32_t>(m_device, frame, "channel.objectId");
  if (fb.data) {
    id = fb.data[i];

    tsd::logStatus("[viewport] picked object '%u' @ (%i, %i) | z: %f",
        id,
        l.x,
        l.y,
        m_pickedDepth);

  } else {
    tsd::logError("[viewport] failed to pick: unable to map 'objectId' buffer");
  }

  anari::DataType objectType = ANARI_SURFACE;
  if (id != ~0u && id & 0x80000000u) {
    objectType = ANARI_VOLUME;
    id &= 0x7FFFFFFF;
  }

  m_core->setSelectedObject(
      id == ~0u ? nullptr : m_core->tsd.ctx.getObject(objectType, id));

  anari::unmap(m_device, frame, "channel.objectId");
}

void Viewport::setSelectionVisibilityFilterEnabled(bool enabled)
{
  if (!enabled)
    m_rIdx->setFilterFunction({});
  else {
    m_rIdx->setFilterFunction([&](const tsd::Object *obj) {
      return !m_core->tsd.selectedObject || obj == m_core->tsd.selectedObject;
    });
  }
}

void Viewport::updateFrame()
{
  if (!m_anariPass)
    return;

  m_rud.r = m_renderers[m_currentRenderer];
  if (m_cameraModel == CameraModel::Orthographic)
    m_anariPass->setCamera(m_orthoCamera);
  else if (m_cameraModel == CameraModel::Omnidirectional)
    m_anariPass->setCamera(m_omniCamera);
  else
    m_anariPass->setCamera(m_perspCamera);
  m_anariPass->setRenderer(m_rud.r);
  m_anariPass->setWorld(m_rIdx->world());
}

void Viewport::updateCamera(bool force)
{
  if (!m_anariPass)
    return;

  if ((!force && !m_arcball->hasChanged(m_cameraToken)))
    return;

  anari::setParameter(m_device, m_perspCamera, "position", m_arcball->eye());
  anari::setParameter(m_device, m_perspCamera, "direction", m_arcball->dir());
  anari::setParameter(m_device, m_perspCamera, "up", m_arcball->up());

  anari::setParameter(
      m_device, m_orthoCamera, "position", m_arcball->eye_FixedDistance());
  anari::setParameter(m_device, m_orthoCamera, "direction", m_arcball->dir());
  anari::setParameter(m_device, m_orthoCamera, "up", m_arcball->up());
  anari::setParameter(
      m_device, m_orthoCamera, "height", m_arcball->distance() * 0.75f);

  anari::setParameter(m_device, m_omniCamera, "position", m_arcball->eye());
  anari::setParameter(m_device, m_omniCamera, "direction", m_arcball->dir());
  anari::setParameter(m_device, m_omniCamera, "up", m_arcball->up());

  anari::setParameter(m_device,
      m_perspCamera,
      "aspect",
      m_viewportSize.x / float(m_viewportSize.y));
  anari::setParameter(m_device,
      m_orthoCamera,
      "aspect",
      m_viewportSize.x / float(m_viewportSize.y));
  anari::setParameter(
      m_device, m_perspCamera, "apertureRadius", m_apertureRadius);
  anari::setParameter(
      m_device, m_perspCamera, "focusDistance", m_focusDistance);

  anari::setParameter(m_device, m_perspCamera, "fovy", anari::radians(m_fov));

  anari::commitParameters(m_device, m_perspCamera);
  anari::commitParameters(m_device, m_orthoCamera);
  anari::commitParameters(m_device, m_omniCamera);

  if (m_echoCameraConfig)
    echoCameraConfig();
}

void Viewport::updateImage()
{
  if (!m_deviceReadyToUse)
    return;

  auto frame = m_anariPass->getFrame();
  anari::getProperty(
      m_device, frame, "numSamples", m_frameSamples, ANARI_NO_WAIT);

  const auto &tsd_ctx = m_core->tsd;
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

#if 0
  if (m_saveNextFrame) {
    std::string filename = "tsd_viewer-" + m_libName + '-'
        + std::to_string(g_screenshotIndex++) + ".png";
    stbi_write_png(filename.c_str(),
        color.width,
        color.height,
        4,
        color.data,
        4 * color.width);
    tsd::logStatus(
        "[viewport] frame saved to '%s'\n", filename.c_str());
    m_saveNextFrame = false;
  }
#endif
}

void Viewport::echoCameraConfig()
{
  const auto p = m_arcball->eye();
  const auto d = m_arcball->dir();
  const auto u = m_arcball->up();

  tsd::logStatus("Camera:");
  tsd::logStatus("  p: %f, %f, %f", p.x, p.y, p.z);
  tsd::logStatus("  d: %f, %f, %f", d.x, d.y, d.z);
  tsd::logStatus("  u: %f, %f, %f", u.x, u.y, u.z);
}

void Viewport::ui_handleInput()
{
  if (!m_deviceReadyToUse)
    return;

  ImGuiIO &io = ImGui::GetIO();

  if (ImGui::IsWindowFocused() && io.KeysDown[GLFW_KEY_ESCAPE])
    m_core->clearSelected();

  const bool dolly = ImGui::IsMouseDown(ImGuiMouseButton_Right)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && io.KeysDown[GLFW_KEY_LEFT_SHIFT]);
  const bool pan = ImGui::IsMouseDown(ImGuiMouseButton_Middle)
      || (ImGui::IsMouseDown(ImGuiMouseButton_Left)
          && io.KeysDown[GLFW_KEY_LEFT_ALT]);
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

void Viewport::ui_picking()
{
  const ImGuiIO &io = ImGui::GetIO();

  // Pick view center //

  const bool shouldPickCenter = m_cameraModel == CameraModel::Perspective
      && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)
      && io.KeysDown[GLFW_KEY_LEFT_SHIFT];
  if (shouldPickCenter && ImGui::IsWindowHovered()) {
    auto mPos = ImGui::GetMousePos();
    auto wMin = ImGui::GetItemRectMin();
    auto pixel = tsd::math::int2(
        tsd::math::float2(
            m_viewportSize.x - (mPos[0] - wMin[0]), mPos[1] - wMin[1])
        * m_resolutionScale);

    pick(pixel, false);

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
    const auto dir_dv =
        anari::math::normalize(anari::math::cross(dir_du, d)) * imgPlaneSize.y;
    const auto dir_00 = d - .5f * dir_du - .5f * dir_dv;

    const auto screen = anari::math::float2(
        1.f / m_viewportSize.x * pixel.x, (1.f / m_viewportSize.y * pixel.y));

    const auto dir =
        anari::math::normalize(dir_00 + screen.x * dir_du + screen.y * dir_dv);

    const auto p = m_arcball->eye();
    const auto c = p + m_pickedDepth * dir;

    tsd::logStatus(
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
    updateCamera();
  }

  // Pick object //

  const bool shouldPickObject =
      ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && !shouldPickCenter;
  if (shouldPickObject && ImGui::IsWindowHovered()) {
    auto mPos = ImGui::GetMousePos();
    auto wMin = ImGui::GetItemRectMin();
    auto pixel = tsd::math::float2(
                     m_viewportSize.x - (mPos[0] - wMin[0]), mPos[1] - wMin[1])
        * m_resolutionScale;
    pick(tsd::math::int2(pixel), true);
  }
}

void Viewport::ui_contextMenu()
{
  constexpr float INDENT_AMOUNT = 25.f;

  const ImGuiIO &io = ImGui::GetIO();

  const bool openMenu = ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Right)
      || io.KeysDown[GLFW_KEY_MENU];
  if (openMenu && ImGui::IsWindowHovered()) {
    m_coreMenuVisible = true;
    ImGui::OpenPopup(m_coreMenuName.c_str());
  }

  if (ImGui::BeginPopup(m_coreMenuName.c_str())) {
    // Device //
    ImGui::BeginDisabled(!m_core->tsd.sceneLoadComplete);

    ImGui::Text("Device:");
    ImGui::Indent(INDENT_AMOUNT);

    if (ImGui::BeginMenu("Device")) {
      for (auto &libName : m_core->commandLine.libraryList) {
        const bool isThisLibrary = m_libName == libName;
        if (ImGui::RadioButton(libName.c_str(), isThisLibrary))
          setLibrary(libName, false);
      }
      ImGui::EndMenu();
    }
    ImGui::Unindent(INDENT_AMOUNT);

    ImGui::EndDisabled();

    if (m_device) {
      ImGui::Separator();

      // Renderer //

      if (!m_rendererObjects.empty()) {
        ImGui::Text("Renderer:");
        ImGui::Indent(INDENT_AMOUNT);
        if (m_rendererObjects.size() > 1 && ImGui::BeginMenu("subtype")) {
          for (int i = 0; i < m_rendererObjects.size(); i++) {
            const char *rName = m_rendererObjects[i].name().c_str();
            if (ImGui::RadioButton(rName, &m_currentRenderer, i))
              updateFrame();
          }
          ImGui::EndMenu();
        }

        if (!m_rendererObjects.empty() && ImGui::BeginMenu("parameters")) {
          tsd::ui::buildUI_object(
              m_rendererObjects[m_currentRenderer], m_core->tsd.ctx, false);
          ImGui::EndMenu();
        }

        ImGui::Unindent(INDENT_AMOUNT);
        ImGui::Separator();
      }

      // Camera //

      ImGui::Text("Camera:");
      ImGui::Indent(INDENT_AMOUNT);

      if (ImGui::RadioButton(
              "perspective", &m_cameraModel, CameraModel::Perspective)
          || ImGui::RadioButton(
              "orthographic", &m_cameraModel, CameraModel::Orthographic)
          || ImGui::RadioButton(
              "omnidirectional", &m_cameraModel, CameraModel::Omnidirectional))
        updateFrame();

      ImGui::BeginDisabled(m_cameraModel != CameraModel::Perspective);

      if (ImGui::SliderFloat("fov", &m_fov, 0.1f, 180.f))
        updateCamera(true);

      if (ImGui::BeginMenu("DoF")) {
        if (ImGui::DragFloat("aperture", &m_apertureRadius, 0.01f, 0.f, 1.f))
          updateCamera(true);

        if (ImGui::DragFloat(
                "focus distance", &m_focusDistance, 0.1f, 0.f, 1e20f))
          updateCamera(true);
        ImGui::EndMenu();
      }

      ImGui::EndDisabled();

      if (ImGui::Combo("up", &m_arcballUp, "+x\0+y\0+z\0-x\0-y\0-z\0\0")) {
        m_arcball->setAxis(static_cast<manipulators::OrbitAxis>(m_arcballUp));
        resetView();
      }

      if (ImGui::MenuItem("reset view"))
        resetView();

      if (ImGui::Checkbox("echo config", &m_echoCameraConfig)
          && m_echoCameraConfig)
        echoCameraConfig();

      ImGui::Unindent(INDENT_AMOUNT);
      ImGui::Separator();

      // Viewport //

      ImGui::Text("Viewport:");
      ImGui::Indent(INDENT_AMOUNT);

      if (ImGui::BeginMenu("format")) {
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

        ImGui::EndMenu();
      }

      ImGui::Separator();

      if (ImGui::BeginMenu("render resolution")) {
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
        ImGui::EndMenu();
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
      ImGui::Checkbox("highlight selection", &m_highlightSelection);
      ImGui::EndDisabled();

      if (ImGui::Checkbox("only show selection", &m_showOnlySelected))
        setSelectionVisibilityFilterEnabled(m_showOnlySelected);

      ImGui::Separator();

      ImGui::Checkbox("show stats", &m_showOverlay);
      if (ImGui::MenuItem("reset stats")) {
        m_minFL = m_latestFL;
        m_maxFL = m_latestFL;
      }

      ImGui::Separator();

      if (ImGui::MenuItem("take screenshot"))
        m_saveNextFrame = true;

      ImGui::Unindent(INDENT_AMOUNT);
      ImGui::Separator();

      // World //

      ImGui::Text("World:");
      ImGui::Indent(INDENT_AMOUNT);

      if (ImGui::MenuItem("print bounds")) {
        tsd::math::float3 bounds[2];

        anariGetProperty(m_device,
            m_rIdx->world(),
            "bounds",
            ANARI_FLOAT32_BOX3,
            &bounds[0],
            sizeof(bounds),
            ANARI_WAIT);

        tsd::logStatus(
            "[viewport] current world bounds {%f, %f, %f} x {%f, %f, %f}\n",
            bounds[0].x,
            bounds[0].y,
            bounds[0].z,
            bounds[1].x,
            bounds[1].y,
            bounds[1].z);
      }

      ImGui::Unindent(INDENT_AMOUNT);
    }

    if (!ImGui::IsPopupOpen(m_coreMenuName.c_str()))
      m_coreMenuVisible = false;

    ImGui::EndPopup();
  }
}

void Viewport::ui_overlay()
{
  ImGuiIO &io = ImGui::GetIO();
  ImVec2 windowPos = ImGui::GetWindowPos();
  windowPos.x += 10;
  windowPos.y += 35 * io.FontGlobalScale;

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

    ImGui::Checkbox("camera info", &m_showCameraInfo);
    if (m_showCameraInfo) {
      auto at = m_arcball->at();
      auto azel = m_arcball->azel();
      auto dist = m_arcball->distance();

      bool update = ImGui::SliderFloat("az", &azel.x, 0.f, 360.f);
      update |= ImGui::SliderFloat("el", &azel.y, 0.f, 360.f);
      update |= ImGui::DragFloat("dist", &dist);
      update |= ImGui::DragFloat3("at", &at.x);

      if (update)
        m_arcball->setConfig(at, dist, azel);
    }

    ImGui::End();
  }
}

void Viewport::RendererUpdateDelegate::signalParameterUpdated(
    const tsd::Object *o, const tsd::Parameter *p)
{
  if (d && r) {
    o->updateANARIParameter(d, r, *p, p->name().c_str());
    anari::commitParameters(d, r);
  }
}

} // namespace tsd_viewer
