// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "DistributedViewport.h"
// std
#include <cstring>
// stb_image
#include "stb_image_write.h"
// tsd_core
#include <tsd/core/Logging.hpp>
// tsd_ui_imgui
#include <tsd/ui/imgui/tsd_ui_imgui.h>

namespace tsd::ptc {

DistributedViewport::DistributedViewport(tsd::ui::imgui::Application *app,
    RemoteAppStateWindow *win,
    const char *rendererSubtype,
    const char *name)
    : Window(app, name), m_win(win)
{
  stbi_flip_vertically_on_write(1);
  m_rendererSubtype = rendererSubtype;
  m_rendererObject =
      tsd::core::Object(ANARI_RENDERER, m_rendererSubtype.c_str());
  setManipulator(nullptr);
  m_overlayWindowName = "overlay_";
  m_overlayWindowName += name;
  m_coreMenuName = "vpContextMenu_";
  m_coreMenuName += name;
}

DistributedViewport::~DistributedViewport()
{
  m_win->fence();
  teardownDevice();
  if (m_framebufferTexture)
    SDL_DestroyTexture(m_framebufferTexture);
}

void DistributedViewport::buildUI()
{
  ImVec2 _viewportSize = ImGui::GetContentRegionAvail();
  tsd::math::int2 viewportSize(_viewportSize.x, _viewportSize.y);

  if (m_viewportSize != viewportSize)
    reshape(viewportSize);

  updateCamera();

  m_win->fence();

  // Allow workers to get the latest frame/camera/renderer params //

  m_win->fence();

  anari::render(m_device, m_frame);
  anari::wait(m_device, m_frame);

  updateImage();

  if (m_showOverlay)
    ui_overlay();

  ui_contextMenu();

  if (!m_coreMenuVisible)
    ui_handleInput();
}

void DistributedViewport::setManipulator(tsd::rendering::Manipulator *m)
{
  m_arcball = m ? m : &m_localArcball;
}

void DistributedViewport::setWorld(anari::World world, bool resetCameraView)
{
  if (m_world)
    anari::release(m_device, m_world);

  if (!world) {
    world = anari::newObject<anari::World>(m_device);
    resetCameraView = false;
  } else
    anari::retain(m_device, world);

  anari::commitParameters(m_device, world);
  m_world = world;

  if (resetCameraView)
    resetView();

  updateFrame();
}

void DistributedViewport::resetView(bool resetAzEl)
{
  tsd::math::float3 bounds[2] = {{-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}};

  if (!anariGetProperty(m_device,
          m_world,
          "bounds",
          ANARI_FLOAT32_BOX3,
          &bounds[0],
          sizeof(bounds),
          ANARI_WAIT)) {
    tsd::core::logWarning("No bounds returned by the ANARIWorld!");
  }

  auto center = 0.5f * (bounds[0] + bounds[1]);
  auto diag = bounds[1] - bounds[0];

  auto azel = resetAzEl ? tsd::math::float2(0.f, 20.f) : m_arcball->azel();
  m_arcball->setConfig(center, 1.25f * linalg::length(diag), azel);
  m_cameraToken = 0;
}

void DistributedViewport::setDevice(anari::Device d)
{
  teardownDevice();

  m_rud.d = m_device = d;
  m_rud.version = &m_win->ptr()->renderer.version;

  anari::retain(d, d);

  m_frame = anari::newObject<anari::Frame>(d);
  m_camera = anari::newObject<anari::Camera>(d, "perspective");
  m_renderer = anari::newObject<anari::Renderer>(d, m_rendererSubtype.c_str());

  {
    auto &o = m_rendererObject;
    o = tsd::core::parseANARIObjectInfo(d, ANARI_RENDERER, "default");
    o.setName("renderer");
    o.setUpdateDelegate(&m_rud);
    o.updateAllANARIParameters(d, m_renderer);
    anari::commitParameters(d, m_renderer);
    (*m_rud.version)++;
  }

  reshape(m_viewportSize);
  updateFrame();
  updateCamera(true);
}

void DistributedViewport::teardownDevice()
{
  if (!m_device)
    return;

  anari::discard(m_device, m_frame);
  anari::wait(m_device, m_frame);

  anari::release(m_device, m_camera);
  anari::release(m_device, m_renderer);
  anari::release(m_device, m_world);
  anari::release(m_device, m_frame);
  anari::release(m_device, m_device);

  m_camera = nullptr;
  m_renderer = nullptr;
  m_world = nullptr;
  m_frame = nullptr;
  m_device = nullptr;
}

void DistributedViewport::reshape(tsd::math::int2 newSize)
{
  if (newSize.x <= 0 || newSize.y <= 0)
    return;

  m_viewportSize = newSize;
  m_renderSize =
      tsd::math::int2(tsd::math::float2(m_viewportSize) * m_resolutionScale);

  if (m_framebufferTexture)
    SDL_DestroyTexture(m_framebufferTexture);

  m_framebufferTexture = SDL_CreateTexture(m_app->sdlRenderer(),
      SDL_PIXELFORMAT_RGBA32,
      SDL_TEXTUREACCESS_STREAMING,
      newSize.x,
      newSize.y);

  updateFrame();
  updateCamera(true);
}

void DistributedViewport::updateFrame()
{
  auto &f = m_win->ptr()->frame;
  f.size = m_renderSize;
  f.version++;

  m_rud.r = m_renderer;

  anari::setParameter(
      m_device, m_frame, "size", tsd::math::uint2(m_renderSize));
  anari::setParameter(
      m_device, m_frame, "channel.color", ANARI_UFIXED8_RGBA_SRGB);
  anari::setParameter(m_device, m_frame, "channel.objectId", ANARI_UINT32);
  anari::setParameter(m_device, m_frame, "accumulation", true);
  anari::setParameter(m_device, m_frame, "world", m_world);
  anari::setParameter(m_device, m_frame, "camera", m_camera);
  anari::setParameter(m_device, m_frame, "renderer", m_rud.r);

  anari::commitParameters(m_device, m_frame);
}

void DistributedViewport::updateCamera(bool force)
{
  if (!force && !m_arcball->hasChanged(m_cameraToken))
    return;

  auto &c = m_win->ptr()->camera;
  c.position = m_arcball->eye();
  c.direction = m_arcball->dir();
  c.up = m_arcball->up();
  c.aspect = m_viewportSize.x / float(m_viewportSize.y);
  c.fovy = tsd::math::radians(m_fov);
  c.apertureRadius = m_apertureRadius;
  c.focusDistance = m_focusDistance;
  c.version++;

  anari::setParameter(m_device, m_camera, "position", c.position);
  anari::setParameter(m_device, m_camera, "direction", c.direction);
  anari::setParameter(m_device, m_camera, "up", c.up);
  anari::setParameter(m_device, m_camera, "aspect", c.aspect);
  anari::setParameter(m_device, m_camera, "fovy", c.fovy);
  anari::setParameter(m_device, m_camera, "apertureRadius", m_apertureRadius);
  anari::setParameter(m_device, m_camera, "focusDistance", m_focusDistance);

  anari::commitParameters(m_device, m_camera);
}

void DistributedViewport::updateImage()
{
  float duration = 0.f;
  anari::getProperty(m_device, m_frame, "duration", duration);

  m_latestFL = duration * 1000;
  m_minFL = std::min(m_minFL, m_latestFL);
  m_maxFL = std::max(m_maxFL, m_latestFL);

  auto fb = anari::map<uint32_t>(m_device, m_frame, "channel.color");

  if (fb.data) {
    SDL_UpdateTexture(m_framebufferTexture,
        nullptr,
        fb.data,
        fb.width * anari::sizeOf(ANARI_UFIXED8_RGBA_SRGB));
  } else {
    printf("mapped bad frame: %p | %i x %i\n", fb.data, fb.width, fb.height);
  }

  if (m_saveNextFrame) {
    std::string filename =
        "screenshot" + std::to_string(m_screenshotIndex++) + ".png";
    stbi_write_png(
        filename.c_str(), fb.width, fb.height, 4, fb.data, 4 * fb.width);
    printf("frame saved to '%s'\n", filename.c_str());
    m_saveNextFrame = false;
  }

  anari::unmap(m_device, m_frame, "channel.color");

  ImGui::Image((ImTextureID)m_framebufferTexture,
      ImGui::GetContentRegionAvail(),
      ImVec2(0, 1),
      ImVec2(1, 0));
}

void DistributedViewport::ui_handleInput()
{
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

void DistributedViewport::ui_contextMenu()
{
  constexpr float INDENT_AMOUNT = 25.f;

  ImGuiIO &io = ImGui::GetIO();
  const bool rightClicked = ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Right)
      || ImGui::IsKeyDown(ImGuiKey_Menu);

  if (rightClicked && ImGui::IsWindowHovered()) {
    m_coreMenuVisible = true;
    ImGui::OpenPopup(m_coreMenuName.c_str());
  }

  if (ImGui::BeginPopup(m_coreMenuName.c_str())) {
    ImGui::Text("Renderer:");
    ImGui::Indent(INDENT_AMOUNT);
    if (ImGui::BeginMenu("parameters")) {
      tsd::ui::buildUI_object(m_rendererObject, appCore()->tsd.ctx, false);
      ImGui::EndMenu();
    }

    ImGui::Unindent(INDENT_AMOUNT);
    ImGui::Separator();

    ImGui::Text("Camera:");
    ImGui::Indent(INDENT_AMOUNT);

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

    if (ImGui::Combo("up", &m_arcballUp, "+x\0+y\0+z\0-x\0-y\0-z\0\0")) {
      m_arcball->setAxis(static_cast<tsd::rendering::UpAxis>(m_arcballUp));
      resetView();
    }

    if (ImGui::MenuItem("reset view"))
      resetView();

    ImGui::Unindent(INDENT_AMOUNT);
    ImGui::Separator();

    ImGui::Text("DistributedViewport:");
    ImGui::Indent(INDENT_AMOUNT);

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

    ImGui::Checkbox("show stats", &m_showOverlay);
    if (ImGui::MenuItem("reset stats")) {
      m_minFL = m_latestFL;
      m_maxFL = m_latestFL;
    }

    if (ImGui::MenuItem("take screenshot"))
      m_saveNextFrame = true;

    ImGui::Unindent(INDENT_AMOUNT);
    ImGui::Separator();

    ImGui::Text("World:");
    ImGui::Indent(INDENT_AMOUNT);

    if (ImGui::MenuItem("print bounds")) {
      tsd::math::float3 bounds[2];

      anariGetProperty(m_device,
          m_world,
          "bounds",
          ANARI_FLOAT32_BOX3,
          &bounds[0],
          sizeof(bounds),
          ANARI_WAIT);

      printf("current world bounds {%f, %f, %f} x {%f, %f, %f}\n",
          bounds[0].x,
          bounds[0].y,
          bounds[0].z,
          bounds[1].x,
          bounds[1].y,
          bounds[1].z);
    }

    ImGui::Unindent(INDENT_AMOUNT);

    if (!ImGui::IsPopupOpen(m_coreMenuName.c_str()))
      m_coreMenuVisible = false;

    ImGui::EndPopup();
  }
}

void DistributedViewport::ui_overlay()
{
  ImGuiIO &io = ImGui::GetIO();
  ImVec2 windowPos = ImGui::GetWindowPos();
  windowPos.x += 10;
  windowPos.y += 25 * io.FontGlobalScale;

  ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration
      | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize
      | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
      | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;

  ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always);

  if (ImGui::Begin(m_overlayWindowName.c_str(), nullptr, window_flags)) {
    ImGui::Text("viewport: %i x %i", m_viewportSize.x, m_viewportSize.y);
    ImGui::Text("  render: %i x %i", m_renderSize.x, m_renderSize.y);
    ImGui::Text(" latency: %.2fms", m_latestFL);
    ImGui::Text("   (min): %.2fms", m_minFL);
    ImGui::Text("   (max): %.2fms", m_maxFL);

    ImGui::Separator();

    static bool showCameraInfo = false;

    ImGui::Checkbox("camera info", &showCameraInfo);

    if (showCameraInfo) {
      auto at = m_arcball->at();
      auto azel = m_arcball->azel();
      auto dist = m_arcball->distance();

      bool update = ImGui::SliderFloat("az", &azel.x, 0.f, 360.f);
      update |= ImGui::SliderFloat("el", &azel.y, 0.f, 360.f);
      update |= ImGui::DragFloat("dist", &dist);

      if (update)
        m_arcball->setConfig(at, dist, azel);
    }

    ImGui::End();
  }
}

void DistributedViewport::RendererUpdateDelegate::signalParameterUpdated(
    const tsd::core::Object *o, const tsd::core::Parameter *p)
{
  o->updateANARIParameter(d, r, *p, p->name().c_str());
  anari::commitParameters(d, r);
  (*version)++;
}

} // namespace tsd::ptc
