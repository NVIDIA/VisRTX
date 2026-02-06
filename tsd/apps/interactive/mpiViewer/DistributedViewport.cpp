// Copyright 2024-2026 NVIDIA Corporation
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

namespace tsd::mpi_viewer {

constexpr float INDENT_AMOUNT = 25.f;

DistributedViewport::DistributedViewport(tsd::ui::imgui::Application *app,
    DistributedSceneController *dapp,
    const char *name)
    : Window(app, name), m_dapp(dapp)
{
  setManipulator(nullptr);
  resetView(true);
  m_overlayWindowName = "overlay_";
  m_overlayWindowName += name;
  m_coreMenuName = "vpContextMenu_";
  m_coreMenuName += name;
}

DistributedViewport::~DistributedViewport()
{
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
  m_dapp->executeFrame();
  updateImage();

  if (m_showOverlay)
    ui_overlay();

  if (m_showTimeline)
    ui_timeControls();

  ui_handleInput();
  ui_menuBar();
}

void DistributedViewport::setManipulator(tsd::rendering::Manipulator *m)
{
  m_arcball = m ? m : &m_localArcball;
}

void DistributedViewport::resetView(bool resetAzEl)
{
  tsd::math::float3 bounds[2] = {{-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}};

#if 0
  if (!anariGetProperty(m_device,
          m_world,
          "bounds",
          ANARI_FLOAT32_BOX3,
          &bounds[0],
          sizeof(bounds),
          ANARI_WAIT)) {
    tsd::core::logWarning("No bounds returned by the ANARIWorld!");
  }
#endif

  auto center = 0.5f * (bounds[0] + bounds[1]);
  auto diag = bounds[1] - bounds[0];

  auto azel = resetAzEl ? tsd::math::float2(0.f, 20.f) : m_arcball->azel();
  m_arcball->setConfig(center, 1.25f * linalg::length(diag), azel);
  m_cameraToken = 0;
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

  auto *frameState = m_dapp->distributedState()->frame.write();
  frameState->size = m_renderSize;

  updateCamera(true);
}

void DistributedViewport::updateCamera(bool force)
{
  if (!force && !m_arcball->hasChanged(m_cameraToken))
    return;

  auto *c = m_dapp->distributedState()->camera.write();
  c->position = m_arcball->eye();
  c->direction = m_arcball->dir();
  c->up = m_arcball->up();
  c->fovy = tsd::math::radians(m_fov);
  c->aspect = m_viewportSize.x / float(m_viewportSize.y);
  c->apertureRadius = m_apertureRadius;
  c->focusDistance = m_focusDistance;
}

void DistributedViewport::updateImage()
{
  auto d = m_dapp->anariDevice();
  auto f = m_dapp->anariFrame();

  float duration = 0.f;
  anari::getProperty(d, f, "duration", duration);

  m_latestFL = duration * 1000;
  m_minFL = std::min(m_minFL, m_latestFL);
  m_maxFL = std::max(m_maxFL, m_latestFL);

  auto fb = anari::map<uint32_t>(d, f, "channel.color");

  if (fb.data) {
    SDL_UpdateTexture(m_framebufferTexture,
        nullptr,
        fb.data,
        fb.width * anari::sizeOf(ANARI_UFIXED8_RGBA_SRGB));
  } else {
    tsd::core::logError(
        "mapped bad frame: %p | %i x %i\n", fb.data, fb.width, fb.height);
  }

  if (m_saveNextFrame) {
    stbi_flip_vertically_on_write(1);
    std::string filename =
        "screenshot" + std::to_string(m_screenshotIndex++) + ".png";
    stbi_write_png(
        filename.c_str(), fb.width, fb.height, 4, fb.data, 4 * fb.width);
    printf("frame saved to '%s'\n", filename.c_str());
    m_saveNextFrame = false;
  }

  anari::unmap(d, f, "channel.color");

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

void DistributedViewport::ui_menuBar()
{
  if (ImGui::BeginMenuBar()) {
    if (ImGui::BeginMenu("Renderer")) {
      bool changed = false;
      changed |=
          ImGui::ColorEdit4("background", &m_localState.renderer.background.x);
      changed |= ImGui::ColorEdit3(
          "ambientColor", &m_localState.renderer.ambientColor.x);
      changed |= ImGui::DragFloat("ambientRadiance",
          &m_localState.renderer.ambientRadiance,
          0.005f,
          0.f,
          100.f);
      changed |= ImGui::Checkbox("denoise", &m_localState.renderer.denoise);
      if (changed)
        *m_dapp->distributedState()->renderer.write() = m_localState.renderer;
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Camera")) {
      auto at = m_arcball->at();
      auto azel = m_arcball->azel();
      auto dist = m_arcball->distance();

      bool update = ImGui::SliderFloat("az", &azel.x, 0.f, 360.f);
      update |= ImGui::SliderFloat("el", &azel.y, 0.f, 360.f);
      update |= ImGui::DragFloat("dist", &dist);

      if (update)
        m_arcball->setConfig(at, dist, azel);

      if (ImGui::Combo("up", &m_arcballUp, "+x\0+y\0+z\0-x\0-y\0-z\0\0")) {
        m_arcball->setAxis(static_cast<tsd::rendering::UpAxis>(m_arcballUp));
        resetView();
      }

      if (ImGui::SliderFloat("fov", &m_fov, 0.1f, 180.f))
        updateCamera(true);

      ImGui::Separator();

      if (ImGui::BeginMenu("DoF")) {
        if (ImGui::DragFloat("aperture", &m_apertureRadius, 0.01f, 0.f, 1.f))
          updateCamera(true);

        if (ImGui::DragFloat(
                "focus distance", &m_focusDistance, 0.1f, 0.f, 1e20f))
          updateCamera(true);
        ImGui::EndMenu();
      }

      ImGui::Separator();

      float zoomSpeed = m_arcball->zoomSpeed();
      if (ImGui::DragFloat("zoom speed", &zoomSpeed, 0.01f, 0.01f, 1e20f))
        m_arcball->setZoomSpeed(zoomSpeed);

      ImGui::Separator();

      if (ImGui::MenuItem("reset view"))
        resetView();

      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Viewport")) {
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

      ImGui::Checkbox("show timeline", &m_showTimeline);
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

    if (ImGui::BeginMenu("World")) {
      if (ImGui::MenuItem("print bounds")) {
        tsd::core::logWarning("TODO: implement world bounds printout\n");
#if 0
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
#endif
      }

      ImGui::EndMenu();
    }

    ImGui::EndMenuBar();
  }
}

void DistributedViewport::ui_overlay()
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
    ImGui::Text(" library: %s", m_dapp->anariLibraryName());
    ImGui::Separator();
    ImGui::Text("viewport: %i x %i", m_viewportSize.x, m_viewportSize.y);
    ImGui::Text("  render: %i x %i", m_renderSize.x, m_renderSize.y);
    ImGui::Separator();
    ImGui::Text(" latency: %.2fms", m_latestFL);
    ImGui::Text("   (min): %.2fms", m_minFL);
    ImGui::Text("   (max): %.2fms", m_maxFL);
    ImGui::End();
  }
}

void DistributedViewport::ui_timeControls()
{
  ImVec2 windowPos = ImGui::GetWindowPos();
  float windowHeight = ImGui::GetWindowHeight();
  float windowWidth = ImGui::GetWindowWidth();
  windowPos.x += windowWidth * 0.1f;
  windowPos.y += windowHeight - 50;

  ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always);
  ImGui::SetNextWindowBgAlpha(0.25f);
  ImGui::SetNextWindowSize(ImVec2(windowWidth * 0.8f, 0));

  ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration
      | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize
      | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
      | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;

  ImGui::PushID(this);
  if (ImGui::Begin("timeline", nullptr, window_flags)) {
    bool update = false;

    ImGuiStyle &style = ImGui::GetStyle();

    ImGui::SetNextItemWidth(
        ImGui::GetWindowWidth() - style.WindowPadding.x * 2);
    update |= ImGui::SliderFloat(
        "##time", &m_localState.animation.time, 0.f, 1.f, "");

    if (update)
      *m_dapp->distributedState()->animation.write() = m_localState.animation;

    ImGui::End();
  }
  ImGui::PopID();
}

int DistributedViewport::windowFlags() const
{
  return ImGuiWindowFlags_MenuBar;
}

} // namespace tsd::mpi_viewer
