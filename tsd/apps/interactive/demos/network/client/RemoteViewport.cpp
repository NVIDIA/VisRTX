// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RemoteViewport.h"
// tsd_ui_imgui
#include "imgui.h"
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/tsd_ui_imgui.h"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"
// std
#include <algorithm>

namespace tsd::ui::imgui {

RemoteViewport::RemoteViewport(Application *app,
    tsd::rendering::Manipulator *m,
    tsd::network::NetworkChannel *c,
    const char *name)
    : Window(app, name)
{
  setManipulator(m);
  setNetworkChannel(c);
  setupRenderPipeline();
}

RemoteViewport::~RemoteViewport()
{
  m_channel->removeAllHandlers();
}

void RemoteViewport::buildUI()
{
  ImVec2 _viewportSize = ImGui::GetContentRegionAvail();
  tsd::math::int2 viewportSize(_viewportSize.x, _viewportSize.y);

  bool isConnected = m_channel && m_channel->isConnected();

  if (m_viewportSize != viewportSize || m_wasConnected != isConnected)
    reshape(viewportSize);

  if (!m_wasConnected && isConnected) {
    m_channel->send(
        tsd::network::make_message(MessageType::SERVER_REQUEST_VIEW));
  }

  m_wasConnected = isConnected;
  m_incomingFramePass->setEnabled(isConnected);

  updateCamera();
  m_pipeline.render();

  ui_menubar();

  if (m_outputPass) {
    ImGui::Image((ImTextureID)m_outputPass->getTexture(),
        ImGui::GetContentRegionAvail(),
        ImVec2(0, 1),
        ImVec2(1, 0));
  }

  ui_handleInput();

  // Render the overlay after input handling so it does not interfere.
  if (m_showOverlay)
    ui_overlay();
}

void RemoteViewport::setManipulator(tsd::rendering::Manipulator *m)
{
  m_arcball = m ? m : &m_localArcball;
}

void RemoteViewport::setNetworkChannel(tsd::network::NetworkChannel *c)
{
  m_channel = c;

  c->registerHandler(MessageType::CLIENT_RECEIVE_FRAME_BUFFER_COLOR,
      [&](const tsd::network::Message &msg) {
        if (msg.header.payload_length != m_incomingColorBuffer.size()) {
          tsd::core::logWarning(
              "[Client] Received color buffer size does not match current"
              " viewport size");
          return;
        }
        std::memcpy(m_incomingColorBuffer.data(),
            msg.payload.data(),
            msg.header.payload_length);
      });
}

void RemoteViewport::saveSettings(tsd::core::DataNode &root)
{
  root.reset(); // clear all previous values, if they exist

  // Base window settings //

  Window::saveSettings(root);
}

void RemoteViewport::loadSettings(tsd::core::DataNode &root)
{
  Window::loadSettings(root);
}

void RemoteViewport::setupRenderPipeline()
{
  m_clearPass = m_pipeline.emplace_back<tsd::rendering::ClearBuffersPass>();
  m_incomingFramePass =
      m_pipeline.emplace_back<tsd::rendering::CopyToColorBufferPass>();
  m_outputPass = m_pipeline.emplace_back<tsd::rendering::CopyToSDLTexturePass>(
      m_app->sdlRenderer());

  m_clearPass->setClearColor(tsd::math::float4(1.f, 0.f, 0.f, 1.f));
  m_incomingFramePass->setExternalBuffer(m_incomingColorBuffer);

  reshape(m_viewportSize);
}

void RemoteViewport::reshape(tsd::math::int2 newSize)
{
  if (newSize.x <= 0 || newSize.y <= 0)
    return;

  m_viewportSize = newSize;
  m_pipeline.setDimensions(newSize.x, newSize.y);
  m_incomingColorBuffer.resize(newSize.x * newSize.y * 4);
  std::fill(m_incomingColorBuffer.begin(), m_incomingColorBuffer.end(), 0);

  if (m_channel && m_channel->isConnected()) {
    tsd::network::RenderSession::Frame::Config frameConfig;
    frameConfig.size = tsd::math::uint2(newSize.x, newSize.y);
    m_channel->send(tsd::network::make_message(
        MessageType::SERVER_SET_FRAME_CONFIG, &frameConfig));
  }
}

void RemoteViewport::updateCamera()
{
  if (!m_channel || !m_channel->isConnected())
    return;

  if (m_arcball->hasChanged(m_cameraToken)) {
    tsd::network::RenderSession::View viewMsg;
    viewMsg.azeldist.x = m_arcball->azel().x;
    viewMsg.azeldist.y = m_arcball->azel().y;
    viewMsg.azeldist.z = m_arcball->distance();
    viewMsg.lookat = m_arcball->at();

    m_channel->send(
        tsd::network::make_message(MessageType::SERVER_SET_VIEW, &viewMsg));
  }
}

void RemoteViewport::ui_menubar()
{
  if (ImGui::BeginMenuBar()) {
    // Viewport //

    if (ImGui::BeginMenu("Viewport")) {
      ImGui::Checkbox("show info overlay", &m_showOverlay);
      ImGui::EndMenu();
    }

    ImGui::EndMenuBar();
  }
}

void RemoteViewport::ui_handleInput()
{
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

void RemoteViewport::ui_overlay()
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
    ImGui::Text("viewport: %i x %i", m_viewportSize.x, m_viewportSize.y);

#if 0
    ImGui::Text(" display: %.2fms", m_latestFL);
    ImGui::Text("   ANARI: %.2fms", m_latestAnariFL);
    ImGui::Text("   (min): %.2fms", m_minFL);
    ImGui::Text("   (max): %.2fms", m_maxFL);
#endif

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

      if (update) {
        m_arcball->setConfig(at, dist, azel);
        m_arcball->setFixedDistance(fixedDist);
      }
    }
  }
  ImGui::EndChild();

  ImGui::PopStyleColor();
}

int RemoteViewport::windowFlags() const
{
  return ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoScrollbar;
}

} // namespace tsd::ui::imgui
