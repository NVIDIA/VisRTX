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
// tsd_rendering
#include "tsd/rendering/view/ManipulatorToTSD.hpp"
// std
#include <algorithm>

namespace tsd::ui::imgui {

RemoteViewport::RemoteViewport(Application *app,
    tsd::rendering::Manipulator *m,
    tsd::network::NetworkChannel *c,
    const char *name)
    : BaseViewport(app, name)
{
  disconnect();
  setManipulator(m);
  setNetworkChannel(c);
  BaseViewport::imagePipeline_setup();
}

RemoteViewport::~RemoteViewport()
{
  disconnect();

  if (m_channel)
    m_channel->removeAllHandlers();

  BaseViewport::imagePipeline_teardown();
}

void RemoteViewport::buildUI()
{
  BaseViewport::buildUI();

  ImVec2 _viewportSize = ImGui::GetContentRegionAvail();
  tsd::math::int2 viewportSize(_viewportSize.x, _viewportSize.y);

  bool isConnected = m_channel && m_channel->isConnected();
  BaseViewport::viewport_setActive(isConnected);

  if (m_viewport.size != viewportSize || m_wasConnected != isConnected)
    reshape(viewportSize);

  if (!m_wasConnected && isConnected) {
    m_receivedCameraIdx = TSD_INVALID_INDEX;
    m_receivedRendererIdx = TSD_INVALID_INDEX;
    m_channel->send(MessageType::SERVER_REQUEST_CURRENT_CAMERA);
    m_channel->send(MessageType::SERVER_REQUEST_CURRENT_RENDERER);
  } else if (m_wasConnected && !isConnected) {
    disconnect();
  }

  m_wasConnected = isConnected;
  m_incomingFramePass->setEnabled(isConnected);

  updateRenderer();
  updateCamera();
  BaseViewport::imagePipeline_render();

  ImGui::BeginDisabled(!isConnected);
  ui_menubar();
  ImGui::EndDisabled();

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

  // Render the overlay after input handling so it does not interfere.
  if (m_showOverlay)
    ui_overlay();
  BaseViewport::ui_animationSlider();
}

void RemoteViewport::setManipulator(tsd::rendering::Manipulator *m)
{
  m_camera.arcball = m ? m : &m_camera.localArcball;
}

void RemoteViewport::setNetworkChannel(tsd::network::NetworkChannel *c)
{
  m_channel = c;

  if (!c)
    return;

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

  c->registerHandler(MessageType::CLIENT_RECEIVE_CURRENT_RENDERER,
      [this](const tsd::network::Message &msg) {
        auto rendererIdx = *tsd::network::payloadAs<size_t>(msg);
        tsd::core::logDebug(
            "[Client] Received current renderer index %zu from server.",
            rendererIdx);
        m_receivedRendererIdx = rendererIdx;
      });

  c->registerHandler(MessageType::CLIENT_RECEIVE_CURRENT_CAMERA,
      [this](const tsd::network::Message &msg) {
        auto cameraIdx = *tsd::network::payloadAs<size_t>(msg);
        tsd::core::logDebug(
            "[Client] Received current camera index %zu from server.",
            cameraIdx);
        m_receivedCameraIdx = cameraIdx;
      });
}

void RemoteViewport::disconnect()
{
  m_receivedRendererIdx = TSD_INVALID_INDEX;
  m_renderers.current = {};
  m_renderers.objects.clear();
  m_prevRenderer = {};
  m_receivedCameraIdx = TSD_INVALID_INDEX;
  m_camera.current = {};
  m_prevCamera = {};
}

void RemoteViewport::imagePipeline_populate(tsd::rendering::ImagePipeline &p)
{
  m_clearPass = p.emplace_back<tsd::rendering::ClearBuffersPass>();
  m_incomingFramePass = p.emplace_back<tsd::rendering::CopyToColorBufferPass>();
  m_outputPass = p.emplace_back<tsd::rendering::CopyToSDLTexturePass>(
      m_app->sdlRenderer());

  m_clearPass->setClearColor(tsd::math::float4(1.f, 0.f, 0.f, 1.f));
  m_incomingFramePass->setExternalBuffer(m_incomingColorBuffer);

  reshape(m_viewport.size);
}

void RemoteViewport::camera_resetView(bool /*resetAzEl*/)
{
  tsd::core::logWarning(
      "Camera view reset is not currently supported in RemoteViewport.");
}

void RemoteViewport::camera_centerView()
{
  tsd::core::logWarning(
      "Camera center view is not currently supported in RemoteViewport.");
}

void RemoteViewport::renderer_resetParameterDefaults()
{
  if (!m_renderers.current)
    return;

  tsd::core::logWarning(
      "Renderer parameter reset is not currently supported in RemoteViewport.");
}

void RemoteViewport::reshape(tsd::math::int2 newSize)
{
  if (newSize.x <= 0 || newSize.y <= 0)
    return;

  m_incomingColorBuffer.resize(newSize.x * newSize.y * 4);
  std::fill(m_incomingColorBuffer.begin(), m_incomingColorBuffer.end(), 0);

  if (m_channel && m_channel->isConnected()) {
    tsd::network::RenderSession::Frame::Config frameConfig;
    frameConfig.size = tsd::math::uint2(newSize.x, newSize.y);
    m_channel->send(MessageType::SERVER_SET_FRAME_CONFIG, &frameConfig);
  }
}

void RemoteViewport::updateRenderer()
{
  if (!m_channel || !m_channel->isConnected())
    return;

  auto *ctx = appContext();
  auto &scene = ctx->tsd.scene;
  if (m_renderers.objects.empty()
      && scene.numberOfObjects(ANARI_RENDERER) != 0) {
    auto fr = scene.getObject<tsd::scene::Renderer>(0);
    m_renderers.objects = scene.renderersOfDevice(fr->rendererDeviceName());
  } else if (m_renderers.objects.empty()) {
    // We only update when we have the list of renderers from the server
    return;
  }

  const bool rendererFromServer =
      !m_renderers.current && m_receivedRendererIdx != TSD_INVALID_INDEX;
  const bool rendererChanged = m_renderers.current != m_prevRenderer;

  if (rendererFromServer) {
    m_renderers.current = m_receivedRendererIdx >= m_renderers.objects.size()
        ? tsd::scene::RendererAppRef{}
        : m_renderers.objects[m_receivedRendererIdx];
    m_prevRenderer = m_renderers.current;
  } else if (rendererChanged) {
    auto currentIdx =
        m_renderers.current ? m_renderers.current->index() : TSD_INVALID_INDEX;
    m_receivedRendererIdx = currentIdx;
    m_channel->send(MessageType::SERVER_SET_CURRENT_RENDERER, &currentIdx);
    m_prevRenderer = m_renderers.current;
  }
}

void RemoteViewport::updateCamera()
{
  if (!m_channel || !m_channel->isConnected())
    return;

  const bool cameraFromServer =
      !m_camera.current && m_receivedCameraIdx != TSD_INVALID_INDEX;
  const bool cameraChanged =
      m_camera.current && m_camera.current != m_prevCamera;

  if (cameraFromServer) {
    auto *ctx = appContext();
    auto &scene = ctx->tsd.scene;
    camera_setCurrent(scene.getObject<tsd::scene::Camera>(m_receivedCameraIdx));
    m_camera.arcballToken = {};
    m_manipulatorSynchronized = false;
    return;
  } else if (cameraChanged) {
    auto currentIdx = m_camera.current->index();
    m_receivedCameraIdx = currentIdx;
    m_channel->send(MessageType::SERVER_SET_CURRENT_CAMERA, &currentIdx);
    m_prevCamera = m_camera.current;
    m_camera.arcballToken = {};
    m_manipulatorSynchronized = false;
  }

  if (!m_manipulatorSynchronized && m_camera.current
      && m_camera.current->numMetadata() > 0) {
    tsd::rendering::updateManipulatorFromCamera(
        *m_camera.arcball, *m_camera.current);
    m_manipulatorSynchronized = true;
  }

  if (!m_manipulatorSynchronized
      || !m_camera.arcball->hasChanged(m_camera.arcballToken))
    return;

  tsd::rendering::updateCameraObject(*m_camera.current, *m_camera.arcball);
}

void RemoteViewport::ui_menubar()
{
  if (ImGui::BeginMenuBar()) {
    if (ImGui::BeginMenu("Viewport")) {
      ImGui::Checkbox("show info overlay", &m_showOverlay);
      ImGui::EndMenu();
    }

    BaseViewport::ui_menubar_Renderer();
    BaseViewport::ui_menubar_Camera();
    BaseViewport::ui_menubar_TransformManipulator();

    ImGui::EndMenuBar();
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
    ImGui::Text("viewport: %i x %i", m_viewport.size.x, m_viewport.size.y);
  }
  ImGui::EndChild();

  ImGui::PopStyleColor();
}

} // namespace tsd::ui::imgui
