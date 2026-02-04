// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RenderServer.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
#include "tsd/core/Timer.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"
// tsd_network
#include "tsd/network/messages/NewObject.hpp"
#include "tsd/network/messages/ParameterChange.hpp"
#include "tsd/network/messages/ParameterRemove.hpp"
#include "tsd/network/messages/RemoveObject.hpp"
#include "tsd/network/messages/TransferArrayData.hpp"
#include "tsd/network/messages/TransferLayer.hpp"
#include "tsd/network/messages/TransferScene.hpp"

namespace tsd::network {

RenderServer::RenderServer(int argc, const char **argv)
{
  tsd::core::setLogToStdout();
  tsd::core::logStatus("[Server] Parsing command line...");
  m_core.parseCommandLine(argc, argv);
}

RenderServer::~RenderServer() = default;

void RenderServer::run(short port)
{
  m_port = port;

  setup_Scene();
  setup_ANARIDevice();
  setup_Manipulator();
  setup_RenderPipeline();
  setup_Messaging();

  m_server->start();

  tsd::core::logStatus("[Server] Listening on port %i...", int(port));

  while (m_currentMode != ServerMode::SHUTDOWN) {
    bool wasRendering = m_currentMode == ServerMode::RENDERING;

    m_currentMode =
        m_server->isConnected() ? m_nextMode : ServerMode::DISCONNECTED;

    if (m_currentMode == ServerMode::DISCONNECTED) {
      m_lastSentFrame = {}; // reset any pending frame sends
      if (m_previousMode != ServerMode::DISCONNECTED) {
        tsd::core::logStatus("[Server] Listening on port %i...", int(port));
        m_server->restart();
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } else if (m_currentMode == ServerMode::RENDERING) {
      tsd::core::logDebug("[Server] Rendering frame...");
      update_FrameConfig();
      update_View();
      m_renderPipeline.render();
      send_FrameBuffer();
    } else if (m_currentMode == ServerMode::SEND_SCENE) {
      tsd::core::logStatus("[Server] Serializing + sending scene...");

      tsd::core::Timer timer;
      timer.start();
      tsd::network::messages::TransferScene sceneMsg(&m_core.tsd.scene);
      m_server->send(MessageType::CLIENT_RECEIVE_SCENE, std::move(sceneMsg))
          .get();
      timer.end();
      tsd::core::logStatus("[Server] ...done! (%.3f s)", timer.seconds());

      set_Mode(wasRendering ? ServerMode::RENDERING : ServerMode::PAUSED);
    } else {
      if (m_previousMode != ServerMode::PAUSED)
        tsd::core::logStatus("[Server] Rendering paused...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    m_previousMode = m_currentMode;
  }

  tsd::core::logStatus("[Server] Shutting down...");

  m_server->stop();
  m_server->removeAllHandlers();

  anari::release(m_device, m_camera);
  anari::release(m_device, m_renderer);
  m_core.anari.releaseRenderIndex(m_device);
  m_core.anari.releaseAllDevices();
}

void RenderServer::setup_Scene()
{
  tsd::core::logStatus("[Server] Setting up scene from command line...");
  m_core.setupSceneFromCommandLine();
  tsd::core::logStatus(
      "%s", tsd::core::objectDBInfo(m_core.tsd.scene.objectDB()).c_str());
  tsd::core::logStatus("[Server] Scene setup complete.");
}

void RenderServer::setup_ANARIDevice()
{
  tsd::core::logStatus("[Server] Loading 'environment' device...");
  auto device = m_core.anari.loadDevice("environment");
  if (!device) {
    tsd::core::logError("[Server] Failed to load 'environment' ANARI device.");
    std::exit(EXIT_FAILURE);
  }

  auto &scene = m_core.tsd.scene;

  m_device = device;
  m_renderIndex = m_core.anari.acquireRenderIndex(scene, device);
  m_camera = anari::newObject<anari::Camera>(device, "perspective");
  m_renderer = anari::newObject<anari::Renderer>(device, "default");

  anari::setParameter(device, m_renderer, "ambientRadiance", 1.f);
  anari::commitParameters(device, m_renderer);
}

void RenderServer::setup_Manipulator()
{
  tsd::core::logStatus("[Server] Setting up manipulator...");

  tsd::math::float3 bounds[2] = {{-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}};
  auto &scene = m_core.tsd.scene;
  if (!anariGetProperty(m_device,
          m_renderIndex->world(),
          "bounds",
          ANARI_FLOAT32_BOX3,
          &bounds[0],
          sizeof(bounds),
          ANARI_WAIT)) {
    tsd::core::logWarning("[Server] anari::World returned no bounds!");
  }

  auto center = 0.5f * (bounds[0] + bounds[1]);
  auto diag = bounds[1] - bounds[0];

  m_manipulator.setConfig(center, 1.25f * linalg::length(diag), {0.f, 20.f});

  auto azel = m_manipulator.azel();
  auto dist = m_manipulator.distance();
  auto lookat = m_manipulator.at();

  m_session.view.azeldist = {azel.x, azel.y, dist};
  m_session.view.lookat = lookat;
}

void RenderServer::setup_RenderPipeline()
{
  tsd::core::logStatus("[Server] Setting up render pipeline...");

  m_renderPipeline.setDimensions(
      m_session.frame.config.size.x, m_session.frame.config.size.y);

  auto *arp =
      m_renderPipeline.emplace_back<tsd::rendering::AnariSceneRenderPass>(
          m_device);
  arp->setWorld(m_renderIndex->world());
  arp->setRenderer(m_renderer);
  arp->setCamera(m_camera);
  arp->setEnableIDs(false);

  auto *ccbp =
      m_renderPipeline.emplace_back<tsd::rendering::CopyFromColorBufferPass>();
  ccbp->setExternalBuffer(m_session.frame.buffers.color);
}

void RenderServer::setup_Messaging()
{
  tsd::core::logStatus("[Server] Setting up messaging...");

  m_server = std::make_shared<NetworkServer>(m_port);

  // Handlers //

  m_server->registerHandler(
      MessageType::ERROR, [](const tsd::network::Message &msg) {
        tsd::core::logError("[Server] Received error from client: '%s'",
            tsd::network::payloadAs<char>(msg));
      });

  m_server->registerHandler(
      MessageType::PING, [](const tsd::network::Message &msg) {
        tsd::core::logStatus("[Server] Received PING from client");
      });

  m_server->registerHandler(
      MessageType::DISCONNECT, [&](const tsd::network::Message &msg) {
        tsd::core::logStatus("[Server] Client signaled disconnection.");
        set_Mode(ServerMode::DISCONNECTED);
      });

  m_server->registerHandler(MessageType::SERVER_START_RENDERING,
      [&](const tsd::network::Message &msg) {
        tsd::core::logStatus(
            "[Server] Starting rendering as requested by client.");
        set_Mode(ServerMode::RENDERING);
      });

  m_server->registerHandler(MessageType::SERVER_STOP_RENDERING,
      [&](const tsd::network::Message &msg) {
        tsd::core::logStatus(
            "[Server] Stopping rendering as requested by client.");
        set_Mode(ServerMode::PAUSED);
        if (m_lastSentFrame.valid())
          m_lastSentFrame.get();
      });

  m_server->registerHandler(
      MessageType::SERVER_SHUTDOWN, [&](const tsd::network::Message &msg) {
        tsd::core::logStatus("[Server] Shutdown message received from client.");
        set_Mode(ServerMode::SHUTDOWN);
      });

  m_server->registerHandler(MessageType::SERVER_SET_FRAME_CONFIG,
      [&](const tsd::network::Message &msg) {
        auto *config = &m_session.frame.config;
        auto pos = 0u;
        if (tsd::network::payloadRead(msg, pos, config)) {
          m_session.frame.configVersion++;
          tsd::core::logDebug(
              "[Server] Received frame config: size=(%u,%u), version=%d",
              config->size.x,
              config->size.y,
              m_session.frame.configVersion);
        } else {
          tsd::core::logError(
              "[Server] Invalid payload for SERVER_SET_FRAME_CONFIG");
        }
      });

  m_server->registerHandler(
      MessageType::SERVER_SET_VIEW, [&](const tsd::network::Message &msg) {
        auto *view = &m_session.view;
        auto pos = 0u;
        if (tsd::network::payloadRead(msg, pos, view)) {
          m_session.viewVersion++;
          tsd::core::logDebug(
              "[Server] Received view: azel=(%f,%f), dist=%f, "
              "lookat=(%f,%f,%f), version=%d",
              view->azeldist.x,
              view->azeldist.y,
              view->azeldist.z,
              view->lookat.x,
              view->lookat.y,
              view->lookat.z,
              m_session.viewVersion);
        } else {
          tsd::core::logError("[Server] Invalid payload for SERVER_SET_VIEW");
        }
      });

  m_server->registerHandler(MessageType::SERVER_SET_OBJECT_PARAMETER,
      [this](const tsd::network::Message &msg) {
        tsd::network::messages::ParameterChange paramChange(
            msg, &m_core.tsd.scene);
        paramChange.execute();
      });

  m_server->registerHandler(MessageType::SERVER_REMOVE_OBJECT_PARAMETER,
      [this](const tsd::network::Message &msg) {
        tsd::network::messages::ParameterRemove paramRemove(
            msg, &m_core.tsd.scene);
        paramRemove.execute();
      });

  m_server->registerHandler(MessageType::SERVER_SET_ARRAY_DATA,
      [this](const tsd::network::Message &msg) {
        tsd::network::messages::TransferArrayData arrayData(
            msg, &m_core.tsd.scene);
        arrayData.execute();
      });

  m_server->registerHandler(
      MessageType::SERVER_ADD_OBJECT, [this](const tsd::network::Message &msg) {
        tsd::network::messages::NewObject newObj(msg, &m_core.tsd.scene);
        newObj.execute();
      });

  m_server->registerHandler(MessageType::SERVER_REMOVE_OBJECT,
      [this](const tsd::network::Message &msg) {
        tsd::network::messages::RemoveObject removeObj(msg, &m_core.tsd.scene);
        removeObj.execute();
      });

  m_server->registerHandler(MessageType::SERVER_REMOVE_ALL_OBJECTS,
      [this](const tsd::network::Message &) {
        m_core.tsd.scene.removeAllObjects();
      });

  m_server->registerHandler(MessageType::SERVER_UPDATE_LAYER,
      [this](const tsd::network::Message &msg) {
        tsd::network::messages::TransferLayer layerMsg(msg, &m_core.tsd.scene);
        layerMsg.execute();
      });

  m_server->registerHandler(MessageType::SERVER_REQUEST_FRAME_CONFIG,
      [s = m_server, session = &m_session](const tsd::network::Message &msg) {
        tsd::core::logDebug("[Server] Client requested frame config.");
        s->send(make_message(
            MessageType::CLIENT_RECEIVE_FRAME_CONFIG, &session->frame.config));
      });

  m_server->registerHandler(MessageType::SERVER_REQUEST_VIEW,
      [s = m_server, session = &m_session](const tsd::network::Message &msg) {
        tsd::core::logDebug("[Server] Client requested view.");
        s->send(make_message(MessageType::CLIENT_RECEIVE_VIEW, &session->view));
      });

  m_server->registerHandler(MessageType::SERVER_REQUEST_SCENE,
      [this](const tsd::network::Message &msg) {
        tsd::core::logDebug("[Server] Client requested scene...");
        // Notify client a big message is coming...
        m_server->send(MessageType::CLIENT_SCENE_TRANSFER_BEGIN);
        set_Mode(ServerMode::SEND_SCENE);
      });
}

void RenderServer::update_FrameConfig()
{
  if (m_session.frame.configVersion == m_sessionVersions.frameConfigVersion)
    return;

  m_renderPipeline.setDimensions(
      m_session.frame.config.size.x, m_session.frame.config.size.y);
  m_sessionVersions.frameConfigVersion = m_session.frame.configVersion;

  auto d = m_device;
  anari::setParameter(d,
      m_camera,
      "aspect",
      float(m_session.frame.config.size.x)
          / float(m_session.frame.config.size.y));
  anari::commitParameters(d, m_camera);
}

void RenderServer::update_View()
{
  if (m_session.viewVersion == m_sessionVersions.viewVersion)
    return;

  auto d = m_device;
  m_manipulator.setAzel({m_session.view.azeldist.x, m_session.view.azeldist.y});
  m_manipulator.setDistance(m_session.view.azeldist.z);
  m_manipulator.setCenter(m_session.view.lookat);
  tsd::rendering::updateCameraParametersPerspective(d, m_camera, m_manipulator);
  anari::commitParameters(d, m_camera);
  m_sessionVersions.viewVersion = m_session.viewVersion;
}

void RenderServer::send_FrameBuffer()
{
  if (!is_ready<boost::system::error_code>(m_lastSentFrame)) {
    tsd::core::logStatus(
        "[Server] Previous frame still being sent, skipping this frame.");
    return;
  }

  m_lastSentFrame = m_server->send(
      tsd::network::make_message(MessageType::CLIENT_RECEIVE_FRAME_BUFFER_COLOR,
          m_session.frame.buffers.color));
}

void RenderServer::set_Mode(ServerMode mode)
{
  const bool shuttingDown = m_nextMode == ServerMode::SHUTDOWN
      || m_currentMode == ServerMode::SHUTDOWN;
  if (shuttingDown) // if shutting down, do not change mode
    return;
  m_nextMode = mode;
}

} // namespace tsd::network
