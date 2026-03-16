// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RenderServer.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
#include "tsd/core/Timer.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"
// tsd_rendering
#include "tsd/rendering/view/ManipulatorToTSD.hpp"
// tsd_network
#include "tsd/network/messages/NewObject.hpp"
#include "tsd/network/messages/ParameterChange.hpp"
#include "tsd/network/messages/ParameterRemove.hpp"
#include "tsd/network/messages/RemoveObject.hpp"
#include "tsd/network/messages/TransferArrayData.hpp"
#include "tsd/network/messages/TransferLayer.hpp"
#include "tsd/network/messages/TransferScene.hpp"
// std
#include <cstdlib>

namespace tsd::network {

RenderServer::RenderServer(int argc, const char **argv)
{
  tsd::core::setLogToStdout(true);
  tsd::core::logStatus("[Server] Parsing command line...");
  m_ctx.parseCommandLine(argc, argv);
}

RenderServer::~RenderServer() = default;

void RenderServer::run(short port)
{
  m_port = port;

  setup_Scene();
  setup_ANARIDevice();
  setup_Camera();
  setup_ImagePipeline();
  setup_Messaging();

  m_server->start();

  tsd::core::logStatus("[Server] Listening on port %i...", int(port));

  while (m_nextMode != ServerMode::SHUTDOWN) {
    bool wasRendering = m_currentMode == ServerMode::RENDERING;

    m_currentMode =
        m_server->isConnected() ? m_nextMode : ServerMode::DISCONNECTED;

    if (m_currentMode == ServerMode::DISCONNECTED) {
      if (m_previousMode != ServerMode::DISCONNECTED) {
        tsd::core::logStatus("[Server] Listening on port %i...", int(port));
        m_server->restart();
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } else if (m_currentMode == ServerMode::RENDERING) {
      if (m_previousMode != ServerMode::RENDERING)
        tsd::core::logDebug("[Server] Rendering frames...");
      update_FrameConfig();
      m_renderPipeline.render();
      send_FrameBuffer();
    } else if (m_currentMode == ServerMode::SEND_SCENE) {
      tsd::core::logStatus("[Server] Serializing + sending scene...");

      tsd::core::Timer timer;
      timer.start();
      tsd::network::messages::TransferScene sceneMsg(&m_ctx.tsd.scene);
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

  m_camera = {};
  m_ctx.anari.releaseRenderIndex(m_device);
  m_ctx.anari.releaseAllDevices();
}

void RenderServer::setup_Scene()
{
  tsd::core::logStatus("[Server] Setting up scene from command line...");
  m_ctx.setupSceneFromCommandLine();
  tsd::core::logStatus(
      "%s", tsd::scene::objectDBInfo(m_ctx.tsd.scene.objectDB()).c_str());
  tsd::core::logStatus("[Server] Scene setup complete.");
}

void RenderServer::setup_ANARIDevice()
{
  tsd::core::logStatus("[Server] Loading 'environment' device...");
  const char *libNameEnv = std::getenv("ANARI_LIBRARY");
  if (!libNameEnv) {
    tsd::core::logWarning(
        "[Server] ANARI_LIBRARY environment variable not set,"
        " defaulting to 'helide'");
    libNameEnv = "helide"; // default to helide if env var not set
  } else {
    tsd::core::logStatus(
        "[Server] ANARI_LIBRARY environment variable set to '%s'", libNameEnv);
  }

  m_libName = libNameEnv;

  auto device = m_ctx.anari.loadDevice(m_libName);
  if (!device) {
    tsd::core::logError(
        "[Server] Failed to load '%s' ANARI device.", m_libName.c_str());
    std::exit(EXIT_FAILURE);
  }

  auto &scene = m_ctx.tsd.scene;

  m_device = device;
  m_renderIndex = m_ctx.anari.acquireRenderIndex(scene, m_libName, device);
  m_camera = scene.defaultCamera();
  m_renderers = scene.renderersOfDevice(m_libName).empty()
      ? scene.createStandardRenderers(m_libName, device)
      : scene.renderersOfDevice(m_libName);
  m_currentRenderer = m_renderers[0];
}

void RenderServer::setup_Camera()
{
  tsd::core::logStatus("[Server] Setting up camera...");
  tsd::rendering::Manipulator manipulator;
  manipulator.setConfig(m_renderIndex->computeDefaultView());
  tsd::rendering::updateCameraObject(*m_camera, manipulator, true);
}

void RenderServer::setup_ImagePipeline()
{
  tsd::core::logStatus("[Server] Setting up render pipeline...");

  m_renderPipeline.setDimensions(
      m_session.frame.config.size.x, m_session.frame.config.size.y);

  auto *arp =
      m_renderPipeline.emplace_back<tsd::rendering::AnariSceneRenderPass>(
          m_device);
  arp->setWorld(m_renderIndex->world());
  arp->setRenderer(m_renderIndex->renderer(m_currentRenderer->index()));
  arp->setCamera(m_renderIndex->camera(m_camera->index()));
  arp->setEnableIDs(false);
  m_sceneImagePass = arp;

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

  m_server->registerHandler(MessageType::SERVER_SET_OBJECT_PARAMETER,
      [this](const tsd::network::Message &msg) {
        tsd::network::messages::ParameterChange paramChange(
            msg, &m_ctx.tsd.scene);
        paramChange.execute();
      });

  m_server->registerHandler(MessageType::SERVER_REMOVE_OBJECT_PARAMETER,
      [this](const tsd::network::Message &msg) {
        tsd::network::messages::ParameterRemove paramRemove(
            msg, &m_ctx.tsd.scene);
        paramRemove.execute();
      });

  m_server->registerHandler(MessageType::SERVER_SET_CURRENT_RENDERER,
      [this](const tsd::network::Message &msg) {
        size_t idx = 0;
        uint32_t pos = 0;
        if (tsd::network::payloadRead(msg, pos, &idx)) {
          if (idx < m_renderers.size()) {
            auto renderer = m_renderers[idx];
            tsd::core::logDebug(
                "[Server] Setting current renderer to index %u (subtype '%s')",
                idx,
                renderer->subtype().c_str());
            m_currentRenderer = renderer;
            m_sceneImagePass->setRenderer(m_renderIndex->renderer(idx));
          } else {
            tsd::core::logError(
                "[Server] Invalid renderer index %u in "
                "SERVER_SET_CURRENT_RENDERER",
                idx);
          }
        } else {
          tsd::core::logError(
              "[Server] Invalid payload for SERVER_SET_CURRENT_RENDERER");
        }
      });

  m_server->registerHandler(MessageType::SERVER_SET_CURRENT_CAMERA,
      [this](const tsd::network::Message &msg) {
        size_t idx = 0;
        uint32_t pos = 0;
        if (tsd::network::payloadRead(msg, pos, &idx)) {
          auto camera = m_ctx.tsd.scene.getObject<tsd::scene::Camera>(idx);
          if (camera) {
            tsd::core::logDebug(
                "[Server] Setting current camera to index %u (subtype '%s')",
                idx,
                camera->subtype().c_str());
            m_sceneImagePass->setCamera(m_renderIndex->camera(idx));
          } else {
            tsd::core::logError(
                "[Server] Invalid camera index %u in "
                "SERVER_SET_CURRENT_CAMERA",
                idx);
          }
        } else {
          tsd::core::logError(
              "[Server] Invalid payload for SERVER_SET_CURRENT_CAMERA");
        }
      });

  m_server->registerHandler(MessageType::SERVER_SET_ARRAY_DATA,
      [this](const tsd::network::Message &msg) {
        tsd::network::messages::TransferArrayData arrayData(
            msg, &m_ctx.tsd.scene);
        arrayData.execute();
      });

  m_server->registerHandler(
      MessageType::SERVER_ADD_OBJECT, [this](const tsd::network::Message &msg) {
        tsd::network::messages::NewObject newObj(msg, &m_ctx.tsd.scene);
        newObj.execute();
      });

  m_server->registerHandler(MessageType::SERVER_REMOVE_OBJECT,
      [this](const tsd::network::Message &msg) {
        tsd::network::messages::RemoveObject removeObj(msg, &m_ctx.tsd.scene);
        removeObj.execute();
      });

  m_server->registerHandler(MessageType::SERVER_REMOVE_ALL_OBJECTS,
      [this](const tsd::network::Message &) {
        m_ctx.tsd.scene.removeAllObjects();
      });

  m_server->registerHandler(MessageType::SERVER_UPDATE_LAYER,
      [this](const tsd::network::Message &msg) {
        tsd::network::messages::TransferLayer layerMsg(msg, &m_ctx.tsd.scene);
        layerMsg.execute();
      });

  m_server->registerHandler(MessageType::SERVER_SAVE_STATE_FILE,
      [this](const tsd::network::Message &msg) {
        std::string filename;
        uint32_t pos = 0;
        if (tsd::network::payloadRead(msg, pos, filename)) {
          tsd::core::logStatus(
              "[Server] Saving state file '%s' as requested by client.",
              filename.c_str());
          tsd::io::save_Scene(m_ctx.tsd.scene, filename.c_str());
        } else {
          tsd::core::logError(
              "[Server] Invalid payload for SERVER_SAVE_STATE_FILE");
        }
      });

  m_server->registerHandler(MessageType::SERVER_REQUEST_FRAME_CONFIG,
      [s = m_server, session = &m_session](const tsd::network::Message &msg) {
        tsd::core::logDebug("[Server] Client requested frame config.");
        s->send(
            MessageType::CLIENT_RECEIVE_FRAME_CONFIG, &session->frame.config);
      });

  m_server->registerHandler(MessageType::SERVER_REQUEST_CURRENT_RENDERER,
      [this, s = m_server](const tsd::network::Message &msg) {
        tsd::core::logDebug("[Server] Client requested current renderer.");
        auto idx = m_currentRenderer->index();
        s->send(MessageType::CLIENT_RECEIVE_CURRENT_RENDERER, &idx);
      });

  m_server->registerHandler(MessageType::SERVER_REQUEST_CURRENT_CAMERA,
      [this, s = m_server](const tsd::network::Message &msg) {
        tsd::core::logDebug("[Server] Client requested current camera.");
        auto idx = m_camera->index();
        s->send(MessageType::CLIENT_RECEIVE_CURRENT_CAMERA, &idx);
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
  auto c = m_renderIndex->camera(m_camera->index());
  anari::setParameter(d,
      c,
      "aspect",
      float(m_session.frame.config.size.x)
          / float(m_session.frame.config.size.y));
  anari::commitParameters(d, c);
}

void RenderServer::send_FrameBuffer()
{
  if (!is_ready<boost::system::error_code>(m_lastSentFrame)) {
    tsd::core::logStatus(
        "[Server] Previous frame still being sent, skipping this frame.");
    return;
  }

  m_lastSentFrame =
      m_server->send(MessageType::CLIENT_RECEIVE_FRAME_BUFFER_COLOR,
          m_session.frame.buffers.color);
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
