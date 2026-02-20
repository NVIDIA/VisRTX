// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RenderServer.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
#include "tsd/core/Timer.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"
// tsd_network
#include "tsd/network/JsonHelpers.hpp"
#include "tsd/network/messages/NewObject.hpp"
#include "tsd/network/messages/ParameterChange.hpp"
#include "tsd/network/messages/ParameterRemove.hpp"
#include "tsd/network/messages/RemoveObject.hpp"
#include "tsd/network/messages/TransferArrayData.hpp"
#include "tsd/network/messages/TransferLayer.hpp"
#include "tsd/network/messages/TransferScene.hpp"
// std
#include <cstring>

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
        s->send(
            MessageType::CLIENT_RECEIVE_FRAME_CONFIG, &session->frame.config);
      });

  m_server->registerHandler(MessageType::SERVER_REQUEST_VIEW,
      [s = m_server, session = &m_session](const tsd::network::Message &msg) {
        tsd::core::logDebug("[Server] Client requested view.");
        s->send(MessageType::CLIENT_RECEIVE_VIEW, &session->view);
      });

  m_server->registerHandler(MessageType::SERVER_REQUEST_SCENE,
      [this](const tsd::network::Message &msg) {
        tsd::core::logDebug("[Server] Client requested scene...");
        // Notify client a big message is coming...
        m_server->send(MessageType::CLIENT_SCENE_TRANSFER_BEGIN);
        set_Mode(ServerMode::SEND_SCENE);
      });

  // -- Volume management handlers ------------------------------------------

  m_server->registerHandler(
      REQUEST_VOLUME_LIST, [this](const tsd::network::Message &) {
        std::string json = buildVolumeListJson();
        m_server->send(VOLUME_LIST, json);
      });

  m_server->registerHandler(
      REQUEST_VOLUME_INFO, [this](const tsd::network::Message &msg) {
        if (msg.header.payload_length < 4u)
          return;
        uint32_t volIndex = 0;
        uint32_t pos = 0;
        if (!tsd::network::payloadRead(msg, pos, &volIndex))
          return;
        std::string json = buildVolumeInfoJson(volIndex);
        m_server->send(VOLUME_INFO, json);
      });

  m_server->registerHandler(
      SET_VOLUME_ATTRIBUTE, [this](const tsd::network::Message &msg) {
        handle_SetVolumeAttribute(msg);
      });

  m_server->registerHandler(
      SET_VOLUME_TF, [this](const tsd::network::Message &msg) {
        handle_SetVolumeTF(msg);
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

// ---------------------------------------------------------------------------
// Volume JSON builders
// ---------------------------------------------------------------------------

std::string RenderServer::buildVolumeListJson() const
{
  auto &scene = m_core.tsd.scene;
  const auto &pool = scene.objectDB().volume;
  auto jList = nlohmann::json::array();
  for (size_t i = 0; i < pool.capacity(); i++) {
    auto ref = pool.at(i);
    if (!ref.data())
      continue;
    jList.push_back({{"index", static_cast<unsigned>(i)},
        {"name", std::string(ref->name())}});
  }
  return jList.dump();
}

std::string RenderServer::buildVolumeInfoJson(uint32_t volIndex) const
{
  auto &scene = m_core.tsd.scene;
  const auto &pool = scene.objectDB().volume;
  if (volIndex >= pool.capacity())
    return "{}";
  auto ref = pool.at(volIndex);
  if (!ref.data())
    return "{}";

  nlohmann::json j;
  j["index"] = volIndex;
  j["name"] = std::string(ref->name());

  objectParamsToJson(j, *ref);

  auto *colorArray = ref->parameterValueAsObject<tsd::core::Array>("color");
  if (colorArray && colorArray->size() > 0) {
    j["numColors"] = colorArray->size();
    if (colorArray->isHost()
        && colorArray->elementType() == ANARI_FLOAT32_VEC4) {
      const auto *colors = colorArray->dataAs<tsd::math::float4>();
      if (colors) {
        auto arr = nlohmann::json::array();
        for (size_t c = 0; c < colorArray->size(); c++)
          arr.push_back({colors[c].x, colors[c].y, colors[c].z, colors[c].w});
        j["colors"] = arr;
      }
    }
  }

  objectMetadataToJson(j, *ref);

  auto *field = ref->parameterValueAsObject<tsd::core::SpatialField>("value");
  if (field) {
    nlohmann::json jf;
    jf["name"] = std::string(field->name());
    jf["subtype"] = std::string(field->subtype().c_str());
    objectParamsToJson(jf, *field);
    objectMetadataToJson(jf, *field);
    j["field"] = jf;
  }

  return j.dump();
}

// ---------------------------------------------------------------------------
// Volume attribute + transfer function handlers
// ---------------------------------------------------------------------------

namespace {

// Read a fixed-width, null-terminated string field from a message payload.
std::string readFixedString(
    const tsd::network::Message &msg, uint32_t offset, size_t fieldLen)
{
  const char *raw =
      reinterpret_cast<const char *>(msg.payload.data() + offset);
  std::string s(raw, raw + fieldLen);
  size_t nul = s.find('\0');
  if (nul != std::string::npos)
    s.resize(nul);
  return s;
}

// Set a parameter on whichever object owns it (volume first, then field).
template <typename T>
void setOnVolumeOrField(tsd::core::Object &vol,
    tsd::core::SpatialField *field,
    const char *name,
    const T &value)
{
  if (vol.parameter(name))
    vol.setParameter(name, value);
  else if (field && field->parameter(name))
    field->setParameter(name, value);
}

} // anonymous namespace

void RenderServer::handle_SetVolumeAttribute(const Message &msg)
{
  constexpr size_t nameLen = 64;
  if (msg.header.payload_length < 4u + nameLen + 4u)
    return;

  uint32_t pos = 0;
  uint32_t volIndex = 0;
  if (!tsd::network::payloadRead(msg, pos, &volIndex))
    return;

  std::string name = readFixedString(msg, pos, nameLen);
  if (name.empty())
    return;
  pos += nameLen;

  uint32_t anariType = 0;
  if (!tsd::network::payloadRead(msg, pos, &anariType))
    return;

  auto &scene = m_core.tsd.scene;
  if (volIndex >= scene.objectDB().volume.capacity())
    return;
  auto volRef = scene.objectDB().volume.at(volIndex);
  if (!volRef.data())
    return;

  auto *field =
      volRef->parameterValueAsObject<tsd::core::SpatialField>("value");

  switch (static_cast<int>(anariType)) {
  case ANARI_BOOL: {
    if (pos + 1 > msg.header.payload_length)
      return;
    bool v = static_cast<uint8_t>(msg.payload[pos]) != 0;
    setOnVolumeOrField(*volRef, field, name.c_str(), v);
    tsd::core::logDebug("[Server] vol[%u].%s = %s",
        volIndex,
        name.c_str(),
        v ? "true" : "false");
    break;
  }
  case ANARI_INT32: {
    int32_t v = 0;
    if (!tsd::network::payloadRead(msg, pos, &v))
      return;
    setOnVolumeOrField(*volRef, field, name.c_str(), v);
    tsd::core::logDebug(
        "[Server] vol[%u].%s = %d", volIndex, name.c_str(), v);
    break;
  }
  case ANARI_FLOAT32: {
    float v = 0.f;
    if (!tsd::network::payloadRead(msg, pos, &v))
      return;
    setOnVolumeOrField(*volRef, field, name.c_str(), v);
    tsd::core::logDebug(
        "[Server] vol[%u].%s = %g", volIndex, name.c_str(), v);
    break;
  }
  case ANARI_STRING: {
    constexpr size_t valLen = 64;
    if (pos + valLen > msg.header.payload_length)
      return;
    std::string val = readFixedString(msg, pos, valLen);
    if (volRef->parameter(name.c_str()))
      volRef->setParameter(name.c_str(), ANARI_STRING, val.c_str());
    else if (field && field->parameter(name.c_str()))
      field->setParameter(name.c_str(), ANARI_STRING, val.c_str());
    tsd::core::logDebug(
        "[Server] vol[%u].%s = '%s'", volIndex, name.c_str(), val.c_str());
    break;
  }
  default:
    break;
  }
}

void RenderServer::handle_SetVolumeTF(const Message &msg)
{
  if (msg.header.payload_length < 4u + 4u + 4u + 4u)
    return;

  uint32_t pos = 0u;
  uint32_t volIndex = 0u, numSamples = 0u, numOpacity = 0u;
  if (!tsd::network::payloadRead(msg, pos, &volIndex)
      || !tsd::network::payloadRead(msg, pos, &numSamples))
    return;
  if (numSamples > 4096u)
    return;

  // Read color samples (RGBA float4 array)
  const size_t colorBytes = numSamples * 4u * sizeof(float);
  if (pos + colorBytes > msg.header.payload_length)
    return;
  const float *rgba =
      reinterpret_cast<const float *>(msg.payload.data() + pos);
  pos += static_cast<uint32_t>(colorBytes);

  // Read opacity control points (XY float2 array)
  if (!tsd::network::payloadRead(msg, pos, &numOpacity))
    return;
  if (numOpacity > 4096u)
    return;
  const size_t opacityBytes = numOpacity * 2u * sizeof(float);
  if (pos + opacityBytes > msg.header.payload_length)
    return;
  const float *xy =
      reinterpret_cast<const float *>(msg.payload.data() + pos);
  pos += static_cast<uint32_t>(opacityBytes);

  // Flags bitmask indicating which optional fields are present:
  //   bit 0 → valueRange (2 floats), bit 1 → opacity, bit 2 → unitDistance
  uint32_t flags = 0;
  if (!tsd::network::payloadRead(msg, pos, &flags))
    return;

  // Resolve volume
  auto &scene = m_core.tsd.scene;
  if (volIndex >= scene.objectDB().volume.capacity())
    return;
  auto volRef = scene.objectDB().volume.at(volIndex);
  if (!volRef.data())
    return;

  // Apply color array
  std::vector<tsd::math::float4> colors(numSamples);
  for (uint32_t i = 0; i < numSamples; i++)
    colors[i] = tsd::math::float4(
        rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], rgba[i * 4 + 3]);
  auto colorArray = scene.createArray(ANARI_FLOAT32_VEC4, numSamples);
  colorArray->setData(colors.data());
  volRef->setParameterObject("color", *colorArray);

  // Apply opacity control points
  std::vector<tsd::math::float2> opacityPts(numOpacity);
  for (uint32_t i = 0; i < numOpacity; i++)
    opacityPts[i] = tsd::math::float2(xy[i * 2], xy[i * 2 + 1]);
  volRef->setMetadataArray(
      "opacityControlPoints", ANARI_FLOAT32_VEC2, opacityPts.data(), numOpacity);

  // Optional fields keyed by flags bitmask
  if (flags & 0x1) {
    float lo = 0.f, hi = 0.f;
    if (!tsd::network::payloadRead(msg, pos, &lo)
        || !tsd::network::payloadRead(msg, pos, &hi))
      return;
    float range[2] = {lo, hi};
    volRef->setParameter("valueRange", ANARI_FLOAT32_BOX1, range);
  }
  if (flags & 0x2) {
    float opacity = 0.f;
    if (!tsd::network::payloadRead(msg, pos, &opacity))
      return;
    volRef->setParameter("opacity", opacity);
  }
  if (flags & 0x4) {
    float unitDistance = 0.f;
    if (!tsd::network::payloadRead(msg, pos, &unitDistance))
      return;
    volRef->setParameter("unitDistance", unitDistance);
  }

  tsd::core::logDebug(
      "[Server] Set TF for volume %u (%zu colors, %u opacity pts)",
      volIndex,
      size_t(numSamples),
      numOpacity);
}

} // namespace tsd::network
