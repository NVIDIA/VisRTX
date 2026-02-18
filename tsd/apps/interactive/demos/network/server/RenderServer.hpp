// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
// tsd_network
#include "tsd/network/NetworkChannel.hpp"
// tsd_app
#include "tsd/app/Core.h"
// tsd_rendering
#include "tsd/rendering/pipeline/RenderPipeline.h"
#include "tsd/rendering/view/ManipulatorToAnari.hpp"

#include "../RenderSession.hpp"

namespace tsd::network {

// Extended TSD protocol message types for volume management.
// These augment the base MessageType enum (IDs 0-21 + 255) with
// volume list / info / attribute / transfer-function operations (110-118).
// Extended TSD protocol message types for volume management (IDs 110-118).
enum TSDVolumeMessage : uint8_t
{
  REQUEST_VOLUME_LIST = 110,
  VOLUME_LIST = 111,
  SET_VOLUME_TF = 112,
  REQUEST_VOLUME_INFO = 116,
  VOLUME_INFO = 117,
  SET_VOLUME_ATTRIBUTE = 118,
};

// Wire-format type tags for SET_VOLUME_ATTRIBUTE payloads.
enum AttributeType : uint8_t
{
  ATTR_BOOL = 0,
  ATTR_INT32 = 1,
  ATTR_FLOAT32 = 2,
  ATTR_STRING = 3,
};

struct RenderServer
{
  RenderServer(int argc, const char **argv);
  ~RenderServer();

  void run(short port = 12345);

 private:
  enum class ServerMode
  {
    DISCONNECTED,
    PAUSED,
    RENDERING,
    SEND_SCENE,
    SHUTDOWN
  };

  void setup_Scene();
  void setup_ANARIDevice();
  void setup_Manipulator();
  void setup_RenderPipeline();
  void setup_Messaging();
  void update_FrameConfig();
  void update_View();
  void send_FrameBuffer();
  void set_Mode(ServerMode mode);
  std::string buildVolumeListJson() const;
  std::string buildVolumeInfoJson(uint32_t volIndex) const;
  void handle_SetVolumeAttribute(const Message &msg);
  void handle_SetVolumeTF(const Message &msg);

  // Data //

  short m_port{12345};

  RenderSession m_session;
  tsd::app::Core m_core;

  std::shared_ptr<NetworkServer> m_server;
  MessageFuture m_lastSentFrame;

  anari::Device m_device{nullptr};
  anari::Camera m_camera{nullptr};
  anari::Renderer m_renderer{nullptr};
  tsd::rendering::Manipulator m_manipulator;
  tsd::rendering::RenderIndex *m_renderIndex{nullptr};
  tsd::rendering::RenderPipeline m_renderPipeline;
  ServerMode m_currentMode{ServerMode::DISCONNECTED};
  ServerMode m_nextMode{ServerMode::DISCONNECTED};
  ServerMode m_previousMode{ServerMode::DISCONNECTED};

  struct SessionVersions
  {
    int frameConfigVersion{-1};
    int viewVersion{-1};
  } m_sessionVersions;
};

} // namespace tsd::network
