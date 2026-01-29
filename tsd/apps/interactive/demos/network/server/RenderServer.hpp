// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <memory>
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

struct RenderServer
{
  RenderServer(int argc, const char **argv);
  ~RenderServer();

  void run(short port = 12345);

 private:
  void setup_Scene();
  void setup_ANARIDevice();
  void setup_Manipulator();
  void setup_RenderPipeline();
  void setup_Messaging();
  void update_FrameConfig();
  void update_View();
  void send_FrameBuffer();

  enum class ServerMode
  {
    DISCONNECTED,
    PAUSED,
    RENDERING,
    SHUTDOWN
  };

  // Data //

  short m_port{12345};
  bool m_running{true};

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
  ServerMode m_mode{ServerMode::DISCONNECTED};
  ServerMode m_previousMode{ServerMode::DISCONNECTED};

  struct SessionVersions
  {
    int frameConfigVersion{-1};
    int viewVersion{-1};
  } m_sessionVersions;
};

} // namespace tsd::network
