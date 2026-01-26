// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// std
#include <iostream>
#include <memory>
#include <thread>
// tsd_network
#include "tsd/network/NetworkChannel.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// stb_image
#include "stb_image_write.h"

#include "../RenderSession.hpp"

using tsd::network::MessageType;

int main()
{
  // Global setup //

  tsd::core::setLogToStdout();

  bool running = false;
  int framesReceived = 0;
  std::string filename = "server_test_frame.png";

  stbi_flip_vertically_on_write(1);

  tsd::network::MessagePayload frameData;

  // Client setup //

  auto session = std::make_shared<tsd::network::RenderSession>();
  auto client =
      std::make_shared<tsd::network::NetworkClient>("127.0.0.1", 12345);

  // Handlers //

  client->registerHandler(
      MessageType::ERROR, [](const tsd::network::Message &msg) {
        tsd::core::logError("[Client] Received error from server: '%s'",
            tsd::network::payloadAs<char>(msg));
      });

  client->registerHandler(
      MessageType::CLIENT_PING, [](const tsd::network::Message &msg) {
        tsd::core::logStatus("[Client] Received PING from server");
      });

  client->registerHandler(MessageType::CLIENT_RECEIVE_FRAME_CONFIG,
      [session](const tsd::network::Message &msg) {
        auto pos = 0u;
        tsd::network::payloadRead(msg, pos, &session->frame.config);
        tsd::core::logStatus("[Client] Received frame config: size=(%u,%u)",
            session->frame.config.size.x,
            session->frame.config.size.y);
      });

  client->registerHandler(MessageType::CLIENT_RECEIVE_VIEW,
      [session](const tsd::network::Message &msg) {
        auto pos = 0u;
        tsd::network::payloadRead(msg, pos, &session->view);
        tsd::core::logStatus(
            "[Client] Received view: azel=(%f,%f), dist=%f, "
            "lookat=(%f,%f,%f)",
            session->view.azeldist.x,
            session->view.azeldist.y,
            session->view.azeldist.z,
            session->view.lookat.x,
            session->view.lookat.y,
            session->view.lookat.z);
      });

  client->registerHandler(MessageType::CLIENT_RECEIVE_FRAME_BUFFER_COLOR,
      [&](const tsd::network::Message &msg) {
        tsd::core::logStatus(
            "[Client] Received color buffer #%i of size %u bytes",
            framesReceived++,
            msg.header.payload_length);
        frameData = msg.payload;
      });

  // Run client //

  client->start();

  for (int i = 0; i < 3; ++i) {
    tsd::core::logStatus("[Client] Sending PING #%d", i + 1);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    client->send(make_message(MessageType::SERVER_PING));
  }

  client->send(make_message(MessageType::SERVER_REQUEST_FRAME_CONFIG));
  client->send(make_message(MessageType::SERVER_REQUEST_VIEW));
  client->send(make_message(MessageType::SERVER_START_RENDERING));

  for (int i = 0; i < 3; ++i) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    tsd::core::logStatus("[Client] Waiting for %i seconds...", i + 1);
  }

  client->send(make_message(MessageType::SERVER_STOP_RENDERING));

  // Shutdown //

  tsd::core::logStatus("[Client] Sending SHUTDOWN");
  client->send(make_message(MessageType::SERVER_SHUTDOWN));

  client->stop();

  // Store fine frame //

  auto size = session->frame.config.size;

  // Write PNG file (4 components = RGBA, stride = width * 4 bytes)
  int result = stbi_write_png(filename.c_str(),
      static_cast<int>(size.x),
      static_cast<int>(size.y),
      4, // RGBA
      frameData.data(),
      static_cast<int>(size.x) * 4);

  if (result) {
    tsd::core::logStatus("[Client] Saved image to '%s'", filename.c_str());
  } else {
    tsd::core::logWarning(
        "[Client] Failed to save image to '%s'", filename.c_str());
  }

  return 0;
}
