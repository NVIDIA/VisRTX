// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <cstdint>
#include <vector>
// tsd_core
#include "tsd/core/TSDMath.hpp"

namespace tsd::network {

// Message types for client-server communication
//
// Note: these are prepended with who is receiving the message for clarity
enum MessageType
{
  // Set state: client -> server
  SERVER_SHUTDOWN = 0,
  SERVER_START_RENDERING,
  SERVER_STOP_RENDERING,
  SERVER_SET_FRAME_CONFIG,
  SERVER_SET_OBJECT_PARAMETER,
  SERVER_REMOVE_OBJECT_PARAMETER,
  SERVER_SET_CURRENT_RENDERER,
  SERVER_SET_CURRENT_CAMERA,
  SERVER_SET_ARRAY_DATA,
  SERVER_ADD_OBJECT,
  SERVER_REMOVE_OBJECT,
  SERVER_REMOVE_ALL_OBJECTS,
  SERVER_UPDATE_LAYER,
  SERVER_UPDATE_TIME,
  SERVER_SAVE_STATE_FILE,

  // Get state: server -> client
  CLIENT_RECEIVE_FRAME_BUFFER_COLOR,
  CLIENT_RECEIVE_FRAME_CONFIG,
  CLIENT_RECEIVE_CURRENT_RENDERER,
  CLIENT_RECEIVE_SCENE,
  CLIENT_RECEIVE_CURRENT_CAMERA,
  CLIENT_RECEIVE_TIME,
  CLIENT_SCENE_TRANSFER_BEGIN, // notify the client a big message is coming...

  // Request state: client-> server
  SERVER_REQUEST_FRAME_CONFIG,
  SERVER_REQUEST_CURRENT_RENDERER,
  SERVER_REQUEST_CURRENT_CAMERA,
  SERVER_REQUEST_SCENE,

  // All ping messages
  PING,

  // All disconnections
  DISCONNECT,

  // All errors
  ERROR = 255
};

struct RenderSession
{
  struct Frame
  {
    struct Config
    {
      tsd::math::uint2 size{800, 600};
    } config;
    int configVersion{0};

    struct Buffers
    {
      std::vector<uint8_t> color;
    } buffers;
    int buffersVersion{0};
  } frame;
};

} // namespace tsd::network
