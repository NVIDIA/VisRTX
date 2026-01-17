// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// boost asio
#include <boost/asio.hpp>
// std
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>
// tsd
#include "tsd/core/FlatMap.hpp"

namespace asio = boost::asio;

namespace tsd::network {

using asio::ip::tcp;

using MessageHandler = std::function<void(const struct Message &)>;
using HandlerMap = tsd::core::FlatMap<uint8_t, MessageHandler>;
using MessagePayload = std::vector<std::byte>;

constexpr uint8_t MESSAGE_TYPE_INVALID = 255;

struct Message
{
  struct Header
  {
    uint8_t type{MESSAGE_TYPE_INVALID}; // Type of message
    uint32_t payload_length{0}; // Length of payload in bytes
  } header;

  MessagePayload payload; // Actual payload data
};

// Helper functions ///////////////////////////////////////////////////////////

inline Message make_message(uint8_t type)
{
  Message msg;
  msg.header.type = type;
  msg.header.payload_length = 0;
  return msg;
}

inline Message make_message(uint8_t type, const std::string &data)
{
  Message msg;
  msg.header.type = type;
  msg.header.payload_length = static_cast<uint32_t>(data.size());
  msg.payload.reserve(data.size() + 1);
  msg.payload.resize(data.size());
  std::memcpy(msg.payload.data(), data.data(), data.size());
  msg.payload.push_back(std::byte(0)); // Null-terminate for safety
  msg.header.payload_length++;
  return msg;
}

template <typename T>
inline Message make_message(uint8_t type, const T *data, uint32_t count = 1)
{
  static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>,
      "Message payload must be a POD type");
  Message msg;
  msg.header.type = type;
  msg.header.payload_length = static_cast<uint32_t>(sizeof(T) * count);
  msg.payload.resize(sizeof(T) * count);
  std::memcpy(msg.payload.data(), data, sizeof(T) * count);
  return msg;
}

template <typename T>
inline Message make_message(uint8_t type, const std::vector<T> &data)
{
  static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>,
      "Message payload must be a POD type");
  return make_message(type, data.data(), static_cast<uint32_t>(data.size()));
}

template <typename T>
inline T *payloadAs(Message &msg)
{
  return reinterpret_cast<T *>(msg.payload.data());
}

template <typename T>
inline const T *payloadAs(const Message &msg)
{
  return reinterpret_cast<const T *>(msg.payload.data());
}

template <typename T>
inline bool safeCopyFromPayload(const Message &msg, T *dst, size_t count = 1)
{
  if (msg.header.payload_length != sizeof(T) * count)
    return false;
  std::memcpy(dst, msg.payload.data(), sizeof(T) * count);
  return true;
}

template <typename T>
inline void safeCopyToPayload(Message &msg, T *src, size_t count = 1)
{
  msg.payload.resize(sizeof(T) * count);
  std::memcpy(msg.payload.data(), src, sizeof(T) * count);
}

inline std::vector<std::byte> serialize_message(const Message &msg)
{
  if (msg.payload.size() != msg.header.payload_length) {
    throw std::runtime_error(
        "Message payload size does not match header payload_length");
  }

  std::vector<std::byte> buffer;
  buffer.resize(sizeof(Message::Header) + msg.payload.size());
  auto *dst = buffer.data();
  std::memcpy(dst, &msg.header, sizeof(Message::Header));
  if (!msg.payload.empty()) {
    dst += sizeof(Message::Header);
    std::memcpy(dst, msg.payload.data(), msg.payload.size());
  }
  return buffer;
}

} // namespace tsd::network
