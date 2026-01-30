// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// boost asio
#include <boost/asio.hpp>
// std
#include <string.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>
// tsd_core
#include "tsd/core/DataTree.hpp"
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

struct StructuredMessage
{
  StructuredMessage() = default;
  StructuredMessage(const Message &msg); // parse received message
  virtual ~StructuredMessage() = default;

  void fromMessage(const Message &msg);
  Message toMessage(uint8_t type);

  const tsd::core::DataTree &tree() const;

  // Child classes override to associate behavior on receipt using tree contents
  virtual void execute() = 0;

 protected:
  tsd::core::DataTree m_tree;
};

// Helper functions ///////////////////////////////////////////////////////////

// write to payload //

template <typename T>
void payloadWrite(Message &msg, const T *data, uint32_t length = 1);
void payloadWrite(Message &msg, const std::string &str);

// read from payload //

template <typename T>
T *payloadAs(Message &msg);
template <typename T>
const T *payloadAs(const Message &msg);

template <typename T>
bool payloadRead(
    const Message &msg, uint32_t &offset, T *data, uint32_t length = 1);
bool payloadRead(const Message &msg, uint32_t &offset, std::string &str);

// make_message() //

Message make_message(uint8_t type);
Message make_message(uint8_t type, const std::string &data);
template <typename T>
Message make_message(uint8_t type, const T *data, uint32_t count = 1);
template <typename T>
Message make_message(uint8_t type, const std::vector<T> &data);

///////////////////////////////////////////////////////////////////////////////
// Inlined definitions ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// StucturedMessage definitions //

inline StructuredMessage::StructuredMessage(const Message &msg)
{
  fromMessage(msg);
}

inline void StructuredMessage::fromMessage(const Message &msg)
{
  m_tree.load(msg.payload);
}

inline Message StructuredMessage::toMessage(uint8_t type)
{
  Message msg;
  msg.header.type = type;
  m_tree.save(msg.payload);
  msg.header.payload_length = uint32_t(msg.payload.size());
  return msg;
}

inline const tsd::core::DataTree &StructuredMessage::tree() const
{
  return m_tree;
}

// Helper functions //

template <typename T>
inline void payloadWrite(Message &msg, const T *data, uint32_t length)
{
  static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>,
      "Message payload must be a POD type");
  size_t old_size = msg.payload.size();
  msg.payload.resize(old_size + sizeof(T) * length);
  std::memcpy(msg.payload.data() + old_size, data, sizeof(T) * length);
  msg.header.payload_length += static_cast<uint32_t>(sizeof(T) * length);
}

inline void payloadWrite(Message &msg, const std::string &str)
{
  size_t old_size = msg.payload.size();
  msg.payload.resize(old_size + str.size() + 1);
  std::memcpy(msg.payload.data() + old_size, str.data(), str.size());
  msg.header.payload_length += static_cast<uint32_t>(str.size() + 1);
  msg.payload.back() = std::byte(0); // Null-terminate
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
inline bool payloadRead(
    const Message &msg, uint32_t &offset, T *data, uint32_t length)
{
  static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>,
      "Message payload must be a POD type");
  size_t expected_size = sizeof(T) * length;
  if (msg.header.payload_length < (expected_size + offset))
    return false;
  std::memcpy(data, msg.payload.data() + offset, expected_size);
  offset += static_cast<uint32_t>(expected_size);
  return true;
}

inline bool payloadRead(const Message &msg, uint32_t &offset, std::string &str)
{
  if (offset >= msg.header.payload_length)
    return false;
  const char *start =
      reinterpret_cast<const char *>(msg.payload.data() + offset);
  size_t max_length = msg.header.payload_length - offset;
  size_t length = strnlen(start, max_length);
  str.assign(start, length);
  offset += static_cast<uint32_t>(length + 1); // +1 for null-terminator
  return true;
}

inline Message make_message(uint8_t type)
{
  Message msg;
  msg.header.type = type;
  return msg;
}

inline Message make_message(uint8_t type, const std::string &data)
{
  Message msg = make_message(type);
  payloadWrite(msg, data);
  return msg;
}

template <typename T>
inline Message make_message(uint8_t type, const T *data, uint32_t count)
{
  Message msg = make_message(type);
  payloadWrite(msg, data, count);
  return msg;
}

template <typename T>
inline Message make_message(uint8_t type, const std::vector<T> &data)
{
  static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>,
      "Message payload must be a POD type");
  return make_message(type, data.data(), static_cast<uint32_t>(data.size()));
}

} // namespace tsd::network
