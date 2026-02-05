// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Message.hpp"
// std
#include <future>
#include <memory>
#include <optional>
#include <thread>

namespace tsd::network {

using MessageFuture = std::future<boost::system::error_code>;

struct NetworkChannel : public std::enable_shared_from_this<NetworkChannel>
{
  NetworkChannel();
  virtual ~NetworkChannel();

  bool isConnected() const;

  //// Receive messages ////

  void registerHandler(uint8_t messageType, MessageHandler handler);
  void removeHandler(uint8_t messageType);
  void removeAllHandlers();

  //// Send messages ////

  MessageFuture send(Message &&msg);
  MessageFuture send(uint8_t type, StructuredMessage &&msg);

  /* No payload */
  MessageFuture send(uint8_t type);

  /* With payloads */
  MessageFuture send(uint8_t type, const std::string &str);
  template <typename T>
  MessageFuture send(uint8_t type, const T *data, uint32_t count = 1);
  template <typename T>
  MessageFuture send(uint8_t type, const std::vector<T> &data);

 protected:
  void start_messaging();
  void stop_messaging();

  void read_header();
  void read_payload(std::shared_ptr<Message> msg);
  void invoke_handler(std::shared_ptr<Message> msg);
  void log_asio_error(
      const boost::system::error_code &error, const char *context);

  asio::io_context m_io_context;
  std::thread m_io_thread;

  using WorkGuard = asio::executor_work_guard<asio::io_context::executor_type>;
  std::optional<WorkGuard> m_work;

  tcp::socket m_socket;
  HandlerMap m_handlers;

 private:
  Message make_message(uint8_t type);
  Message make_message(uint8_t type, const std::string &data);
  template <typename T>
  Message make_message(uint8_t type, const T *data, uint32_t count);
};

struct NetworkServer : public NetworkChannel
{
  NetworkServer(short port);
  ~NetworkServer() override = default;

  void start();
  void restart(); // must be running already
  void stop();

 private:
  void start_accept();

  tcp::acceptor m_acceptor;
};

struct NetworkClient : public NetworkChannel
{
  NetworkClient() = default;
  NetworkClient(const std::string &host, short port);
  ~NetworkClient() override = default;

  void connect(const std::string &host, short port);
  void disconnect();
};

// Inline definitions /////////////////////////////////////////////////////////

template <typename T>
inline MessageFuture NetworkChannel::send(
    uint8_t type, const T *data, uint32_t count)
{
  static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>,
      "Message payload must be a POD type");
  return send(make_message(type, data, count));
}

template <typename T>
inline MessageFuture NetworkChannel::send(
    uint8_t type, const std::vector<T> &data)
{
  return send(
      make_message(type, data.data(), static_cast<uint32_t>(data.size())));
}

inline Message NetworkChannel::make_message(uint8_t type)
{
  Message msg;
  msg.header.type = type;
  return msg;
}

inline Message NetworkChannel::make_message(
    uint8_t type, const std::string &data)
{
  Message msg = make_message(type);
  payloadWrite(msg, data);
  return msg;
}

template <typename T>
inline Message NetworkChannel::make_message(
    uint8_t type, const T *data, uint32_t count)
{
  static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>,
      "Message payload must be a POD type");
  Message msg = make_message(type);
  payloadWrite(msg, data, count);
  return msg;
}

// Inlined helper functions ///////////////////////////////////////////////////

template <typename R>
inline bool is_ready(const std::future<R> &f)
{
  return !f.valid()
      || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

} // namespace tsd::network
