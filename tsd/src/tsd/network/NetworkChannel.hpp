// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Message.hpp"
// std
#include <future>
#include <memory>
#include <thread>

namespace tsd::network {

using MessageFuture = std::future<boost::system::error_code>;

struct NetworkChannel : public std::enable_shared_from_this<NetworkChannel>
{
  NetworkChannel();
  virtual ~NetworkChannel();

  bool isConnected() const;

  // Receive messages //

  void registerHandler(uint8_t messageType, MessageHandler handler);
  void removeHandler(uint8_t messageType);
  void removeAllHandlers();

  // Send messages //

  MessageFuture send(Message &&msg);
  MessageFuture send(uint8_t type, StructuredMessage &&msg);

 protected:
  void start_messaging();
  void stop_messaging();

  void read_header();
  void read_payload();
  void invoke_handler();
  void log_asio_error(
      const boost::system::error_code &error, const char *context);

  asio::io_context m_io_context;
  std::thread m_io_thread;
  asio::executor_work_guard<asio::io_context::executor_type> m_work;

  tcp::socket m_socket;
  Message m_currentMessage;
  HandlerMap m_handlers;
};

struct NetworkServer : public NetworkChannel
{
  NetworkServer(short port);
  ~NetworkServer() override = default;

  void start();
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

// Inlined helper functions ///////////////////////////////////////////////////

template <typename R>
inline bool is_ready(const std::future<R> &f)
{
  return !f.valid()
      || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

} // namespace tsd::network
