// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "NetworkChannel.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"

namespace tsd::network {

// Helper functions ///////////////////////////////////////////////////////////

template <typename FCN>
static void async_invoke(boost::asio::io_context &io_context, FCN &&f)
{
  boost::asio::post(io_context, [f = std::forward<FCN>(f)]() mutable { f(); });
}

// NetworkChannel definitions /////////////////////////////////////////////////

NetworkChannel::NetworkChannel() : m_socket(m_io_context)
{
  m_io_context.stop();
}

NetworkChannel::~NetworkChannel()
{
  stop_messaging();
}

bool NetworkChannel::isConnected() const
{
  return m_socket.is_open();
}

void NetworkChannel::registerHandler(uint8_t type, MessageHandler handler)
{
  m_handlers[type] = handler;
}

void NetworkChannel::removeHandler(uint8_t messageType)
{
  m_handlers.erase(messageType);
}

void NetworkChannel::removeAllHandlers()
{
  m_handlers.clear();
}

MessageFuture NetworkChannel::send(Message &&msg)
{
  using MessagePromise = std::promise<boost::system::error_code>;
  auto promise = std::make_shared<MessagePromise>();
  auto future = promise->get_future();

  if (!isConnected()) {
    log_asio_error(asio::error::not_connected, "Send");
    promise->set_value(asio::error::not_connected); // Set error in promise
    return future;
  }

  auto self = shared_from_this();

  auto header = std::make_shared<Message::Header>();
  *header = msg.header;
  asio::async_write(m_socket,
      asio::buffer(header.get(), sizeof(Message::Header)),
      [self, header, promise](
          const boost::system::error_code &error, std::size_t) {
        self->log_asio_error(error, "Send(Header)");
        if (header->payload_length == 0)
          promise->set_value(error);
      });

  if (header->payload_length == 0)
    return future;

  auto payload = std::make_shared<MessagePayload>();
  *payload = std::move(msg.payload);
  asio::async_write(m_socket,
      asio::buffer(*payload),
      [self, payload, promise](
          const boost::system::error_code &error, std::size_t) {
        self->log_asio_error(error, "Send(Payload)");
        promise->set_value(error);
      });

  return future;
}

MessageFuture NetworkChannel::send(uint8_t type, StructuredMessage &&msg)
{
  Message message = msg.toMessage(type);
  return send(std::move(message));
}

MessageFuture NetworkChannel::send(uint8_t type)
{
  return send(make_message(type));
}

MessageFuture NetworkChannel::send(uint8_t type, const std::string &str)
{
  return send(make_message(type, str));
}

void NetworkChannel::start_messaging()
{
  tsd::core::logDebug("[NetworkChannel] starting channel");
  stop_messaging();
  m_work.emplace(asio::make_work_guard(m_io_context));
  m_io_context.restart();
  m_io_thread = std::thread([this]() {
    tsd::core::logDebug("[NetworkChannel] starting IO thread");
    try {
      m_io_context.run();
    } catch (const std::exception &e) {
      tsd::core::logError(
          "[NetworkChannel] IO thread context error: %s", e.what());
    } catch (...) {
      tsd::core::logError("[NetworkChannel] IO thread context unknown error");
    }
    tsd::core::logDebug("[NetworkChannel] IO thread stopped");
  });
}

void NetworkChannel::stop_messaging()
{
  try {
    if (m_socket.is_open()) {
      boost::system::error_code ec{};
      m_socket.shutdown(tcp::socket::shutdown_both, ec);
      m_socket.close(ec);
    }
    if (!m_io_context.stopped()) {
      m_io_context.stop();
      if (m_io_thread.joinable())
        m_io_thread.join();
      m_work.reset();
      m_io_context.restart();
      m_io_context.poll(); // drain any remaining tasks
      m_io_context.stop();
    }
  } catch (const std::system_error &e) {
    tsd::core::logError(
        "[NetworkChannel] System error during stop: %s", e.what());
  } catch (const std::exception &e) {
    tsd::core::logError("[NetworkChannel] Error during stop: %s", e.what());
  }
}

void NetworkChannel::read_header()
{
  if (!isConnected()) {
    tsd::core::logError("[NetworkChannel] Cannot read header: not connected");
    return;
  }

  auto message = std::make_shared<Message>();
  auto self = shared_from_this();
  asio::async_read(m_socket,
      asio::buffer(&message->header, sizeof(Message::Header)),
      [this, self, message](const boost::system::error_code &error,
          std::size_t bytes_transferred) {
        log_asio_error(error, "ReadHeader");
        if (!error)
          read_payload(message); // Read next message
      });
}

void NetworkChannel::read_payload(std::shared_ptr<Message> msg)
{
  if (!isConnected()) {
    tsd::core::logError("[NetworkChannel] Cannot read payload: not connected");
    return;
  }

  if (msg->header.payload_length == 0) {
    async_invoke(m_io_context, [this, msg]() {
      invoke_handler(msg);
      read_header(); // Read next msg
    });
    return;
  }

  msg->payload.resize(msg->header.payload_length);

  auto self = shared_from_this();
  asio::async_read(m_socket,
      asio::buffer(msg->payload.data(), msg->header.payload_length),
      [this, self, msg](const boost::system::error_code &error,
          std::size_t bytes_transferred) {
        log_asio_error(error, "ReadPayload");
        if (!error) {
          invoke_handler(msg);
          read_header(); // Read next msg
        }
      });
}

void NetworkChannel::invoke_handler(std::shared_ptr<Message> msg)
{
  // Invoke handler if registered
  if (auto *handler = m_handlers.at(msg->header.type); handler != nullptr) {
    (*handler)(*msg);
  } else {
    tsd::core::logWarning(
        "[NetworkChannel] No handler registered for message type %d",
        static_cast<int>(msg->header.type));
  }
}

void NetworkChannel::log_asio_error(
    const boost::system::error_code &error, const char *context)
{
  if (!error)
    return;

  if (error == asio::error::eof) {
    tsd::core::logStatus(
        "[NetworkChannel] %s: connection closed by peer", context);
  } else if (error == asio::error::connection_reset) {
    tsd::core::logStatus(
        "[NetworkChannel] %s: connection reset by peer", context);
  } else if (error == asio::error::not_connected) {
    tsd::core::logWarning("[NetworkChannel] %s: not connected", context);
  } else if (error) {
    tsd::core::logError(
        "[NetworkChannel] %s error: %s", context, error.message().c_str());
  }

  if (m_socket.is_open()) {
    boost::system::error_code ec{};
    m_socket.shutdown(tcp::socket::shutdown_both, ec);
    m_socket.close(ec);
  }
}

// NetworkServer definitions //////////////////////////////////////////////////

NetworkServer::NetworkServer(short port)
    : m_acceptor(m_io_context, tcp::endpoint(tcp::v4(), port))
{
  start_accept();
}

void NetworkServer::start()
{
  start_messaging();
}

void NetworkServer::restart()
{
  stop();
  start_accept();
  start();
}

void NetworkServer::stop()
{
  stop_messaging();
}

void NetworkServer::start_accept()
{
  auto socket = std::make_shared<tcp::socket>(m_io_context);
  m_acceptor.async_accept(
      *socket, [this, socket](const boost::system::error_code &error) {
        if (!error) {
          tsd::core::logStatus("[NetworkServer] New connection from %s",
              socket->remote_endpoint().address().to_string().c_str());
          m_socket = std::move(*socket);
          read_header();
          start_accept(); // Accept next connection
        }
      });
}

// NetworkClient definitions //////////////////////////////////////////////////

NetworkClient::NetworkClient(const std::string &host, short port)
{
  connect(host, port);
}

void NetworkClient::connect(const std::string &host, short port)
{
  start_messaging();
  asio::ip::tcp::resolver resolver(m_io_context);
  auto endpoints = resolver.resolve(host, std::to_string(port));
  asio::async_connect(m_socket,
      endpoints,
      [this](const boost::system::error_code &error, const tcp::endpoint &) {
        if (!error) {
          tsd::core::logStatus("[NetworkClient] Connected to server");
          read_header();
        } else {
          tsd::core::logError(
              "[NetworkClient] Connection error: %s", error.message().c_str());
        }
      });
}

void NetworkClient::disconnect()
{
  stop_messaging();
  tsd::core::logStatus("[NetworkClient] Disconnected from server");
}

} // namespace tsd::network
