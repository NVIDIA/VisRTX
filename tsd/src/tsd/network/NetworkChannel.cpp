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
  io_context.post([f = std::forward<FCN>(f)]() mutable { f(); });
}

// NetworkChannel definitions /////////////////////////////////////////////////

NetworkChannel::NetworkChannel()
    : m_work(asio::make_work_guard(m_io_context)), m_socket(m_io_context)
{}

NetworkChannel::~NetworkChannel()
{
  stop();
}

bool NetworkChannel::isConnected() const
{
  return m_socket.is_open();
}

void NetworkChannel::registerHandler(uint8_t type, MessageHandler handler)
{
  m_handlers[type] = handler;
}

void NetworkChannel::removeAllHandlers()
{
  m_handlers.clear();
}

void NetworkChannel::send(const Message &msg)
{
  if (!isConnected()) {
    tsd::core::logError("[NetworkChannel] Cannot send message: not connected");
    return;
  }

  auto self = shared_from_this();

  auto header = std::make_shared<Message::Header>();
  std::memcpy(header.get(), &msg.header, sizeof(Message::Header));

  asio::async_write(m_socket,
      asio::buffer(header.get(), sizeof(Message::Header)),
      [self, header](const boost::system::error_code &error, std::size_t) {
        self->log_asio_error(error, "Send(Header)");
      });

  if (msg.header.payload_length == 0)
    return;

  auto payload = std::make_shared<MessagePayload>();
  *payload = msg.payload;
  asio::async_write(m_socket,
      asio::buffer(*payload),
      [self, payload](const boost::system::error_code &error, std::size_t) {
        self->log_asio_error(error, "Send(Payload)");
      });
}

MessageFuture NetworkChannel::sendAsync(const Message &msg)
{
  auto promise = std::make_shared<std::promise<boost::system::error_code>>();
  auto future = promise->get_future();

  if (!isConnected()) {
    log_asio_error(asio::error::not_connected, "SendAsync");
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

  if (msg.header.payload_length == 0)
    return future;

  auto payload = std::make_shared<MessagePayload>();
  *payload = msg.payload;
  asio::async_write(m_socket,
      asio::buffer(*payload),
      [self, payload, promise](
          const boost::system::error_code &error, std::size_t) {
        self->log_asio_error(error, "Send(Payload)");
        promise->set_value(error);
      });

  return future;
}

void NetworkChannel::start()
{
  m_io_thread = std::thread([this]() { m_io_context.run(); });
}

void NetworkChannel::stop()
{
  m_io_context.stop();
  if (m_io_thread.joinable())
    m_io_thread.join();
}

void NetworkChannel::read_header()
{
  if (!isConnected()) {
    tsd::core::logError("[NetworkChannel] Cannot read header: not connected");
    return;
  }

  auto self = shared_from_this();
  asio::async_read(m_socket,
      asio::buffer(&m_currentMessage.header, sizeof(Message::Header)),
      [this, self](const boost::system::error_code &error,
          std::size_t bytes_transferred) {
        log_asio_error(error, "ReadHeader");
        if (!error)
          read_payload(); // Read next message
      });
}

void NetworkChannel::read_payload()
{
  if (!isConnected()) {
    tsd::core::logError("[NetworkChannel] Cannot read payload: not connected");
    return;
  }

  if (m_currentMessage.header.payload_length == 0) {
    async_invoke(m_io_context, [this]() {
      invoke_handler();
      read_header(); // Read next message
    });
    return;
  }

  m_currentMessage.payload.resize(m_currentMessage.header.payload_length);

  auto self = shared_from_this();
  asio::async_read(m_socket,
      asio::buffer(m_currentMessage.payload.data(),
          m_currentMessage.header.payload_length),
      [this, self](const boost::system::error_code &error,
          std::size_t bytes_transferred) {
        log_asio_error(error, "ReadPayload");
        if (!error) {
          invoke_handler();
          read_header(); // Read next message
        }
      });
}

void NetworkChannel::invoke_handler()
{
  Message &msg = m_currentMessage;
  // Invoke handler if registered
  if (auto *handler = m_handlers.at(msg.header.type); handler != nullptr) {
    (*handler)(msg);
  } else {
    tsd::core::logError(
        "[NetworkChannel] No handler registered for message type %d",
        static_cast<int>(msg.header.type));
  }
}

void NetworkChannel::log_asio_error(
    const boost::system::error_code &error, const char *context)
{
  if (error == asio::error::eof) {
    tsd::core::logStatus(
        "[NetworkChannel] %s: connection closed by peer", context);
  } else if (error == asio::error::connection_reset) {
    tsd::core::logStatus(
        "[NetworkChannel] %s: connection reset by peer", context);
  } else if (error == asio::error::not_connected) {
    tsd::core::logError("[NetworkChannel] %s: not connected", context);
  } else if (error) {
    tsd::core::logError(
        "[NetworkChannel] %s error: %s", context, error.message().c_str());
  }
}

// NetworkServer definitions //////////////////////////////////////////////////

NetworkServer::NetworkServer(short port)
    : m_acceptor(m_io_context, tcp::endpoint(tcp::v4(), port))
{
  start_accept();
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
        }
        read_header();
        start_accept(); // Accept next connection
      });
}

// NetworkClient definitions //////////////////////////////////////////////////

NetworkClient::NetworkClient(const std::string &host, short port)
{
  connect(host, port);
}

void NetworkClient::connect(const std::string &host, short port)
{
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
  boost::system::error_code ec{};
  m_socket.shutdown(tcp::socket::shutdown_both, ec);
  if (ec) {
    tsd::core::logError(
        "[NetworkClient] Shutdown error: %s", ec.message().c_str());
  }
  m_socket.close(ec);
  if (ec) {
    tsd::core::logError(
        "[NetworkClient] Close error: %s", ec.message().c_str());
  } else {
    tsd::core::logStatus("[NetworkClient] Disconnected from server");
  }
}

} // namespace tsd::network
