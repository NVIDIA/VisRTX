# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Low-level TCP connection for the TSD binary protocol.

Handles socket management, the 8-byte framed message format, a background
receive thread, and a handler dispatch table.  This is the transport layer
that :class:`~tsd_client.client.TSDClient` builds upon.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
from typing import Callable

from .protocol import HEADER_FORMAT, HEADER_SIZE, MessageType

logger = logging.getLogger("tsd_client.connection")

HandlerFunc = Callable[[int, bytes], None]


class TSDConnection:
    """Thread-safe TCP connection to a TSD-based render server.

    Parameters
    ----------
    host : str
        Server hostname or IP address.
    port : int
        Server TCP port.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 12345):
        self._host = host
        self._port = port
        self._socket: socket.socket | None = None
        self._connected = False
        self._handlers: dict[int, HandlerFunc] = {}
        self._recv_thread: threading.Thread | None = None
        self._running = False
        self._send_lock = threading.Lock()
        self._on_disconnect: Callable[[], None] | None = None
        self._on_connect: Callable[[], None] | None = None

    # -- Properties ----------------------------------------------------------

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def on_disconnect(self) -> Callable[[], None] | None:
        return self._on_disconnect

    @on_disconnect.setter
    def on_disconnect(self, callback: Callable[[], None] | None):
        self._on_disconnect = callback

    @property
    def on_connect(self) -> Callable[[], None] | None:
        return self._on_connect

    @on_connect.setter
    def on_connect(self, callback: Callable[[], None] | None):
        self._on_connect = callback

    # -- Connection ----------------------------------------------------------

    def connect(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout: float = 5.0,
    ):
        """Open a TCP connection to the server.

        Parameters
        ----------
        host, port : str, int
            Override the values passed to the constructor.
        timeout : float
            TCP connection timeout in seconds.
        """
        if host is not None:
            self._host = host
        if port is not None:
            self._port = port

        if self._connected:
            self.disconnect()

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(timeout)
        self._socket.connect((self._host, self._port))
        self._socket.settimeout(None)

        self._connected = True
        self._running = True

        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="tsd-recv"
        )
        self._recv_thread.start()
        logger.info("Connected to %s:%d", self._host, self._port)

        if self._on_connect:
            try:
                self._on_connect()
            except Exception:
                logger.exception("on_connect callback failed")

    def disconnect(self):
        """Gracefully disconnect from the server."""
        self._running = False
        if self._socket:
            try:
                self.send(MessageType.DISCONNECT)
            except Exception:
                pass
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        self._connected = False
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=2.0)
        self._recv_thread = None
        logger.info("Disconnected")

    # -- Handler registry ----------------------------------------------------

    def register_handler(self, msg_type: int, handler: HandlerFunc):
        """Register ``handler(msg_type, payload_bytes)`` for a message type."""
        self._handlers[int(msg_type)] = handler

    def remove_handler(self, msg_type: int):
        """Remove a previously registered handler."""
        self._handlers.pop(int(msg_type), None)

    def get_handler(self, msg_type: int) -> HandlerFunc | None:
        """Return the registered handler for *msg_type*, if any."""
        return self._handlers.get(int(msg_type))

    # -- Sending -------------------------------------------------------------

    def send(self, msg_type: int, payload: bytes = b""):
        """Send a framed message with optional payload."""
        if not self._connected:
            return
        header = struct.pack(HEADER_FORMAT, int(msg_type), len(payload))
        with self._send_lock:
            try:
                self._socket.sendall(header + payload)
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                logger.error("Send error: %s", exc)
                self._connected = False

    # -- Receive loop --------------------------------------------------------

    def _recv_loop(self):
        """Background thread: read framed messages and dispatch to handlers."""
        while self._running and self._connected:
            try:
                header_data = self._recv_exact(HEADER_SIZE)
                if not header_data:
                    break

                msg_type, payload_length = struct.unpack(
                    HEADER_FORMAT, header_data
                )

                payload = b""
                if payload_length > 0:
                    payload = self._recv_exact(payload_length)
                    if payload is None:
                        break

                handler = self._handlers.get(msg_type)
                if handler:
                    try:
                        handler(msg_type, payload)
                    except Exception:
                        logger.exception(
                            "Handler error for message type %d", msg_type
                        )

            except (ConnectionResetError, BrokenPipeError, OSError) as exc:
                logger.debug("Receive loop ended: %s", exc)
                break

        was_connected = self._connected
        self._connected = False
        if was_connected and self._on_disconnect:
            try:
                self._on_disconnect()
            except Exception:
                pass
        logger.info("Receive loop ended")

    def _recv_exact(self, n: int) -> bytes | None:
        """Read exactly *n* bytes from the socket, or ``None`` on EOF."""
        data = bytearray()
        while len(data) < n:
            try:
                chunk = self._socket.recv(n - len(data))
                if not chunk:
                    return None
                data.extend(chunk)
            except (ConnectionResetError, BrokenPipeError, OSError):
                return None
        return bytes(data)
