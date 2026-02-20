# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Low-level TSD binary protocol TCP client.

Implements the wire protocol used by the TSD network server:
  Header (8 bytes): uint8 type | 3 bytes padding | uint32 payload_length
  Payload: variable-length raw bytes

All multi-byte values are little-endian (native x86_64).
"""

import json
import socket
import struct
import threading
import logging
from enum import IntEnum

logger = logging.getLogger("tsdjupyter.tsd_client")

# ---------------------------------------------------------------------------
# Wire format constants
# ---------------------------------------------------------------------------

# C++ struct Message::Header { uint8_t type; uint32_t payload_length; }
# On x86_64: sizeof = 8 (1 + 3 padding + 4)
HEADER_FORMAT = "<B3xI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 8

# C++ struct View { float3 azeldist; float3 lookat; }  → 6 floats
VIEW_FORMAT = "<6f"
VIEW_SIZE = struct.calcsize(VIEW_FORMAT)  # 24

# C++ struct Config { uint2 size; }  → 2 uint32s
FRAME_CONFIG_FORMAT = "<2I"
FRAME_CONFIG_SIZE = struct.calcsize(FRAME_CONFIG_FORMAT)  # 8


class MessageType(IntEnum):
    """Base TSD protocol message type IDs."""

    # Client → Server: set state
    SERVER_SHUTDOWN = 0
    SERVER_START_RENDERING = 1
    SERVER_STOP_RENDERING = 2
    SERVER_SET_FRAME_CONFIG = 3
    SERVER_SET_VIEW = 4
    SERVER_SET_OBJECT_PARAMETER = 5
    SERVER_REMOVE_OBJECT_PARAMETER = 6
    SERVER_SET_ARRAY_DATA = 7
    SERVER_ADD_OBJECT = 8
    SERVER_REMOVE_OBJECT = 9
    SERVER_REMOVE_ALL_OBJECTS = 10
    SERVER_UPDATE_LAYER = 11

    # Server → Client: data
    CLIENT_RECEIVE_FRAME_BUFFER_COLOR = 12
    CLIENT_RECEIVE_FRAME_CONFIG = 13
    CLIENT_RECEIVE_SCENE = 14
    CLIENT_RECEIVE_VIEW = 15
    CLIENT_SCENE_TRANSFER_BEGIN = 16

    # Client → Server: requests
    SERVER_REQUEST_FRAME_CONFIG = 17
    SERVER_REQUEST_VIEW = 18
    SERVER_REQUEST_SCENE = 19

    # Bidirectional
    PING = 20
    DISCONNECT = 21
    ERROR = 255

    # Volume management (110–118)
    REQUEST_VOLUME_LIST = 110
    VOLUME_LIST = 111
    SET_VOLUME_TF = 112
    REQUEST_VOLUME_INFO = 116
    VOLUME_INFO = 117
    SET_VOLUME_ATTRIBUTE = 118


# ---------------------------------------------------------------------------
# TSD Client
# ---------------------------------------------------------------------------


class TSDClient:
    """Thread-safe TCP client for the TSD binary protocol."""

    def __init__(self):
        self._socket: socket.socket | None = None
        self._connected = False
        self._handlers: dict[int, callable] = {}
        self._recv_thread: threading.Thread | None = None
        self._running = False
        self._send_lock = threading.Lock()
        self._on_disconnect: callable | None = None

    # -- Properties ----------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def on_disconnect(self):
        return self._on_disconnect

    @on_disconnect.setter
    def on_disconnect(self, callback):
        self._on_disconnect = callback

    # -- Connection ----------------------------------------------------------

    def connect(self, host: str, port: int, timeout: float = 5.0):
        """Connect to a TSD server."""
        if self._connected:
            self.disconnect()

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(timeout)
        self._socket.connect((host, port))
        self._socket.settimeout(None)

        self._connected = True
        self._running = True
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="tsd-recv"
        )
        self._recv_thread.start()
        logger.info("Connected to %s:%d", host, port)

    def disconnect(self):
        """Gracefully disconnect from the server."""
        self._running = False
        if self._socket:
            try:
                self._send_raw(MessageType.DISCONNECT)
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

    # -- Message handlers ----------------------------------------------------

    def register_handler(self, msg_type: int, handler: callable):
        """Register a callback ``handler(msg_type, payload_bytes)``."""
        self._handlers[int(msg_type)] = handler

    def remove_handler(self, msg_type: int):
        self._handlers.pop(int(msg_type), None)

    # -- Sending -------------------------------------------------------------

    def send(self, msg_type: int, payload: bytes = b""):
        """Send a message with optional payload."""
        if not self._connected:
            return
        self._send_raw(msg_type, payload)

    def send_frame_config(self, width: int, height: int):
        self.send(
            MessageType.SERVER_SET_FRAME_CONFIG,
            struct.pack(FRAME_CONFIG_FORMAT, width, height),
        )

    def send_view(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
        lookat: tuple[float, float, float],
    ):
        self.send(
            MessageType.SERVER_SET_VIEW,
            struct.pack(
                VIEW_FORMAT,
                azimuth,
                elevation,
                distance,
                lookat[0],
                lookat[1],
                lookat[2],
            ),
        )

    def start_rendering(self):
        self.send(MessageType.SERVER_START_RENDERING)

    def stop_rendering(self):
        self.send(MessageType.SERVER_STOP_RENDERING)

    def request_view(self):
        self.send(MessageType.SERVER_REQUEST_VIEW)

    def request_frame_config(self):
        self.send(MessageType.SERVER_REQUEST_FRAME_CONFIG)

    def ping(self):
        self.send(MessageType.PING)

    def request_scene(self, timeout: float = 30.0) -> bytes:
        """Request the current scene from the server.

        Blocks until CLIENT_RECEIVE_SCENE is received or timeout. The payload
        is the serialized scene (opaque binary).

        Returns
        -------
        bytes
            Raw scene payload. Empty bytes if not connected or on error.
        """
        if not self._connected:
            return b""
        result: list[bytes | None] = [None]
        event = threading.Event()
        msg_key = int(MessageType.CLIENT_RECEIVE_SCENE)
        prev_handler = self._handlers.get(msg_key)

        def on_response(_msg_type: int, payload: bytes) -> None:
            result[0] = bytes(payload) if payload else b""
            _restore()
            event.set()

        def _restore():
            if prev_handler is not None:
                self._handlers[msg_key] = prev_handler
            else:
                self._handlers.pop(msg_key, None)

        self.register_handler(MessageType.CLIENT_RECEIVE_SCENE, on_response)
        self.send(MessageType.SERVER_REQUEST_SCENE)
        if not event.wait(timeout=timeout):
            _restore()
            raise TimeoutError(
                f"Scene request timed out after {timeout}s"
            )
        return result[0] if result[0] is not None else b""

    # -- Volume management ---------------------------------------------------

    def request_volume_list(self, timeout: float = 5.0) -> list:
        """Request the list of volumes (lightweight: index + name only).

        Returns
        -------
        list
            List of ``{"index": int, "name": str}`` dicts.
        """
        if not self._connected:
            return []
        result: list[list | None] = [None]
        event = threading.Event()

        def on_response(_msg_type: int, payload: bytes) -> None:
            try:
                text = payload.decode("utf-8", errors="replace").rstrip("\x00")
                data = json.loads(text)
                result[0] = list(data) if isinstance(data, list) else []
            except (json.JSONDecodeError, ValueError):
                result[0] = []
            self.remove_handler(MessageType.VOLUME_LIST)
            event.set()

        self.register_handler(MessageType.VOLUME_LIST, on_response)
        self.send(MessageType.REQUEST_VOLUME_LIST)
        if not event.wait(timeout=timeout):
            self.remove_handler(MessageType.VOLUME_LIST)
            raise TimeoutError(
                f"Volume list request timed out after {timeout}s"
            )
        return result[0] if result[0] is not None else []

    def request_volume_info(self, volume_index: int, timeout: float = 5.0) -> dict:
        """Request full attribute info for a single volume (blocking).

        Parameters
        ----------
        volume_index : int
            Index of the volume (from ``request_volume_list``).

        Returns
        -------
        dict
            Volume attributes including parameters, metadata, colors, and
            ``field`` sub-object.
        """
        if not self._connected:
            return {}
        result: list[dict | None] = [None]
        event = threading.Event()

        def on_response(_msg_type: int, payload: bytes) -> None:
            try:
                text = payload.decode("utf-8", errors="replace").rstrip("\x00")
                data = json.loads(text)
                result[0] = data if isinstance(data, dict) else {}
            except (json.JSONDecodeError, ValueError):
                result[0] = {}
            self.remove_handler(MessageType.VOLUME_INFO)
            event.set()

        self.register_handler(MessageType.VOLUME_INFO, on_response)
        self.send(MessageType.REQUEST_VOLUME_INFO,
                  struct.pack("<I", volume_index))
        if not event.wait(timeout=timeout):
            self.remove_handler(MessageType.VOLUME_INFO)
            raise TimeoutError(
                f"Volume info request timed out after {timeout}s"
            )
        return result[0] if result[0] is not None else {}

    # ANARI data type enum values (mirrors the C++ ANARIDataType defines)
    _ANARI_TYPE_MAP = {
        "bool": 103,      # ANARI_BOOL
        "int32": 1016,    # ANARI_INT32
        "float32": 1068,  # ANARI_FLOAT32
        "string": 101,    # ANARI_STRING
    }

    def set_volume_attribute(
        self, volume_index: int, name: str, param_type: str, value
    ):
        """Set a parameter on a volume (or its spatial field).

        Parameters
        ----------
        volume_index : int
            Index of the volume in the scene.
        name : str
            Parameter name (e.g. ``"elevationScale"``).
        param_type : str
            One of ``"bool"``, ``"int32"``, ``"float32"``, ``"string"``.
        value
            Value matching the type.
        """
        anari_type = self._ANARI_TYPE_MAP.get(param_type)
        if anari_type is None:
            raise ValueError(f"Unknown param_type: {param_type}")

        name_bytes = (name + "\0").encode("utf-8")[:64].ljust(64, b"\x00")
        vol_b = struct.pack("<I", volume_index)
        type_b = struct.pack("<I", anari_type)
        if param_type == "bool":
            payload = vol_b + name_bytes + type_b + struct.pack("<B", 1 if value else 0)
        elif param_type == "int32":
            payload = vol_b + name_bytes + type_b + struct.pack("<i", int(value))
        elif param_type == "float32":
            payload = vol_b + name_bytes + type_b + struct.pack("<f", float(value))
        elif param_type == "string":
            val_bytes = (str(value) + "\0").encode("utf-8")[:64].ljust(64, b"\x00")
            payload = vol_b + name_bytes + type_b + val_bytes
        self.send(MessageType.SET_VOLUME_ATTRIBUTE, payload)

    def set_volume_tf(
        self,
        volume_index: int,
        color_rgba: list[tuple[float, float, float, float]],
        opacity_xy: list[tuple[float, float]],
        value_range: tuple[float, float] | None = None,
        opacity: float | None = None,
        unit_distance: float | None = None,
    ):
        """Set transfer function for a volume.

        Parameters
        ----------
        volume_index : int
            Index of the volume in the scene (from volume list).
        color_rgba : list of (r, g, b, a)
            Color samples (e.g. 256 entries); each 0–1.
        opacity_xy : list of (x, y)
            Opacity control points (x = value position, y = opacity), 0–1.
        value_range : (float, float) or None
            Optional value range override.
        opacity : float or None
            Optional global opacity scalar.
        unit_distance : float or None
            Optional unit distance.
        """
        num_samples = len(color_rgba)
        num_opacity = len(opacity_xy)
        if num_samples == 0 or num_samples > 4096 or num_opacity == 0:
            raise ValueError(
                "color_rgba length 1–4096 and at least one opacity point required"
            )
        payload = bytearray(struct.pack("<II", volume_index, num_samples))
        for r, g, b, a in color_rgba:
            payload.extend(struct.pack("<4f", r, g, b, a))
        payload.extend(struct.pack("<I", num_opacity))
        for x, y in opacity_xy:
            payload.extend(struct.pack("<2f", x, y))

        # Flags bitmask: bit 0 = valueRange, bit 1 = opacity, bit 2 = unitDistance
        flags = 0
        if value_range is not None:
            flags |= 0x1
        if opacity is not None:
            flags |= 0x2
        if unit_distance is not None:
            flags |= 0x4
        payload.extend(struct.pack("<I", flags))

        if value_range is not None:
            payload.extend(struct.pack("<2f", value_range[0], value_range[1]))
        if opacity is not None:
            payload.extend(struct.pack("<f", opacity))
        if unit_distance is not None:
            payload.extend(struct.pack("<f", unit_distance))
        self.send(MessageType.SET_VOLUME_TF, bytes(payload))

    # -- Internal ------------------------------------------------------------

    def _send_raw(self, msg_type: int, payload: bytes = b""):
        header = struct.pack(HEADER_FORMAT, int(msg_type), len(payload))
        with self._send_lock:
            try:
                self._socket.sendall(header + payload)
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                logger.error("Send error: %s", exc)
                self._connected = False

    def _recv_loop(self):
        """Background thread: read messages and dispatch to handlers."""
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
