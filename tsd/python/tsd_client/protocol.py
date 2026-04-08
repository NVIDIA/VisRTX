# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
TSD network protocol constants.

Defines the base ``MessageType`` enum (matching ``RenderSession.hpp`` in
VisRTX/TSD) and the binary wire format constants shared by all TSD-based
servers.

Wire format
-----------
Header (8 bytes): ``<B3xI`` — 1 byte message type, 3 bytes padding,
4 bytes payload length (little-endian uint32).

Payload: variable-length raw bytes; interpretation depends on message type.
All multi-byte values are little-endian.
"""

import struct
from enum import IntEnum

# ---------------------------------------------------------------------------
# Wire format constants
# ---------------------------------------------------------------------------

HEADER_FORMAT = "<B3xI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 8

FRAME_CONFIG_FORMAT = "<2I"
FRAME_CONFIG_SIZE = struct.calcsize(FRAME_CONFIG_FORMAT)  # 8


# ---------------------------------------------------------------------------
# Base TSD message types  (must match RenderSession.hpp MessageType enum)
# ---------------------------------------------------------------------------

class MessageType(IntEnum):
    """Base TSD protocol message types.

    These values must match the ``tsd::network::MessageType`` enum defined in
    ``VisRTX/tsd/apps/interactive/network/RenderSession.hpp``.

    Application-specific extensions should use values >= 100 and be defined
    in a separate IntEnum subclass in the downstream project.
    """

    # Client → Server: set state
    SERVER_SHUTDOWN = 0
    SERVER_START_RENDERING = 1
    SERVER_STOP_RENDERING = 2
    SERVER_SET_FRAME_CONFIG = 3
    SERVER_SET_OBJECT_PARAMETER = 4
    SERVER_REMOVE_OBJECT_PARAMETER = 5
    SERVER_SET_CURRENT_RENDERER = 6
    SERVER_SET_CURRENT_CAMERA = 7
    SERVER_SET_ARRAY_DATA = 8
    SERVER_ADD_OBJECT = 9
    SERVER_REMOVE_OBJECT = 10
    SERVER_REMOVE_ALL_OBJECTS = 11
    SERVER_UPDATE_LAYER = 12
    SERVER_UPDATE_TIME = 13
    SERVER_SAVE_STATE_FILE = 14

    # Server → Client: push state
    CLIENT_RECEIVE_FRAME_BUFFER_COLOR = 15
    CLIENT_RECEIVE_FRAME_CONFIG = 16
    CLIENT_RECEIVE_CURRENT_RENDERER = 17
    CLIENT_RECEIVE_SCENE = 18
    CLIENT_RECEIVE_CURRENT_CAMERA = 19
    CLIENT_RECEIVE_TIME = 20
    CLIENT_SCENE_TRANSFER_BEGIN = 21

    # Client → Server: request state
    SERVER_REQUEST_FRAME_CONFIG = 22
    SERVER_REQUEST_CURRENT_RENDERER = 23
    SERVER_REQUEST_CURRENT_CAMERA = 24
    SERVER_REQUEST_SCENE = 25

    # Misc
    PING = 26
    DISCONNECT = 27

    # Application-specific extensions (>= 100)
    GH_SET_RENDERER_PARAM = 109
    GH_SET_CAMERA_ATTRIBUTE = 119
    GH_EXPORT_FRAME = 134
    GH_EXPORT_FRAME_RESULT = 135

    ERROR = 255
