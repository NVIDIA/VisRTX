# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""TSDViewer — base Jupyter widget and client for TSD server interaction."""

from .tsd_client import (
    TSDClient,
    MessageType,
    HEADER_FORMAT,
    HEADER_SIZE,
    VIEW_FORMAT,
    VIEW_SIZE,
    FRAME_CONFIG_FORMAT,
    FRAME_CONFIG_SIZE,
)
from .viewer import TSDViewer
from .transfer_function import TransferFunctionWidget

__all__ = [
    "TSDClient",
    "TSDViewer",
    "TransferFunctionWidget",
    "MessageType",
    "HEADER_FORMAT",
    "HEADER_SIZE",
    "VIEW_FORMAT",
    "VIEW_SIZE",
    "FRAME_CONFIG_FORMAT",
    "FRAME_CONFIG_SIZE",
]
