# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Reusable Jupyter widget panels for TSD-based render servers.

These panels work with any :class:`~tsd_client.client.TSDClient` (or subclass)
and provide interactive controls for transfer functions, clip planes,
and scene tree inspection.
"""

from .transfer_function_panel import TransferFunctionPanel
from .clip_planes_panel import ClipPlanesPanel
from .datatree_panel import DataTreePanel
from .light_panel import LightPanel

__all__ = [
    "TransferFunctionPanel",
    "ClipPlanesPanel",
    "DataTreePanel",
    "LightPanel",
]
