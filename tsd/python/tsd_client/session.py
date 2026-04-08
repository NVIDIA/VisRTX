# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
High-level session helper that bundles client, viewer, and panels.

A :class:`TSDSession` manages the lifecycle of a :class:`TSDClient`,
an :class:`OrbitTSDViewer`, and any number of panels so notebook cells
can be concise and cleanup is automatic::

    from tsd_client import TSDSession

    session = TSDSession("127.0.0.1", 49000)
    display(session.viewer)
    display(session.dt_panel)

    # when done:
    session.disconnect()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .client import TSDClient
from .viewer import OrbitTSDViewer
from .panels import DataTreePanel, ClipPlanesPanel, TransferFunctionPanel, LightPanel

if TYPE_CHECKING:
    pass

logger = logging.getLogger("tsd_client.session")

ShowcaseViewer = OrbitTSDViewer


class TSDSession:
    """Convenience wrapper that owns a client, viewer, and panels.

    Parameters
    ----------
    host : str
        Server hostname or IP.
    port : int
        Server port.
    auto_connect : bool
        Connect to the server immediately (default ``True``).
    viewer_width, viewer_height : int
        Viewer canvas size in pixels.
    detachable : bool
        Allow the viewer to be detached into a popup window.
    panels : list[str] | None
        Which panels to create automatically.  Valid names are
        ``"datatree"``, ``"clip_planes"``, ``"lights"``, and
        ``"transfer_function"``.
        Defaults to ``["datatree", "clip_planes", "lights", "transfer_function"]``.
    dt_title : str
        Title for the DataTree panel.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 49000,
        *,
        auto_connect: bool = True,
        viewer_width: int = 1024,
        viewer_height: int = 480,
        detachable: bool = True,
        panels: list[str] | None = None,
        dt_title: str = "Scene (DataTree)",
    ):
        self.client = TSDClient(host, port, auto_connect=auto_connect)
        if auto_connect and not self.client.connected:
            raise RuntimeError(f"Could not connect to {host}:{port}")

        self.viewer: OrbitTSDViewer = OrbitTSDViewer(
            self.client,
            width=viewer_width,
            height=viewer_height,
            detachable=detachable,
        )

        if panels is None:
            panels = ["datatree", "clip_planes", "lights", "transfer_function"]

        self.dt_panel: DataTreePanel | None = None
        self.clip_panel: ClipPlanesPanel | None = None
        self.tf_panel: TransferFunctionPanel | None = None
        self.light_panel: LightPanel | None = None

        for name in panels:
            self._create_panel(name, dt_title=dt_title)

        logger.info("TSDSession connected to %s:%d", host, port)

    def _create_panel(self, name: str, **kwargs) -> None:
        if name == "datatree":
            self.dt_panel = DataTreePanel(
                self.client, title=kwargs.get("dt_title", "Scene (DataTree)")
            )
        elif name == "clip_planes":
            self.clip_panel = ClipPlanesPanel(self.client)
        elif name == "lights":
            self.light_panel = LightPanel(self.client)
        elif name == "transfer_function":
            self.tf_panel = TransferFunctionPanel(self.client)
        else:
            raise ValueError(f"Unknown panel name: {name!r}")

    @property
    def scene_graph(self):
        """Shortcut to ``self.client.scene_graph``."""
        return self.client.scene_graph

    def disconnect(self) -> None:
        """Stop rendering, close all widgets, and drop the TCP connection."""
        if self.viewer is not None:
            try:
                self.viewer.disconnect()
            except Exception:
                pass

        for widget in (self.viewer, self.dt_panel, self.clip_panel, self.light_panel, self.tf_panel):
            if widget is not None:
                try:
                    widget.close()
                except Exception:
                    pass

        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                pass

        self.viewer = None
        self.dt_panel = None
        self.clip_panel = None
        self.light_panel = None
        self.tf_panel = None
        self.client = None
        logger.info("TSDSession disconnected")

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self.client is None:
            return "TSDSession(disconnected)"
        return (
            f"TSDSession({self.client.host}:{self.client.port}, "
            f"connected={self.client.connected})"
        )
