# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Base TSD client with scene graph management.

:class:`TSDClient` wraps :class:`~tsd_client.connection.TSDConnection` and adds
scene graph caching, the standard request/response patterns from the base TSD
protocol, and layer update handling.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import threading

from .connection import TSDConnection
from .protocol import MessageType, FRAME_CONFIG_FORMAT
from .scene import SceneGraph

logger = logging.getLogger("tsd_client")


class TSDClient:
    """Thread-safe TCP client for a TSD-based render server.

    Parameters
    ----------
    host : str
        Server hostname or IP address.
    port : int
        Server TCP port (default 12345).
    auto_connect : bool
        If True (default), connect immediately.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 12345,
        auto_connect: bool = True,
    ):
        self._conn = TSDConnection(host, port)

        # Last frame config sent (for restoring after export)
        self._frame_config: tuple[int, int] | None = None

        # Scene graph cache
        self._scene_graph: SceneGraph | None = None
        self._scene_stale = True
        self._scene_version = 0
        self._on_scene_changed: callable | None = None
        # ``request_scene`` registers a one-shot CLIENT_RECEIVE_SCENE handler;
        # concurrent calls overwrite each other and steal the wrong payload.
        self._scene_request_lock = threading.Lock()

        if auto_connect:
            self.connect()

    # -- Delegated connection properties ------------------------------------

    @property
    def host(self) -> str:
        return self._conn.host

    @property
    def port(self) -> int:
        return self._conn.port

    @property
    def connected(self) -> bool:
        return self._conn.connected

    @property
    def on_disconnect(self):
        return self._conn.on_disconnect

    @on_disconnect.setter
    def on_disconnect(self, callback):
        self._conn.on_disconnect = callback

    @property
    def on_connect(self):
        return self._conn.on_connect

    @on_connect.setter
    def on_connect(self, callback):
        self._conn.on_connect = callback

    # -- Scene graph properties ---------------------------------------------

    @property
    def scene_graph(self) -> SceneGraph | None:
        """Cached scene graph, auto-fetched on first access."""
        if self._scene_stale or self._scene_graph is None:
            self.refresh_scene()
        return self._scene_graph

    def cached_scene_graph(self) -> SceneGraph | None:
        """Return the scene graph only if already loaded and fresh.

        Does not send ``SERVER_REQUEST_SCENE``. Use this on hot paths (e.g.
        orbit updates) so a scene pull cannot race with rendering mode on the
        server.
        """
        if self._scene_stale or self._scene_graph is None:
            return None
        return self._scene_graph

    @property
    def scene_version(self) -> int:
        return self._scene_version

    @property
    def on_scene_changed(self):
        return self._on_scene_changed

    @on_scene_changed.setter
    def on_scene_changed(self, callback):
        self._on_scene_changed = callback

    # -- Connection ----------------------------------------------------------

    def connect(self, host: str | None = None, port: int | None = None, timeout: float = 5.0):
        """Connect to the server."""
        self._conn.connect(host, port, timeout)

        self.register_handler(
            MessageType.SERVER_UPDATE_LAYER,
            self._handle_server_layer_update,
        )

        self._scene_graph = None
        self._scene_stale = True

    def disconnect(self):
        """Gracefully disconnect."""
        self._conn.disconnect()

    # -- Handler registry (delegated) ----------------------------------------

    def register_handler(self, msg_type: int, handler):
        self._conn.register_handler(msg_type, handler)

    def remove_handler(self, msg_type: int):
        self._conn.remove_handler(msg_type)

    # -- Sending (base protocol) ---------------------------------------------

    def send(self, msg_type: int, payload: bytes = b""):
        """Send a message with optional payload."""
        self._conn.send(msg_type, payload)

    def send_frame_config(self, width: int, height: int):
        self._frame_config = (int(width), int(height))
        self.send(
            MessageType.SERVER_SET_FRAME_CONFIG,
            struct.pack(FRAME_CONFIG_FORMAT, width, height),
        )

    def start_rendering(self):
        self.send(MessageType.SERVER_START_RENDERING)

    def stop_rendering(self):
        self.send(MessageType.SERVER_STOP_RENDERING)

    def request_frame_config(self):
        self.send(MessageType.SERVER_REQUEST_FRAME_CONFIG)

    def ping(self):
        self.send(MessageType.PING)

    # -- Renderer parameters -------------------------------------------------

    _RENDERER_PARAM_TYPES = {"bool": 0, "int32": 1, "float32": 2, "string": 3}

    def set_renderer_param(
        self,
        name: str,
        value,
        param_type: str | None = None,
    ) -> None:
        """Set a parameter on the active renderer.

        Parameters
        ----------
        name : str
            Parameter name (e.g. ``"pixelSamples"``, ``"denoise"``,
            ``"ambientRadiance"``).
        value
            New value.
        param_type : str, optional
            One of ``'bool'``, ``'int32'``, ``'float32'``, ``'string'``.
            Inferred from *value* if omitted.
        """
        if param_type is None:
            if isinstance(value, bool):
                param_type = "bool"
            elif isinstance(value, int):
                param_type = "int32"
            elif isinstance(value, float):
                param_type = "float32"
            elif isinstance(value, str):
                param_type = "string"
            else:
                raise ValueError(
                    f"Cannot infer param_type for {name!r}={value!r}; "
                    f"pass param_type explicitly"
                )
        type_code = self._RENDERER_PARAM_TYPES.get(param_type)
        if type_code is None:
            raise ValueError(
                f"Unknown param_type {param_type!r}; "
                f"expected one of: {', '.join(self._RENDERER_PARAM_TYPES)}"
            )
        name_bytes = (name + "\0").encode("utf-8")[:64].ljust(64, b"\x00")
        type_b = struct.pack("<B", type_code)
        if param_type == "bool":
            payload = name_bytes + type_b + struct.pack("<B", 1 if value else 0)
        elif param_type == "int32":
            payload = name_bytes + type_b + struct.pack("<i", int(value))
        elif param_type == "float32":
            payload = name_bytes + type_b + struct.pack("<f", float(value))
        elif param_type == "string":
            val_b = (str(value) + "\0").encode("utf-8")[:64].ljust(64, b"\x00")
            payload = name_bytes + type_b + val_b
        self.send(MessageType.GH_SET_RENDERER_PARAM, payload)

    # -- Camera parameters ---------------------------------------------------

    def set_camera_attribute(
        self,
        name: str,
        value,
        param_type: str | None = None,
        camera_index: int | None = None,
    ) -> None:
        """Set an ANARI camera parameter on the server.

        Uses ``SERVER_SET_OBJECT_PARAMETER`` targeting the camera in the
        TSD scene object pool.

        Parameters
        ----------
        name : str
            Parameter name (e.g. ``"position"``, ``"direction"``, ``"up"``,
            ``"fovy"``, ``"apertureRadius"``).
        value
            Value matching the type.
        param_type : str, optional
            Wire type (e.g. ``'float32_vec3'``).  Inferred if omitted.
        camera_index : int, optional
            Camera pool index.  If *None*, uses the first camera found
            in the scene graph (usually index 0).
        """
        from .anari_types import ANARI_CAMERA
        from .scene import build_parameter_change_payload, _infer_param_type

        if param_type is None:
            param_type = _infer_param_type(value)
        if param_type is None:
            raise ValueError(
                f"Cannot infer param_type for {name!r}={value!r}; "
                f"pass param_type explicitly"
            )
        if camera_index is None:
            sg = self.cached_scene_graph()
            if sg is not None and sg.cameras:
                camera_index = sg.cameras[0].object_index
            else:
                camera_index = 0
        sg = self.cached_scene_graph()
        wire_size_t = sg._wire_size_t if sg else 8
        payload = build_parameter_change_payload(
            ANARI_CAMERA,
            camera_index,
            [(name, param_type, value)],
            wire_size_t=wire_size_t,
        )
        self.send(MessageType.SERVER_SET_OBJECT_PARAMETER, payload)

    def set_camera_pose(
        self,
        position: tuple[float, float, float],
        direction: tuple[float, float, float],
        up: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        """Set camera position, direction, and up vector in one call.

        Parameters
        ----------
        position : (float, float, float)
            Camera eye position.
        direction : (float, float, float)
            View direction vector.
        up : (float, float, float)
            Up vector (default Y-up).
        """
        self.set_camera_attribute("position", position)
        self.set_camera_attribute("direction", direction)
        self.set_camera_attribute("up", up)

    # -- Scene management ----------------------------------------------------

    def request_scene(self, timeout: float = 30.0) -> bytes:
        """Request the current scene from the server (blocking)."""
        if not self.connected:
            return b""
        with self._scene_request_lock:
            result: list[bytes | None] = [None]
            event = threading.Event()
            prev = self._conn.get_handler(MessageType.CLIENT_RECEIVE_SCENE)

            def on_response(_msg_type: int, payload: bytes) -> None:
                result[0] = bytes(payload) if payload else b""
                if prev is not None:
                    self.register_handler(MessageType.CLIENT_RECEIVE_SCENE, prev)
                else:
                    self.remove_handler(MessageType.CLIENT_RECEIVE_SCENE)
                event.set()

            self.register_handler(MessageType.CLIENT_RECEIVE_SCENE, on_response)
            self.send(MessageType.SERVER_REQUEST_SCENE)
            if not event.wait(timeout=timeout):
                if prev is not None:
                    self.register_handler(MessageType.CLIENT_RECEIVE_SCENE, prev)
                else:
                    self.remove_handler(MessageType.CLIENT_RECEIVE_SCENE)
                raise TimeoutError(f"Scene request timed out after {timeout}s")
            # VisRTX-style servers can return to PAUSED after SEND_SCENE if they did
            # not treat the connection as actively rendering in that loop iteration.
            self.start_rendering()
            return result[0] if result[0] is not None else b""

    def request_scene_graph(self, timeout: float = 30.0) -> SceneGraph:
        """Request and parse the scene into a :class:`SceneGraph`."""
        raw = self.request_scene(timeout=timeout)
        return SceneGraph.from_bytes(raw)

    def refresh_scene(self, timeout: float = 30.0) -> None:
        """Force a full re-fetch of the scene graph."""
        if not self.connected:
            return
        try:
            sg = self.request_scene_graph(timeout=timeout)
        except (TimeoutError, Exception):
            logger.exception("Failed to refresh scene graph")
            return
        sg._client = self
        self._scene_graph = sg
        self._scene_stale = False
        self._scene_version += 1
        self._fire_scene_changed(None)

    def invalidate_scene(self) -> None:
        """Mark the cached scene graph as stale."""
        self._scene_stale = True

    def send_layer_update(self, scene_graph: SceneGraph, layer_name: str):
        """Send a modified layer back to the server.

        The local scene graph is assumed to have been mutated in-place already
        (e.g. via ``_apply_srt`` or ``set_object_visible``), so the cached
        scene is **not** invalidated — avoiding an expensive full scene
        round-trip that would freeze rendering.
        """
        payload = scene_graph.build_layer_payload(layer_name)
        if payload is None:
            logger.warning("Layer '%s' not found in scene graph", layer_name)
            return
        self.send(MessageType.SERVER_UPDATE_LAYER, payload)

    # -- Transfer-function file I/O -----------------------------------------

    def save_volume_tf(
        self,
        volume_index: int,
        path: str,
        timeout: float = 5.0,
    ) -> str:
        """Fetch the current transfer function for a volume and save it.

        Writes standard JSON (``colorMap``, ``opacityControlPoints``, and
        optional ``valueRange`` / ``opacity`` / ``unitDistance``).

        Parameters
        ----------
        volume_index : int
            Volume pool index.
        path : str
            Local file path to write (e.g. ``"tf.json"``).
        timeout : float
            Seconds to wait for volume info from the scene graph.

        Returns
        -------
        str
            The *path* written.
        """
        if not self.connected:
            raise RuntimeError("Not connected to server")
        sg = self.scene_graph
        if sg is None:
            raise RuntimeError("No scene graph available")
        info = sg.volume_info(volume_index)
        if info is None:
            raise ValueError(f"Volume {volume_index} not found in scene")
        color_map = info.get("colors") or info.get("colorMap") or []
        opacity_pts = info.get("opacityControlPoints") or []
        doc: dict = {
            "colorMap": color_map,
            "opacityControlPoints": opacity_pts,
        }
        if "valueRange" in info:
            doc["valueRange"] = info["valueRange"]
        if "opacity" in info:
            doc["opacity"] = info["opacity"]
        if "unitDistance" in info:
            doc["unitDistance"] = info["unitDistance"]
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        return path

    def load_volume_tf(self, volume_index: int, path: str) -> str:
        """Load a transfer function from a file and apply it to a volume.

        Supported formats:

        - ``.1dt`` — one ``r g b a`` per line.
        - Standard JSON — ``colorMap``, ``opacityControlPoints``, etc.
        - ParaView JSON — ``RGBPoints`` flat array.

        Parameters
        ----------
        volume_index : int
            Volume pool index.
        path : str
            Local file path to read.

        Returns
        -------
        str
            The *path* read.
        """
        from .utils import parse_1dt as _parse_1dt, parse_tf_json as _parse_tf_json

        if not self.connected:
            raise RuntimeError("Not connected to server")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        path_lower = path.lower()
        value_range = None
        opacity = None
        unit_dist = None
        if path_lower.endswith(".1dt"):
            color_map, opacity_pts = _parse_1dt(text)
        elif (
            path_lower.endswith(".json")
            or "colorMap" in text
            or "RGBPoints" in text
            or "{" in text
        ):
            color_map, opacity_pts, value_range, opacity, unit_dist = (
                _parse_tf_json(text)
            )
        else:
            color_map, opacity_pts = _parse_1dt(text)
        if not color_map or not opacity_pts:
            raise ValueError(
                f"Failed to parse transfer function from {path!r}: "
                "no colorMap/opacityControlPoints or .1dt data"
            )
        sg = self.cached_scene_graph() or self.scene_graph
        if sg is None:
            raise RuntimeError("No scene graph available")
        sg.set_volume_tf(
            volume_index,
            color_map,
            opacity_pts,
            value_range=value_range,
            opacity=opacity,
            unit_distance=unit_dist,
        )
        return path

    # -- Frame export --------------------------------------------------------

    AOV_COLOR = 0
    AOV_DEPTH = 1
    AOV_ALBEDO = 2
    AOV_NORMAL = 3
    AOV_EDGES = 4
    AOV_OBJECT_ID = 5
    AOV_PRIMITIVE_ID = 6
    AOV_INSTANCE_ID = 7

    AOV_NAMES = {
        "color": AOV_COLOR,
        "depth": AOV_DEPTH,
        "albedo": AOV_ALBEDO,
        "normal": AOV_NORMAL,
        "edges": AOV_EDGES,
        "objectId": AOV_OBJECT_ID,
        "object_id": AOV_OBJECT_ID,
        "primitiveId": AOV_PRIMITIVE_ID,
        "primitive_id": AOV_PRIMITIVE_ID,
        "instanceId": AOV_INSTANCE_ID,
        "instance_id": AOV_INSTANCE_ID,
    }

    def export_frame(
        self,
        path: str,
        width: int = 1920,
        height: int = 1080,
        spp: int = 128,
        aov: str = "color",
        timeout: float = 60.0,
        *,
        depth_min: float | None = None,
        depth_max: float | None = None,
        invert: bool = False,
    ) -> str:
        """Request the server to render the current view at the given size and
        samples per pixel, then save the result to a local JPEG file.

        Parameters
        ----------
        path : str
            Local filesystem path where the JPEG will be written (e.g. ``"frame.jpg"``).
        width : int
            Export width in pixels (1-8192). Default 1920.
        height : int
            Export height in pixels (1-8192). Default 1080.
        spp : int
            Samples per pixel (1-65536). Default 128.
        aov : str
            AOV channel to export. One of ``'color'`` (default), ``'depth'``,
            ``'albedo'``, ``'normal'``, ``'edges'``, ``'objectId'``,
            ``'primitiveId'``, ``'instanceId'``.
        timeout : float
            Max seconds to wait for the server response. Default 60.
        depth_min : float or None
            Minimum depth value for ``'depth'`` AOV normalization. When *None*
            (default), the server auto-ranges from the actual depth buffer.
        depth_max : float or None
            Maximum depth value for ``'depth'`` AOV normalization. When *None*
            (default), the server auto-ranges from the actual depth buffer.
        invert : bool
            For ``'edges'`` AOV: if *True*, invert edges (black edges on white
            background instead of white on black). Default *False*.

        Returns
        -------
        str
            The same *path* (for chaining).

        Raises
        ------
        TimeoutError
            If the server does not respond with the frame within *timeout*.
        ValueError
            If *aov* is not a recognized AOV name.
        """
        if not self.connected:
            raise RuntimeError("Not connected to server")
        aov_code = self.AOV_NAMES.get(aov)
        if aov_code is None:
            raise ValueError(
                f"Unknown AOV {aov!r}; expected one of: "
                + ", ".join(sorted(set(self.AOV_NAMES.keys())))
            )
        width = max(1, min(8192, int(width)))
        height = max(1, min(8192, int(height)))
        spp = max(1, min(65536, int(spp)))

        p0 = 0.0
        p1 = 0.0
        if aov_code == self.AOV_DEPTH:
            p0 = float(depth_min) if depth_min is not None else 0.0
            p1 = float(depth_max) if depth_max is not None else 0.0
        elif aov_code == self.AOV_EDGES:
            p0 = 1.0 if invert else 0.0

        result_path: list[str | None] = [None]
        event = threading.Event()

        def on_response(_msg_type: int, payload: bytes) -> None:
            try:
                with open(path, "wb") as f:
                    f.write(payload)
                result_path[0] = path
            except OSError as e:
                logger.exception("Failed to write export frame to %s: %s", path, e)
            self.remove_handler(MessageType.GH_EXPORT_FRAME_RESULT)
            event.set()

        self.register_handler(MessageType.GH_EXPORT_FRAME_RESULT, on_response)
        self.send(
            MessageType.GH_EXPORT_FRAME,
            struct.pack("<IIiBff", width, height, spp, aov_code, p0, p1),
        )
        if not event.wait(timeout=timeout):
            self.remove_handler(MessageType.GH_EXPORT_FRAME_RESULT)
            self._restore_frame_config()
            raise TimeoutError(
                f"Export frame request timed out after {timeout}s"
            )
        self._restore_frame_config()
        return result_path[0] if result_path[0] is not None else path

    def _restore_frame_config(self) -> None:
        """Re-send the last interactive frame config to the server."""
        if self._frame_config is not None and self.connected:
            self.send_frame_config(*self._frame_config)

    def _fire_scene_changed(self, layer_name: str | None) -> None:
        if self._on_scene_changed:
            try:
                self._on_scene_changed(self._scene_graph, layer_name)
            except Exception:
                logger.exception("on_scene_changed callback failed")

    def _handle_server_layer_update(self, _msg_type: int, payload: bytes) -> None:
        if self._scene_graph is None:
            self._scene_stale = True
            return
        layer_name = self._scene_graph.apply_layer_update(payload)
        if layer_name is not None:
            self._scene_stale = False
            self._scene_version += 1
            logger.debug("Server pushed layer update: %s", layer_name)
            self._fire_scene_changed(layer_name)
        else:
            self._scene_stale = True
