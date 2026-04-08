# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared utility functions for the tsd-client package."""

from __future__ import annotations

import json
import logging
import math
from typing import Callable

_logger_jupyter = logging.getLogger("tsd_client.utils")


# ---------------------------------------------------------------------------
# Scalar math
# ---------------------------------------------------------------------------

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between *a* and *b*."""
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    """Hermite smoothstep for ease-in-out motion (clamps *t* to [0, 1])."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *v* to [*lo*, *hi*]."""
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# 3-D vector math
# ---------------------------------------------------------------------------

def normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalise a 3-D vector to unit length."""
    x, y, z = v
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 0.0:
        return (0.0, 0.0, 1.0)
    return (x / length, y / length, z / length)


def slerp3(
    t: float,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Spherical linear interpolation between two unit vectors."""
    a = normalize3(a)
    b = normalize3(b)
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    if dot < 0.0:
        b = (-b[0], -b[1], -b[2])
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    theta = math.acos(dot)
    if theta < 1e-7:
        return normalize3((
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
        ))
    sin_theta = math.sin(theta)
    w0 = math.sin((1.0 - t) * theta) / sin_theta
    w1 = math.sin(t * theta) / sin_theta
    return (
        w0 * a[0] + w1 * b[0],
        w0 * a[1] + w1 * b[1],
        w0 * a[2] + w1 * b[2],
    )


def slerp3_exact(
    t: float,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Slerp that preserves the sign of *b* (no short-path flip).

    Unlike :func:`slerp3`, this variant never negates *b*, so the
    interpolation always arrives at the original target.  Required for
    direction/up vectors and globe positions where *v* and *-v* are
    not equivalent.
    """
    a = normalize3(a)
    b = normalize3(b)
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    theta = math.acos(dot)
    if theta < 1e-7:
        return normalize3((
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
        ))
    sin_theta = math.sin(theta)
    w0 = math.sin((1.0 - t) * theta) / sin_theta
    w1 = math.sin(t * theta) / sin_theta
    return (
        w0 * a[0] + w1 * b[0],
        w0 * a[1] + w1 * b[1],
        w0 * a[2] + w1 * b[2],
    )


# ---------------------------------------------------------------------------
# Transfer-function file parsing
# ---------------------------------------------------------------------------

def parse_1dt(
    text: str,
) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float]]]:
    """Parse a ``.1dt`` file (one ``r g b a`` per line).

    Returns ``(color_map, opacity_pts)``.
    """
    color_map: list[tuple[float, float, float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                r, g, b, a = (
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                )
                color_map.append((r, g, b, a))
            except ValueError:
                continue
    if not color_map:
        return [], []
    opacity_pts = [(0.0, 0.0), (1.0, 1.0)]
    return color_map, opacity_pts


def parse_tf_json(
    text: str,
) -> tuple[
    list[tuple[float, float, float, float]],
    list[tuple[float, float]],
    tuple[float, float] | None,
    float | None,
    float | None,
]:
    """Parse a transfer-function JSON file (standard or ParaView ``RGBPoints``).

    Returns ``(color_map, opacity_pts, value_range, opacity, unit_distance)``.
    """
    data = json.loads(text)
    color_map: list[tuple[float, float, float, float]] = []
    opacity_pts: list[tuple[float, float]] = []
    value_range: tuple[float, float] | None = None
    opacity: float | None = None
    unit_distance: float | None = None

    if "RGBPoints" in data:
        pts = data["RGBPoints"]
        for i in range(0, len(pts) - 3, 4):
            r, g, b = float(pts[i + 1]), float(pts[i + 2]), float(pts[i + 3])
            color_map.append((r, g, b, 1.0))
        opacity_pts = [(0.0, 0.0), (1.0, 1.0)]
    else:
        for c in data.get("colorMap", data.get("colors", [])):
            if len(c) >= 4:
                color_map.append(
                    (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
                )
            elif len(c) >= 3:
                color_map.append(
                    (float(c[0]), float(c[1]), float(c[2]), 1.0)
                )
        for p in data.get("opacityControlPoints", []):
            if len(p) >= 2:
                opacity_pts.append((float(p[0]), float(p[1])))
        vr = data.get("valueRange")
        if vr and len(vr) >= 2:
            value_range = (float(vr[0]), float(vr[1]))
        if "opacity" in data:
            opacity = float(data["opacity"])
        if "unitDistance" in data:
            unit_distance = float(data["unitDistance"])

    if not opacity_pts:
        opacity_pts = [(0.0, 0.0), (1.0, 1.0)]
    return color_map, opacity_pts, value_range, opacity, unit_distance


# ---------------------------------------------------------------------------
# Jupyter (ipywidgets / anywidget)
# ---------------------------------------------------------------------------


def run_on_kernel_loop(fn: Callable[[], None]) -> bool:
    """Run *fn* on the Jupyter kernel IOLoop when available.

    Synced widget traits must be updated from the kernel thread; assigning
    from a worker thread (e.g. after ``scene_graph`` on a background thread)
    often never reaches the browser.  Returns True if *fn* was scheduled;
    if False, call *fn* yourself on the current thread (non-notebook use).
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        kernel = getattr(ip, "kernel", None) if ip else None
        loop = getattr(kernel, "io_loop", None) if kernel else None
        if loop is not None and hasattr(loop, "add_callback"):
            loop.add_callback(fn)
            return True
    except Exception:
        _logger_jupyter.debug("Kernel loop scheduling unavailable", exc_info=True)
    return False


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def guess_type(val) -> str | None:
    """Infer the wire param type string from a Python value."""
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int32"
    if isinstance(val, float):
        return "float32"
    if isinstance(val, str):
        return "string"
    if isinstance(val, (list, tuple)) and all(
        isinstance(x, (int, float)) for x in val
    ):
        n = len(val)
        if n == 2:
            return "float32_vec2"
        if n == 3:
            return "float32_vec3"
        if n == 4:
            return "float32_vec4"
    return None


def resolve_client(client_or_viewer):
    """Extract a :class:`TSDClient` from a client or viewer instance.

    Panels accept either a ``TSDClient`` (or subclass) or a viewer widget
    that exposes a ``.client`` attribute.  This helper transparently unwraps
    the viewer's ``.client`` attribute when a viewer is passed.
    """
    from .client import TSDClient  # noqa: deferred to avoid circular imports

    if isinstance(client_or_viewer, TSDClient):
        return client_or_viewer
    client = getattr(client_or_viewer, "client", None)
    if isinstance(client, TSDClient):
        return client
    raise TypeError(
        f"Expected TSDClient (or subclass) or a viewer with .client, "
        f"got {type(client_or_viewer).__name__}"
    )


def split_null_separated(buf: bytes) -> list[str]:
    """Split a null-separated byte buffer into a list of UTF-8 strings."""
    parts: list[str] = []
    start = 0
    for i, b in enumerate(buf):
        if b == 0:
            if i > start:
                parts.append(buf[start:i].decode("utf-8", errors="replace"))
            start = i + 1
    if start < len(buf):
        parts.append(buf[start:].decode("utf-8", errors="replace"))
    return parts


# ---------------------------------------------------------------------------
# DataNode → widget attribute conversion (shared by DataTreePanel / LightPanel)
# ---------------------------------------------------------------------------

def datanode_to_attr(name: str, node, read_only: bool) -> dict | None:
    """Convert a TSD parameter DataNode into a JSON-serializable attribute dict.

    Shared helper used by :class:`~tsd_client.panels.DataTreePanel` and
    :class:`~tsd_client.panels.LightPanel` to avoid duplicated logic.

    TSD parameters are nested structures::

        'paramName' (dtype=0, container)
            'value'       -> actual value (scalar, vec, string, object ref)
            'enabled'     -> bool
            'description' -> string tooltip
            'min' / 'max' -> range constraints
            'stringValues' -> enum options for string params

    If the node is a leaf (no ``value`` child), it is treated as a direct
    value node (used for metadata).
    """
    from .anari_types import anari_type_name as _atn

    value_child = node.child("value")
    vnode = value_child if value_child is not None else node

    if vnode.object_index is not None:
        return {
            "name": name,
            "type": "object_ref",
            "value": f"{_atn(vnode.dtype)}@{vnode.object_index}",
            "readOnly": True,
        }

    if vnode.is_array:
        return {
            "name": name,
            "type": "array",
            "value": f"[{vnode.array_count} \u00d7 {_atn(vnode.dtype)}]",
            "readOnly": True,
        }

    val = vnode.value
    if val is None:
        return None

    if isinstance(val, (bytes, bytearray)):
        return {
            "name": name,
            "type": "binary",
            "value": f"<{len(val)} bytes>",
            "readOnly": True,
        }

    enabled_child = node.child("enabled")
    if enabled_child is not None and enabled_child.value is False:
        read_only = True

    ptype = guess_type(val)
    if ptype is None:
        return {
            "name": name,
            "type": "unknown",
            "value": str(val),
            "readOnly": True,
        }

    if isinstance(val, tuple):
        val = [float(x) for x in val]

    entry: dict = {
        "name": name,
        "type": ptype,
        "value": val,
        "readOnly": read_only,
    }

    desc_child = node.child("description")
    if desc_child is not None and desc_child.value:
        entry["description"] = str(desc_child.value)

    min_child = node.child("min")
    max_child = node.child("max")
    if min_child is not None and max_child is not None:
        miv, mav = min_child.value, max_child.value
        if isinstance(miv, (int, float)) and isinstance(mav, (int, float)):
            entry["min"] = float(miv)
            entry["max"] = float(mav)

    sv_child = node.child("stringValues")
    if sv_child is not None and sv_child.children:
        opts = [
            c.value
            for c in sv_child.children.values()
            if isinstance(c.value, str)
        ]
        if opts:
            entry["stringValues"] = opts

    return entry
