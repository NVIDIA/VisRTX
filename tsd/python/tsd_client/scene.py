# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
High-level scene graph introspection and manipulation.

Provides :class:`SceneGraph` for navigating layers and object pools,
:class:`TransformRef` for mutating transform nodes, and :class:`ObjectRef`
for reading/writing object parameters — all generic to any TSD-based server.
"""

from __future__ import annotations

import json
import logging
import math
import struct
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .anari_types import (
    ANARI_UNKNOWN, ANARI_STRING,
    ANARI_BOOL, ANARI_INT32, ANARI_UINT32,
    ANARI_FLOAT32, ANARI_FLOAT32_VEC2, ANARI_FLOAT32_VEC3, ANARI_FLOAT32_VEC4,
    ANARI_FLOAT32_MAT3,
    ANARI_ARRAY1D,
    ANARI_CAMERA, ANARI_GEOMETRY, ANARI_LIGHT, ANARI_MATERIAL,
    ANARI_RENDERER,
    ANARI_SAMPLER, ANARI_SPATIAL_FIELD, ANARI_SURFACE, ANARI_VOLUME,
    anari_type_name,
    is_object_type as _is_object_type,
    element_size as _element_size,
    unpack_value as _unpack_value,
)
from .datatree import DataNode, DataTree
from .utils import guess_type as _guess_type

if TYPE_CHECKING:
    from .client import TSDClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_node_value_for_info(node: DataNode) -> Any:
    """Return a JSON-serializable value from a DataNode, or None to skip."""
    if node.object_index is not None:
        return None
    if node.is_array and isinstance(node.value, (bytes, bytearray)):
        try:
            sz = _element_size(node.dtype)
            if sz > 0 and node.array_count > 0:
                items = []
                raw = bytes(node.value)
                for i in range(node.array_count):
                    off = i * sz
                    if off + sz > len(raw):
                        break
                    val = _unpack_value(raw[off : off + sz], node.dtype)
                    if isinstance(val, (bytes, bytearray)):
                        break
                    items.append(val)
                if items:
                    return items
        except (struct.error, ValueError, KeyError):
            pass
        return None
    if node.value is not None and not isinstance(node.value, (bytes, bytearray)):
        return node.value
    if isinstance(node.value, (bytes, bytearray)) and node.dtype != ANARI_UNKNOWN:
        try:
            sz = _element_size(node.dtype)
            if sz > 0 and len(node.value) >= sz:
                val = _unpack_value(bytes(node.value[:sz]), node.dtype)
                if not isinstance(val, (bytes, bytearray)):
                    return val
        except (struct.error, ValueError, KeyError):
            pass
    return None


def _collect_leaf_params(obj_node: DataNode) -> dict[str, Any]:
    """Walk an object's DataNode and collect all leaf key-value pairs."""
    result: dict[str, Any] = {}
    skip_names = frozenset({"name", "self", "subtype"})

    def add_leaves(node: DataNode) -> None:
        if not node.children:
            if node.name in skip_names:
                return
            val = _data_node_value_for_info(node)
            if val is not None:
                result[node.name or ""] = val
            return
        for child in node.children.values():
            add_leaves(child)

    for group in ("parameters", "metadata", "params"):
        group_node = obj_node.child(group)
        if group_node is not None:
            add_leaves(group_node)
    return result


_DTYPE_TO_PARAM_TYPE: dict[int, str] = {
    ANARI_BOOL: "bool",
    ANARI_INT32: "int32",
    ANARI_UINT32: "int32",
    ANARI_FLOAT32: "float32",
    ANARI_STRING: "string",
    ANARI_FLOAT32_VEC2: "float32_vec2",
    ANARI_FLOAT32_VEC3: "float32_vec3",
    ANARI_FLOAT32_VEC4: "float32_vec4",
}

_PARAM_TYPE_TO_SETTER: dict[str, str] = {
    "bool": "set_bool",
    "int32": "set_int",
    "float32": "set_float",
    "string": "set_string",
    "float32_vec2": "set_vec2",
    "float32_vec3": "set_vec3",
    "float32_vec4": "set_vec4",
}

# C++ ``tsd::core::INVALID_INDEX`` (~size_t(0)) on the wire
_INVALID_POOL_U64 = (1 << 64) - 1
_INVALID_POOL_U32 = (1 << 32) - 1


def is_valid_object_pool_index(idx: int | None) -> bool:
    """False for None or invalid sentinel indices from the server."""
    if idx is None:
        return False
    if idx == _INVALID_POOL_U64 or idx == _INVALID_POOL_U32:
        return False
    return idx >= 0


def build_parameter_change_payload(
    object_type: int,
    object_index: int,
    params: list[tuple[str, str, Any]],
    *,
    wire_size_t: int = 8,
) -> bytes:
    """Serialize a VisRTX ``ParameterChange`` message body (DataTree only).

    Matches ``tsd::network::messages::ParameterChange`` in VisRTX: root ``o``
    is the object handle, ``p`` is an array of ``{n, v{value,enabled}}``.

    *wire_size_t* must match the scene DataTree on the wire (4 vs 8). A
    mismatch corrupts the payload and the server will not apply parameters.
    """
    tree = DataTree()
    tree._size_t_bytes = int(wire_size_t) if wire_size_t in (4, 8) else 8
    tree.root["o"].set_object(object_type, object_index)
    ps = tree.root["p"]
    for name, param_type, value in params:
        entry = ps.append()
        entry["n"].set_string(name)
        v = entry["v"]
        v["enabled"].set_bool(True)
        if param_type == "object_ref":
            ref_type, ref_idx = value
            v["value"].set_object(ref_type, ref_idx)
        else:
            setter = _PARAM_TYPE_TO_SETTER.get(param_type)
            if setter is None:
                continue
            getattr(v["value"], setter)(value)
    return tree.to_bytes()


def _infer_param_type(value) -> str | None:
    """Infer the wire param_type from a Python value.

    Delegates to :func:`~tsd_client.utils.guess_type`.
    """
    return _guess_type(value)


# ---------------------------------------------------------------------------
# ObjectInfo
# ---------------------------------------------------------------------------

@dataclass
class ObjectInfo:
    """Parsed representation of an ANARI object from the scene's object DB."""

    object_type: int = ANARI_UNKNOWN
    object_index: int = 0
    name: str = ""
    subtype: str = ""
    parameters: dict[str, DataNode] = field(default_factory=dict)
    metadata: dict[str, DataNode] = field(default_factory=dict)
    _data_node: DataNode | None = field(default=None, repr=False)

    @property
    def type_name(self) -> str:
        return anari_type_name(self.object_type)

    def get_parameter(self, name: str, default=None):
        node = self.parameters.get(name)
        if node is None:
            return default
        vnode = self._param_value_node(node)
        return vnode.value if vnode.value is not None else default

    def get_metadata(self, name: str, default=None):
        node = self.metadata.get(name)
        if node is None:
            return default
        return node.value

    @staticmethod
    def _param_value_node(node: DataNode) -> DataNode:
        """Return the leaf holding the actual value for a parameter.

        TSD serializes each parameter as a parent node with children
        ``value`` and ``enabled``.  If the node has a ``value`` child,
        return that; otherwise assume the node itself carries the value
        (e.g. metadata or flat parameter updates).
        """
        child = node.child("value")
        return child if child is not None else node

    def params_dict(self) -> dict[str, Any]:
        """Return all parameters as ``{name: value}`` (skipping object refs)."""
        result: dict[str, Any] = {}
        for name, node in self.parameters.items():
            val = _data_node_value_for_info(self._param_value_node(node))
            if val is not None:
                result[name] = val
        return result

    def metadata_dict(self) -> dict[str, Any]:
        """Return all metadata as ``{name: value}``."""
        result: dict[str, Any] = {}
        for name, node in self.metadata.items():
            val = _data_node_value_for_info(node)
            if val is not None:
                result[name] = val
        return result

    def linked_object_index(self, param_name: str) -> int | None:
        """Get the object index referenced by a parameter."""
        node = self.parameters.get(param_name)
        if node is None:
            return None
        vnode = self._param_value_node(node)
        if vnode.object_index is not None:
            return vnode.object_index
        return None

    def __repr__(self):
        return (
            f"ObjectInfo({self.type_name}@{self.object_index}, "
            f"name={self.name!r}, subtype={self.subtype!r})"
        )


# ---------------------------------------------------------------------------
# LayerNodeInfo
# ---------------------------------------------------------------------------

@dataclass
class LayerNodeInfo:
    """Parsed representation of a single node in a TSD layer tree."""

    name: str = ""
    is_transform: bool = False
    transform_srt: tuple | None = None
    object_type: int = ANARI_UNKNOWN
    object_index: int | None = None
    enabled: bool = True
    children: list[LayerNodeInfo] = field(default_factory=list)
    _data_node: DataNode | None = field(default=None, repr=False)

    @property
    def type_name(self) -> str:
        return anari_type_name(self.object_type)

    @property
    def is_object(self) -> bool:
        return self.object_index is not None and _is_object_type(self.object_type)


# ---------------------------------------------------------------------------
# TransformRef
# ---------------------------------------------------------------------------

class TransformRef:
    """Mutable reference to a transform node in a layer.

    Modify :attr:`position`, :attr:`rotation`, :attr:`scale`, then call
    :meth:`commit` to send the updated layer to the server.
    """

    __slots__ = ("_scene_graph", "_layer_name", "_info")

    def __init__(self, scene_graph: SceneGraph, layer_name: str, info: LayerNodeInfo):
        if not info.is_transform or info._data_node is None:
            raise ValueError("TransformRef requires a transform node with _data_node")
        self._scene_graph = scene_graph
        self._layer_name = layer_name
        self._info = info

    @property
    def layer_name(self) -> str:
        return self._layer_name

    @property
    def name(self) -> str:
        return self._info.name

    def _get_srt(self) -> tuple[float, ...]:
        srt = self._info.transform_srt
        if srt is None or len(srt) != 9:
            return (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return srt

    def _set_srt(
        self,
        scale: tuple[float, float, float] | None = None,
        rotation_deg: tuple[float, float, float] | None = None,
        position: tuple[float, float, float] | None = None,
    ):
        srt = list(self._get_srt())
        if scale is not None:
            srt[0], srt[1], srt[2] = scale
        if rotation_deg is not None:
            srt[3], srt[4], srt[5] = rotation_deg
        if position is not None:
            srt[6], srt[7], srt[8] = position
        SceneGraph._apply_srt(
            self._info,
            (srt[0], srt[1], srt[2]),
            (srt[3], srt[4], srt[5]),
            (srt[6], srt[7], srt[8]),
        )

    @property
    def position(self) -> tuple[float, float, float]:
        srt = self._get_srt()
        return (srt[6], srt[7], srt[8])

    @position.setter
    def position(self, value: tuple[float, float, float]):
        self._set_srt(position=value)

    @property
    def rotation(self) -> tuple[float, float, float]:
        """Euler angles in degrees (azimuth/Y, elevation/X, roll/Z)."""
        srt = self._get_srt()
        return (srt[3], srt[4], srt[5])

    @rotation.setter
    def rotation(self, value: tuple[float, float, float]):
        self._set_srt(rotation_deg=value)

    @property
    def scale(self) -> tuple[float, float, float]:
        srt = self._get_srt()
        return (srt[0], srt[1], srt[2])

    @scale.setter
    def scale(self, value: tuple[float, float, float]):
        self._set_srt(scale=value)

    def set_srt(
        self,
        *,
        position: tuple[float, float, float] | None = None,
        rotation_deg: tuple[float, float, float] | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        """Set position, rotation (degrees), and/or scale in one call."""
        self._set_srt(scale=scale, rotation_deg=rotation_deg, position=position)

    def commit(self, client: TSDClient) -> None:
        """Send the updated layer to the server."""
        client.send_layer_update(self._scene_graph, self._layer_name)

    def __repr__(self):
        return f"TransformRef(layer={self._layer_name!r}, name={self.name!r})"


# ---------------------------------------------------------------------------
# ObjectRef
# ---------------------------------------------------------------------------

class ObjectRef:
    """Mutable reference to a scene object with parameter read/write.

    Parameters are sent to the server immediately when assigned via
    ``__setitem__``, provided a client is attached to the scene graph.
    """

    __slots__ = ("_scene_graph", "_info")

    def __init__(self, scene_graph: SceneGraph, info: ObjectInfo):
        self._scene_graph = scene_graph
        self._info = info

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def subtype(self) -> str:
        return self._info.subtype

    @property
    def object_type(self) -> int:
        return self._info.object_type

    @property
    def object_index(self) -> int:
        return self._info.object_index

    @property
    def type_name(self) -> str:
        return self._info.type_name

    @property
    def parameters(self) -> dict[str, DataNode]:
        return self._info.parameters

    @property
    def metadata(self) -> dict[str, DataNode]:
        return self._info.metadata

    def get_parameter(self, name: str, default=None):
        node = self._info.parameters.get(name)
        if node is None:
            return default
        vnode = ObjectInfo._param_value_node(node)
        return vnode.value if vnode.value is not None else default

    def set_parameter(self, name: str, value, param_type: str | None = None):
        """Set a parameter, updating the local DataTree and notifying the server.

        Parameters
        ----------
        name : str
            Parameter name.
        value
            New value.
        param_type : str, optional
            Wire type string; inferred if omitted.
        """
        if param_type is None:
            existing = self._info.parameters.get(name)
            if existing is not None:
                enode = ObjectInfo._param_value_node(existing)
                param_type = _DTYPE_TO_PARAM_TYPE.get(enode.dtype)
            if param_type is None:
                param_type = _infer_param_type(value)
            if param_type is None:
                raise ValueError(
                    f"Cannot infer param_type for {name!r}={value!r}; "
                    f"pass param_type explicitly"
                )

        pnode = self._info.parameters.get(name)
        if pnode is None:
            pnode = DataNode(name=name)
            self._info.parameters[name] = pnode
            if self._info._data_node is not None:
                params_node = self._info._data_node.child("parameters")
                if params_node is not None:
                    params_node.children[name] = pnode

        node = ObjectInfo._param_value_node(pnode)
        setter = _PARAM_TYPE_TO_SETTER.get(param_type)
        if setter:
            getattr(node, setter)(value)

        self._scene_graph._send_object_parameter(
            self._info.object_type,
            self._info.object_index,
            name,
            param_type,
            value,
        )

    def __getitem__(self, name: str):
        return self.get_parameter(name)

    def __setitem__(self, name: str, value):
        self.set_parameter(name, value)

    def __repr__(self):
        return (
            f"ObjectRef({self._info.type_name}@{self._info.object_index}, "
            f"name={self._info.name!r})"
        )


# ---------------------------------------------------------------------------
# SceneGraph
# ---------------------------------------------------------------------------

class SceneGraph:
    """High-level view of a TSD scene received from the server.

    Construct from a :class:`~tsd_client.client.TSDClient`, raw ``bytes``,
    or a :class:`DataTree`::

        sg = SceneGraph(client)       # fetch from server
        sg = client.scene_graph       # cached, auto-refreshed

    The scene contains:

    - **Layers** — hierarchical trees of transform/object nodes.
    - **Object DB** — pools of ANARI objects (geometry, material, volume, ...).
    """

    ANARI_VOLUME = ANARI_VOLUME
    ANARI_LIGHT = ANARI_LIGHT
    ANARI_SURFACE = ANARI_SURFACE
    ANARI_CAMERA = ANARI_CAMERA
    ANARI_SPATIAL_FIELD = ANARI_SPATIAL_FIELD
    ANARI_GEOMETRY = ANARI_GEOMETRY
    ANARI_MATERIAL = ANARI_MATERIAL
    ANARI_SAMPLER = ANARI_SAMPLER

    def __init__(self, source):
        self._client: TSDClient | None = None
        if isinstance(source, DataTree):
            self._tree = source
        elif isinstance(source, (bytes, bytearray)):
            self._tree = DataTree.from_bytes(source)
        else:
            self._client = source
            raw = source.request_scene()
            self._tree = DataTree.from_bytes(raw)
        self._layers: dict[str, DataNode] = {}
        self._parsed_layers: dict[str, LayerNodeInfo] = {}
        self._objects: dict[tuple[int, int], ObjectInfo] = {}
        self._parse()
        self._wire_size_t = int(getattr(self._tree, "_size_t_bytes", 8) or 8)
        self._next_array_offset = 0

    @classmethod
    def from_bytes(cls, data: bytes | bytearray) -> SceneGraph:
        return cls(DataTree.from_bytes(data))

    # ==================================================================
    # Layer queries
    # ==================================================================

    @property
    def layer_names(self) -> list[str]:
        return list(self._layers.keys())

    def layer_tree(self, layer_name: str) -> LayerNodeInfo | None:
        return self._parsed_layers.get(layer_name)

    def find_node(self, layer_name: str, node_name: str) -> LayerNodeInfo | None:
        root = self._parsed_layers.get(layer_name)
        if root is None:
            return None
        return self._find_node(root, node_name, transforms_only=False)

    def find_transform(self, layer_name: str, node_name: str) -> LayerNodeInfo | None:
        root = self._parsed_layers.get(layer_name)
        if root is None:
            return None
        return self._find_node(root, node_name, transforms_only=True)

    def transform(self, layer_name: str, node_name: str) -> TransformRef | None:
        """Get a mutable reference to a transform node."""
        info = self.find_transform(layer_name, node_name)
        if info is None:
            return None
        return TransformRef(self, layer_name, info)

    def transform_for_object(
        self, object_type: int, object_index: int, layer_name: str | None = None
    ) -> TransformRef | None:
        """Get a mutable reference to the transform that parents the given object."""
        result = self.find_parent_transform(object_type, object_index, layer_name)
        if result is None:
            return None
        lname, info = result
        return TransformRef(self, lname, info)

    def find_parent_transform(
        self,
        object_type: int,
        object_index: int,
        layer_name: str | None = None,
    ) -> tuple[str, LayerNodeInfo] | None:
        layers = (
            [(layer_name, self._parsed_layers[layer_name])]
            if layer_name and layer_name in self._parsed_layers
            else list(self._parsed_layers.items())
        )
        for lname, lroot in layers:
            result = self._find_parent_transform_impl(lroot, object_type, object_index)
            if result is not None:
                return (lname, result)
        return None

    def find_objects_in_layer(
        self, layer_name: str, object_type: int | None = None
    ) -> list[LayerNodeInfo]:
        root = self._parsed_layers.get(layer_name)
        if root is None:
            return []
        results: list[LayerNodeInfo] = []
        self._collect_objects(root, object_type, results)
        return results

    # ==================================================================
    # Object DB queries
    # ==================================================================

    def get_object(self, object_type: int, object_index: int) -> ObjectInfo | None:
        return self._objects.get((object_type, object_index))

    def objects_by_type(self, object_type: int) -> list[ObjectInfo]:
        return [obj for (t, _), obj in self._objects.items() if t == object_type]

    @property
    def volumes(self) -> list[ObjectInfo]:
        return self.objects_by_type(ANARI_VOLUME)

    @property
    def lights(self) -> list[ObjectInfo]:
        return self.objects_by_type(ANARI_LIGHT)

    @property
    def surfaces(self) -> list[ObjectInfo]:
        return self.objects_by_type(ANARI_SURFACE)

    @property
    def cameras(self) -> list[ObjectInfo]:
        return self.objects_by_type(ANARI_CAMERA)

    @property
    def renderers(self) -> list[ObjectInfo]:
        return self.objects_by_type(ANARI_RENDERER)

    @property
    def spatial_fields(self) -> list[ObjectInfo]:
        return self.objects_by_type(ANARI_SPATIAL_FIELD)

    @property
    def geometries(self) -> list[ObjectInfo]:
        return self.objects_by_type(ANARI_GEOMETRY)

    @property
    def materials(self) -> list[ObjectInfo]:
        return self.objects_by_type(ANARI_MATERIAL)

    @property
    def samplers(self) -> list[ObjectInfo]:
        return self.objects_by_type(ANARI_SAMPLER)

    # ==================================================================
    # Scene bounds
    # ==================================================================

    def compute_bounds(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Approximate world-space axis-aligned bounding box.

        Walks the layer tree collecting transform positions.  When the
        bounding box is degenerate (single point or no transforms at all),
        falls back to the camera's ``position`` parameter: the server
        places the camera at ``1.25 * diagonal`` from the world center,
        so the camera distance gives a usable size estimate.

        Returns ``(min_corner, max_corner)`` or ``None`` when the scene
        has no usable spatial data.
        """
        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3

        for root in self._parsed_layers.values():
            self._bounds_walk(root, lo, hi)

        if lo[0] > hi[0]:
            cam = self._camera_position()
            if cam is not None:
                r = math.sqrt(cam[0] ** 2 + cam[1] ** 2 + cam[2] ** 2)
                if r > 1e-6:
                    half = r / 1.25 / 2.0
                    return ((-half, -half, -half), (half, half, half))
            return None

        diag = [hi[i] - lo[i] for i in range(3)]
        if all(d < 1e-6 for d in diag):
            cam = self._camera_position()
            if cam is not None:
                r = math.sqrt(cam[0] ** 2 + cam[1] ** 2 + cam[2] ** 2)
                if r > 1e-6:
                    half = r / 1.25 / 2.0
                    cx, cy, cz = lo[0], lo[1], lo[2]
                    return (
                        (cx - half, cy - half, cz - half),
                        (cx + half, cy + half, cz + half),
                    )

        return (tuple(lo), tuple(hi))  # type: ignore[return-value]

    def _camera_position(self) -> tuple[float, float, float] | None:
        """Read position from the first camera in the object DB."""
        for cam in self.cameras:
            pos_node = cam.parameters.get("position")
            if pos_node is None:
                continue
            val_node = pos_node.child("value")
            v = (val_node or pos_node).value
            if isinstance(v, tuple) and len(v) == 3:
                return v
        return None

    def _bounds_walk(
        self,
        node: LayerNodeInfo,
        lo: list[float],
        hi: list[float],
    ) -> None:
        if node.is_transform and node.transform_srt:
            srt = node.transform_srt
            px, py, pz = srt[6], srt[7], srt[8]
            lo[0] = min(lo[0], px)
            lo[1] = min(lo[1], py)
            lo[2] = min(lo[2], pz)
            hi[0] = max(hi[0], px)
            hi[1] = max(hi[1], py)
            hi[2] = max(hi[2], pz)

        for child in node.children:
            self._bounds_walk(child, lo, hi)

    # ==================================================================
    # Object references (mutable)
    # ==================================================================

    def object_ref(self, object_type: int, object_index: int) -> ObjectRef | None:
        info = self.get_object(object_type, object_index)
        if info is None:
            return None
        return ObjectRef(self, info)

    def volume(self, index: int) -> ObjectRef | None:
        return self.object_ref(ANARI_VOLUME, index)

    def light(self, index: int) -> ObjectRef | None:
        return self.object_ref(ANARI_LIGHT, index)

    def renderer(self, index: int = 0) -> ObjectRef | None:
        return self.object_ref(ANARI_RENDERER, index)

    def camera(self, index: int = 0) -> ObjectRef | None:
        return self.object_ref(ANARI_CAMERA, index)

    def _resolve_volume_color(self, vol: ObjectInfo, info: dict) -> None:
        """Populate ``colorStops`` in *info* from the volume's color parameter.

        Handles four cases:
        1. Scalar float3 color → uniform color gradient
        2. Object-ref to array with full data → sample color stops
        3. Object-ref to proxy array + ``_tfColorStops`` parameter → restore
        4. Object-ref to proxy array, no fallback → leave unset
        """
        arr_idx = vol.linked_object_index("color")

        if arr_idx is None:
            c = info.get("color")
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                info["colorStops"] = [
                    {"t": 0.0, "r": c[0], "g": c[1], "b": c[2]},
                    {"t": 1.0, "r": c[0], "g": c[1], "b": c[2]},
                ]
            return

        rgba = self._read_color_array(vol)
        if rgba is not None:
            info["colorStops"] = self._rgba_to_color_stops(rgba)
            info["colorMap"] = rgba
            return

        saved = info.get("_tfColorStops")
        if isinstance(saved, str):
            try:
                stops = json.loads(saved)
                if isinstance(stops, list) and len(stops) >= 2:
                    info["colorStops"] = stops
            except (json.JSONDecodeError, TypeError):
                pass

    @staticmethod
    def _resolve_volume_opacity(info: dict) -> None:
        """Ensure ``opacityControlPoints`` is populated in *info*.

        Priority:
        1. ``opacityControlPoints`` from metadata with >= 2 valid entries
        2. ``_tfOpacityPoints`` custom parameter (JSON string)
        """
        log = logging.getLogger("tsd_client.scene")

        existing = info.get("opacityControlPoints")
        if isinstance(existing, list) and len(existing) >= 2:
            log.debug("_resolve_volume_opacity: using %d pts from metadata",
                       len(existing))
            return

        saved = info.pop("_tfOpacityPoints", None)
        if isinstance(saved, str):
            try:
                pts = json.loads(saved)
                if isinstance(pts, list) and len(pts) >= 2:
                    info["opacityControlPoints"] = pts
                    log.debug("_resolve_volume_opacity: restored %d pts "
                              "from _tfOpacityPoints", len(pts))
                    return
            except (json.JSONDecodeError, TypeError):
                pass

        log.debug("_resolve_volume_opacity: no opacity points found")

    def _read_color_array(self, vol: ObjectInfo) -> list[list[float]] | None:
        """Follow the volume's ``color`` object-ref to the array and unpack RGBA."""
        arr_idx = vol.linked_object_index("color")
        if arr_idx is None:
            return None
        arr_obj = self._objects.get((ANARI_ARRAY1D, arr_idx))
        if arr_obj is None:
            for at in (504, 505, 506):
                arr_obj = self._objects.get((at, arr_idx))
                if arr_obj is not None:
                    break
        if arr_obj is None or arr_obj._data_node is None:
            return None
        ad = arr_obj._data_node.child("arrayData")
        if ad is None or not ad.is_array or not isinstance(ad.value, (bytes, bytearray)):
            return None
        elem_sz = _element_size(ad.dtype)
        if elem_sz < 12:
            return None
        n_comp = elem_sz // 4
        raw = bytes(ad.value)
        count = min(ad.array_count, len(raw) // elem_sz)
        if count < 2:
            return None
        result: list[list[float]] = []
        for i in range(count):
            off = i * elem_sz
            floats = struct.unpack_from(f"<{n_comp}f", raw, off)
            if n_comp >= 4:
                result.append([floats[0], floats[1], floats[2], floats[3]])
            else:
                result.append([floats[0], floats[1], floats[2], 1.0])
        return result

    @staticmethod
    def _rgba_to_color_stops(
        rgba: list[list[float]], max_stops: int = 17,
    ) -> list[dict]:
        """Downsample an RGBA ramp to a set of color stops for the UI."""
        n = len(rgba)
        if n <= max_stops:
            indices = list(range(n))
        else:
            indices = [round(i * (n - 1) / (max_stops - 1)) for i in range(max_stops)]
        stops = []
        for i in indices:
            t = i / (n - 1) if n > 1 else 0.0
            c = rgba[i]
            stops.append({"t": round(t, 6), "r": c[0], "g": c[1], "b": c[2]})
        return stops

    def volume_info(self, volume_index: int) -> dict | None:
        """Build a volume info dict merging volume + linked spatial field."""
        log = logging.getLogger("tsd_client.scene")
        vol = self.get_object(ANARI_VOLUME, volume_index)
        if vol is None:
            return None
        info: dict = {"index": vol.object_index, "name": vol.name}
        params = vol.params_dict()
        meta = vol.metadata_dict()
        log.debug("volume_info[%d]: params keys=%s, meta keys=%s",
                  volume_index, list(params.keys()), list(meta.keys()))
        info.update(params)
        info.update(meta)
        self._resolve_volume_color(vol, info)
        self._resolve_volume_opacity(info)
        field_obj = None
        for param_name in ("value", "field"):
            field_idx = vol.linked_object_index(param_name)
            if field_idx is not None:
                field_obj = self.get_object(ANARI_SPATIAL_FIELD, field_idx)
                if field_obj is not None:
                    break
        if field_obj is None:
            field_obj = self.get_object(ANARI_SPATIAL_FIELD, volume_index)
        if field_obj is not None:
            field_dict: dict = {"name": field_obj.name, "subtype": field_obj.subtype}
            field_dict.update(field_obj.params_dict())
            field_dict.update(field_obj.metadata_dict())
            if field_obj._data_node is not None:
                field_dict.update(_collect_leaf_params(field_obj._data_node))
            info["field"] = field_dict
        return info

    def volume_list(self) -> list[dict]:
        """Return ``[{index, name}, ...]`` for all volumes."""
        return [{"index": v.object_index, "name": v.name} for v in self.volumes]

    def light_info(self, light_index: int) -> dict | None:
        lt = self.get_object(ANARI_LIGHT, light_index)
        if lt is None:
            return None
        info: dict = {"index": lt.object_index, "name": lt.name, "subtype": lt.subtype}
        info.update(lt.params_dict())
        return info

    def lights_list(self) -> list[dict]:
        return [
            self.light_info(lt.object_index)
            for lt in self.lights
            if self.light_info(lt.object_index) is not None
        ]

    # ==================================================================
    # Parameter dispatch (overridable by subclasses)
    # ==================================================================

    def set_object_parameter(
        self,
        object_type: int,
        object_index: int,
        name: str,
        value,
        param_type: str | None = None,
    ):
        """Set a parameter on a scene object.

        Updates the local DataTree and sends to the server (if a client is
        attached).  For fine-grained control use :meth:`object_ref`.

        Parameters
        ----------
        object_type : int
            ANARI type constant.
        object_index : int
            Object pool index.
        name : str
            Parameter name.
        value
            New value.
        param_type : str, optional
            Wire type string; inferred if omitted.
        """
        ref = self.object_ref(object_type, object_index)
        if ref is None:
            return
        ref.set_parameter(name, value, param_type=param_type)

    def set_volume_tf(
        self,
        volume_index: int,
        color_rgba,
        opacity_xy,
        *,
        value_range=None,
        opacity=None,
        unit_distance=None,
    ):
        """Set a volume's transfer function on the server.

        Creates an ``ANARI_ARRAY1D`` of ``ANARI_FLOAT32_VEC4`` on the
        server via ``SERVER_ADD_OBJECT``, then points the volume's
        ``color`` parameter at it via ``SERVER_SET_OBJECT_PARAMETER``.
        Scalar params (``valueRange``, ``opacity``, ``unitDistance``) are
        sent as a separate ``ParameterChange``.

        Parameters
        ----------
        volume_index : int
            Volume pool index.
        color_rgba : list[tuple]
            RGBA tuples sampled from the colour ramp.
        opacity_xy : list[tuple]
            Opacity control points ``(x, y)`` (stored as metadata).
        value_range : tuple, optional
        opacity : float, optional
        unit_distance : float, optional
        """
        if self._client is None or not getattr(self._client, "connected", False):
            return
        from .protocol import MessageType

        n_samples = len(color_rgba)
        if n_samples < 2:
            return

        rgba_floats: list[float] = []
        for rgba in color_rgba:
            r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
            a = float(rgba[3]) if len(rgba) > 3 else 1.0
            rgba_floats.extend((r, g, b, a))
        raw = struct.pack(f"<{len(rgba_floats)}f", *rgba_floats)

        db_node = self._tree.root.child("objectDB")
        array_pool = db_node.child("array") if db_node else None
        base_count = len(array_pool.children) if array_pool is not None else 0
        arr_idx = base_count + self._next_array_offset
        self._next_array_offset += 1

        tree = DataTree()
        tree._size_t_bytes = self._wire_size_t
        tree.root["self"].set_object(ANARI_ARRAY1D, arr_idx)
        tree.root["subtype"].set_string("")
        tree.root["name"].set_string(f"tf_color_{volume_index}")
        tree.root["arrayData"].set_array_data(ANARI_FLOAT32_VEC4, raw, n_samples)
        tree.root["arrayDim"].set_uvec3((n_samples, 0, 0))
        self._client.send(MessageType.SERVER_ADD_OBJECT, tree.to_bytes())

        color_payload = build_parameter_change_payload(
            ANARI_VOLUME,
            volume_index,
            [("color", "object_ref", (ANARI_ARRAY1D, arr_idx))],
            wire_size_t=self._wire_size_t,
        )
        self._client.send(MessageType.SERVER_SET_OBJECT_PARAMETER, color_payload)

        scalar_params: list[tuple[str, str, Any]] = []
        if value_range is not None:
            scalar_params.append(("valueRange", "float32_vec2", tuple(value_range)))
        if opacity is not None:
            scalar_params.append(("opacity", "float32", float(opacity)))
        if unit_distance is not None:
            scalar_params.append(("unitDistance", "float32", float(unit_distance)))
        if scalar_params:
            payload = build_parameter_change_payload(
                ANARI_VOLUME,
                volume_index,
                scalar_params,
                wire_size_t=self._wire_size_t,
            )
            self._client.send(MessageType.SERVER_SET_OBJECT_PARAMETER, payload)

        tf_meta: list[tuple[str, str, Any]] = [
            ("_tfColorStops", "string",
             json.dumps(self._rgba_to_color_stops(list(color_rgba)))),
        ]
        if opacity_xy:
            tf_meta.append((
                "_tfOpacityPoints", "string",
                json.dumps([[float(p[0]), float(p[1])] for p in opacity_xy]),
            ))
        meta_payload = build_parameter_change_payload(
            ANARI_VOLUME,
            volume_index,
            tf_meta,
            wire_size_t=self._wire_size_t,
        )
        self._client.send(MessageType.SERVER_SET_OBJECT_PARAMETER, meta_payload)

        logging.getLogger("tsd_client.scene").info(
            "set_volume_tf: vol=%d, %d samples, array_idx=%d, %d opacity pts",
            volume_index, n_samples, arr_idx, len(opacity_xy) if opacity_xy else 0,
        )

    def _send_object_parameter(
        self,
        object_type: int,
        object_index: int,
        name: str,
        param_type: str,
        value,
    ) -> None:
        """Send ``SERVER_SET_OBJECT_PARAMETER`` (VisRTX ParameterChange) when a client is attached."""
        if self._client is None or not getattr(self._client, "connected", False):
            return
        from .protocol import MessageType

        payload = build_parameter_change_payload(
            object_type,
            object_index,
            [(name, param_type, value)],
            wire_size_t=self._wire_size_t,
        )
        self._client.send(MessageType.SERVER_SET_OBJECT_PARAMETER, payload)

    # ==================================================================
    # Modifications — transforms
    # ==================================================================

    def set_transform(
        self,
        layer_name: str,
        node_name: str,
        *,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> bool:
        info = self.find_transform(layer_name, node_name)
        if info is None or info._data_node is None:
            return False
        self._apply_srt(info, scale, rotation_deg, position)
        return True

    def set_object_parent_transform(
        self,
        object_type: int,
        object_index: int,
        *,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> str | None:
        result = self.find_parent_transform(object_type, object_index)
        if result is None:
            return None
        lname, info = result
        if info._data_node is None:
            return None
        self._apply_srt(info, scale, rotation_deg, position)
        return lname

    # ==================================================================
    # Layer node enable/disable
    # ==================================================================

    def set_node_enabled(self, layer_name: str, node_name: str, enabled: bool) -> bool:
        info = self.find_node(layer_name, node_name)
        if info is None or info._data_node is None:
            return False
        info.enabled = enabled
        enabled_node = info._data_node.child("enabled")
        if enabled_node is not None:
            enabled_node.set_bool(enabled)
        return True

    def set_object_visible(
        self, object_type: int, object_index: int, visible: bool
    ) -> str | None:
        """Toggle the ``enabled`` flag on the layer tree node that references
        the given object and return the layer name (for sending an update).
        Returns ``None`` if the node wasn't found.
        """
        for lname, lroot in self._parsed_layers.items():
            node = self._find_object_node(lroot, object_type, object_index)
            if node is not None and node._data_node is not None:
                node.enabled = visible
                en = node._data_node.child("enabled")
                if en is not None:
                    en.set_bool(visible)
                return lname
        return None

    @staticmethod
    def _find_object_node(
        root: LayerNodeInfo, object_type: int, object_index: int
    ) -> LayerNodeInfo | None:
        if root.object_type == object_type and root.object_index == object_index:
            return root
        for child in root.children:
            found = SceneGraph._find_object_node(child, object_type, object_index)
            if found is not None:
                return found
        return None

    # ==================================================================
    # Layer serialization (SERVER_UPDATE_LAYER payload)
    # ==================================================================

    def build_layer_payload(self, layer_name: str) -> bytes | None:
        layer_node = self._layers.get(layer_name)
        if layer_node is None:
            return None
        tree = DataTree()
        tree.root["n"].set_string(layer_name)
        tree.root.children["l"] = layer_node
        old_name = layer_node.name
        layer_node.name = "l"
        try:
            return tree.to_bytes()
        finally:
            layer_node.name = old_name

    def apply_layer_update(self, payload: bytes) -> str | None:
        """Apply a ``SERVER_UPDATE_LAYER`` payload received from the server."""
        try:
            tree = DataTree.from_bytes(payload)
        except Exception:
            return None
        name_node = tree.root.child("n")
        layer_node = tree.root.child("l")
        if name_node is None or layer_node is None:
            return None
        layer_name = name_node.get_string()
        self._layers[layer_name] = layer_node
        self._parsed_layers[layer_name] = self._parse_layer_node(layer_node)
        return layer_name

    # ==================================================================
    # Printing / debug
    # ==================================================================

    def print_layers(self):
        for lname, lroot in self._parsed_layers.items():
            print(f"Layer: {lname}")
            self._print_layer_node(lroot, indent=1)

    def print_objects(self, object_type: int | None = None):
        for (t, idx), obj in sorted(self._objects.items()):
            if object_type is not None and t != object_type:
                continue
            params = ", ".join(obj.parameters.keys()) if obj.parameters else ""
            print(
                f"  {obj.type_name}@{idx}: {obj.name!r} "
                f"(subtype={obj.subtype!r})"
                + (f" params=[{params}]" if params else "")
            )

    @staticmethod
    def _print_layer_node(info: LayerNodeInfo, indent: int = 0):
        prefix = "    " * indent
        parts = [f"{prefix}{info.name!r}"]
        if info.is_transform:
            srt = info.transform_srt
            if srt:
                parts.append(
                    f"T=({srt[6]:.2f},{srt[7]:.2f},{srt[8]:.2f}) "
                    f"R=({srt[3]:.1f},{srt[4]:.1f},{srt[5]:.1f}) "
                    f"S=({srt[0]:.2f},{srt[1]:.2f},{srt[2]:.2f})"
                )
        if info.object_index is not None:
            parts.append(f"{anari_type_name(info.object_type)}@{info.object_index}")
        if not info.enabled:
            parts.append("[disabled]")
        print(" | ".join(parts))
        for child in info.children:
            SceneGraph._print_layer_node(child, indent + 1)

    # ==================================================================
    # Internals
    # ==================================================================

    def _parse(self):
        self._parse_layers()
        self._parse_object_db()

    def _parse_layers(self):
        layers_node = self._tree.root.child("layers")
        if layers_node is None:
            return
        for lname, ldata in layers_node.children.items():
            self._layers[lname] = ldata
            self._parsed_layers[lname] = self._parse_layer_node(ldata)

    def _parse_object_db(self):
        db_node = self._tree.root.child("objectDB")
        if db_node is None:
            return
        pool_names = [
            "geometry",
            "sampler",
            "material",
            "surface",
            "spatialfield",
            "volume",
            "light",
            "camera",
            "renderer",
            "array",
        ]
        for pool_name in pool_names:
            pool = db_node.child(pool_name)
            if pool is None:
                continue
            for obj_data in pool.children.values():
                info = self._parse_object(obj_data)
                if info is not None:
                    self._objects[(info.object_type, info.object_index)] = info

    @staticmethod
    def _parse_object(data_node: DataNode) -> ObjectInfo | None:
        info = ObjectInfo()
        info._data_node = data_node
        name_node = data_node.child("name")
        if name_node is not None:
            info.name = name_node.get_string()
        subtype_node = data_node.child("subtype")
        if subtype_node is not None:
            info.subtype = subtype_node.get_string()
        self_node = data_node.child("self")
        if self_node is not None and _is_object_type(self_node.dtype):
            info.object_type = self_node.dtype
            info.object_index = self_node.object_index or 0
        else:
            return None
        params_node = data_node.child("parameters")
        if params_node is not None:
            for pname, pdata in params_node.children.items():
                info.parameters[pname] = pdata
        meta_node = data_node.child("metadata")
        if meta_node is not None:
            for mname, mdata in meta_node.children.items():
                info.metadata[mname] = mdata
        return info

    @staticmethod
    def _parse_layer_node(data_node: DataNode) -> LayerNodeInfo:
        info = LayerNodeInfo()
        info._data_node = data_node
        name_node = data_node.child("name")
        if name_node is not None:
            info.name = name_node.get_string()
        value_node = data_node.child("value")
        if value_node is not None and _is_object_type(value_node.dtype):
            info.object_type = value_node.dtype
            info.object_index = value_node.object_index
        srt_node = data_node.child("transformSRT")
        if srt_node is not None and srt_node.dtype == ANARI_FLOAT32_MAT3:
            info.is_transform = True
            info.transform_srt = srt_node.get_mat3()
        enabled_node = data_node.child("enabled")
        if enabled_node is not None:
            info.enabled = enabled_node.get_bool()
        children_node = data_node.child("children")
        if children_node is not None:
            for child_data in children_node.children.values():
                info.children.append(SceneGraph._parse_layer_node(child_data))
        return info

    @staticmethod
    def _apply_srt(
        info: LayerNodeInfo,
        scale: tuple[float, float, float],
        rotation_deg: tuple[float, float, float],
        position: tuple[float, float, float],
    ):
        srt = scale + rotation_deg + position
        info.transform_srt = srt
        srt_node = info._data_node.child("transformSRT")
        if srt_node is not None:
            srt_node.set_mat3(srt)

    @staticmethod
    def _find_node(
        info: LayerNodeInfo, name: str, transforms_only: bool
    ) -> LayerNodeInfo | None:
        if info.name == name:
            if not transforms_only or info.is_transform:
                return info
        for child in info.children:
            result = SceneGraph._find_node(child, name, transforms_only)
            if result is not None:
                return result
        return None

    @staticmethod
    def _find_parent_transform_impl(
        info: LayerNodeInfo, object_type: int, object_index: int
    ) -> LayerNodeInfo | None:
        for child in info.children:
            if child.object_type == object_type and child.object_index == object_index:
                return info if info.is_transform else None
            result = SceneGraph._find_parent_transform_impl(child, object_type, object_index)
            if result is not None:
                return result
        return None

    @staticmethod
    def _collect_objects(
        info: LayerNodeInfo, object_type: int | None, out: list[LayerNodeInfo]
    ):
        if info.is_object:
            if object_type is None or info.object_type == object_type:
                out.append(info)
        for child in info.children:
            SceneGraph._collect_objects(child, object_type, out)
