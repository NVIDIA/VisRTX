# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Python codec for the TSD DataTree binary format.

Provides :class:`DataNode` and :class:`DataTree` that mirror the C++ types in
``tsd/core/DataTree.hpp``.  The binary (de)serialization is byte-compatible
with the C++ ``DataTree::save`` / ``DataTree::load`` methods, so payloads can
be exchanged over the TSD network protocol (``StructuredMessage``).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

from .anari_types import (
    ANARI_UNKNOWN, ANARI_STRING, ANARI_BOOL,
    ANARI_INT32, ANARI_UINT32, ANARI_INT64, ANARI_UINT64,
    ANARI_FLOAT32, ANARI_FLOAT32_VEC2, ANARI_FLOAT32_VEC3, ANARI_FLOAT32_VEC4,
    ANARI_FLOAT64, ANARI_FLOAT32_MAT3, ANARI_FLOAT32_MAT4,
    is_object_type as _is_object_type,
    element_size as _element_size,
    unpack_value as _unpack_value,
    pack_value as _pack_value,
)
from .utils import split_null_separated as _split_null_separated


# ---------------------------------------------------------------------------
# DataNode
# ---------------------------------------------------------------------------

_APPEND_COUNTER = 0


def _next_synthetic_name() -> str:
    global _APPEND_COUNTER
    name = f"<{_APPEND_COUNTER}>"
    _APPEND_COUNTER += 1
    return name


@dataclass
class DataNode:
    """A node in the DataTree (mirrors ``tsd::core::DataNode``).

    Children are stored in an ordered dict keyed by name.  Unnamed children
    get synthetic names ``<0>``, ``<1>``, ... matching the C++ behaviour.
    """

    name: str = ""
    children: dict[str, DataNode] = field(default_factory=dict)

    dtype: int = ANARI_UNKNOWN
    value: Any = None
    is_array: bool = False
    array_count: int = 0
    object_index: int | None = None

    # -- Children helpers ----------------------------------------------------

    def child(self, name: str) -> DataNode | None:
        return self.children.get(name)

    def __getitem__(self, name: str) -> DataNode:
        if name not in self.children:
            self.children[name] = DataNode(name=name)
        return self.children[name]

    def append(self, name: str = "") -> DataNode:
        if not name:
            name = _next_synthetic_name()
        if name in self.children:
            return self.children[name]
        node = DataNode(name=name)
        self.children[name] = node
        return node

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def foreach_child(self, fn):
        for child in list(self.children.values()):
            fn(child)

    # -- Value setters -------------------------------------------------------

    def set_string(self, s: str):
        self.dtype = ANARI_STRING
        self.value = s
        self.is_array = False

    def set_float(self, v: float):
        self.dtype = ANARI_FLOAT32
        self.value = v
        self.is_array = False

    def set_int(self, v: int):
        self.dtype = ANARI_INT32
        self.value = v
        self.is_array = False

    def set_uint32(self, v: int):
        self.dtype = ANARI_UINT32
        self.value = v
        self.is_array = False

    def set_bool(self, v: bool):
        self.dtype = ANARI_BOOL
        self.value = v
        self.is_array = False

    def set_vec2(self, v: tuple[float, float]):
        self.dtype = ANARI_FLOAT32_VEC2
        self.value = tuple(v)
        self.is_array = False

    def set_vec3(self, v: tuple[float, float, float]):
        self.dtype = ANARI_FLOAT32_VEC3
        self.value = tuple(v)
        self.is_array = False

    def set_vec4(self, v: tuple[float, float, float, float]):
        self.dtype = ANARI_FLOAT32_VEC4
        self.value = tuple(v)
        self.is_array = False

    def set_mat3(self, v: tuple):
        self.dtype = ANARI_FLOAT32_MAT3
        self.value = tuple(v)
        self.is_array = False

    def set_mat4(self, v: tuple):
        self.dtype = ANARI_FLOAT32_MAT4
        self.value = tuple(v)
        self.is_array = False

    def set_object(self, anari_type: int, index: int):
        self.dtype = anari_type
        self.object_index = index
        self.value = None
        self.is_array = False

    def set_uvec3(self, v: tuple[int, int, int]):
        from .anari_types import _ANARI_UINT32_VEC3
        self.dtype = _ANARI_UINT32_VEC3
        self.value = tuple(v)
        self.is_array = False

    def set_array_data(self, element_type: int, raw: bytes, count: int):
        """Store raw array data (already packed to element_type layout)."""
        self.dtype = element_type
        self.value = raw
        self.is_array = True
        self.array_count = count

    # -- Value getters -------------------------------------------------------

    def get_string(self) -> str:
        return str(self.value) if self.value is not None else ""

    def get_float(self) -> float:
        return float(self.value) if self.value is not None else 0.0

    def get_int(self) -> int:
        return int(self.value) if self.value is not None else 0

    def get_bool(self) -> bool:
        return bool(self.value) if self.value is not None else False

    def get_vec2(self) -> tuple[float, float]:
        if isinstance(self.value, tuple) and len(self.value) >= 2:
            return (float(self.value[0]), float(self.value[1]))
        return (0.0, 0.0)

    def get_vec3(self) -> tuple[float, float, float]:
        if isinstance(self.value, tuple) and len(self.value) >= 3:
            return (float(self.value[0]), float(self.value[1]), float(self.value[2]))
        return (0.0, 0.0, 0.0)

    def get_vec4(self) -> tuple[float, float, float, float]:
        if isinstance(self.value, tuple) and len(self.value) >= 4:
            return (float(self.value[0]), float(self.value[1]),
                    float(self.value[2]), float(self.value[3]))
        return (0.0, 0.0, 0.0, 0.0)

    def get_mat3(self) -> tuple:
        if isinstance(self.value, tuple) and len(self.value) == 9:
            return self.value
        return (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # -- Display -------------------------------------------------------------

    def __repr__(self):
        if self.is_array:
            return f"DataNode({self.name!r}, array[{self.array_count}])"
        if self.object_index is not None:
            return f"DataNode({self.name!r}, obj@{self.object_index})"
        if self.value is not None:
            return f"DataNode({self.name!r}, {self.value!r})"
        kids = list(self.children.keys())
        return f"DataNode({self.name!r}, children={kids})"

    def print_tree(self, indent: int = 0):
        prefix = "    " * indent
        if self.is_leaf:
            vstr = ""
            if self.is_array:
                vstr = f" array[{self.array_count}]"
            elif self.object_index is not None:
                vstr = f" obj@{self.object_index} (type={self.dtype})"
            elif self.dtype == ANARI_STRING:
                vstr = f' "{self.value}"'
            elif self.value is not None:
                vstr = f" {self.value}"
            print(f"{prefix}{self.name}:{vstr}")
        else:
            print(f"{prefix}{self.name}:")
            for child in self.children.values():
                child.print_tree(indent + 1)


# ---------------------------------------------------------------------------
# DataTree
# ---------------------------------------------------------------------------


class DataTree:
    """Binary-compatible Python mirror of ``tsd::core::DataTree``."""

    def __init__(self):
        self.root = DataNode(name="root")
        self._size_t_bytes: int = 8

    # -- Deserialization  (matches DataTree::loadImpl) -----------------------

    @classmethod
    def from_bytes(cls, data: bytes | bytearray, size_t_bytes: int = 0) -> DataTree:
        """Deserialize a DataTree from binary data.

        Parameters
        ----------
        data : bytes
            Binary payload (from ``DataTree::save`` or a ``StructuredMessage``).
        size_t_bytes : int
            Width of ``size_t`` in the binary data (4 or 8).
            If 0 (default), auto-detected from the first 8 bytes.
        """
        tree = cls()
        if len(data) < 8:
            raise ValueError(
                f"DataTree payload too small ({len(data)} bytes)"
            )

        if size_t_bytes == 0:
            size_t_bytes = cls._detect_size_t(data)

        tree._size_t_bytes = size_t_bytes
        off = 0
        st_fmt = "<I" if size_t_bytes == 4 else "<Q"

        def _size_t() -> int:
            nonlocal off
            (v,) = struct.unpack_from(st_fmt, data, off)
            off += size_t_bytes
            return v

        def _i32() -> int:
            nonlocal off
            (v,) = struct.unpack_from("<i", data, off)
            off += 4
            return v

        def _u8() -> int:
            nonlocal off
            v = data[off]
            off += 1
            return v

        def _raw(n: int) -> bytes:
            nonlocal off
            if off + n > len(data):
                raise ValueError(
                    f"DataTree: read overrun at offset {off}, "
                    f"want {n} bytes, have {len(data) - off}"
                )
            b = data[off: off + n]
            off += n
            return b

        num_leaves = _size_t()

        for leaf_idx in range(num_leaves):
            name_len = _size_t()
            if name_len > len(data):
                raise ValueError(
                    f"DataTree: leaf {leaf_idx}: name_len={name_len} "
                    f"exceeds buffer ({len(data)} bytes) at offset {off}"
                )
            name = _raw(name_len).decode("utf-8", errors="replace")

            path_len = _size_t()
            if path_len > len(data):
                raise ValueError(
                    f"DataTree: leaf {leaf_idx} ({name!r}): path_len={path_len} "
                    f"exceeds buffer ({len(data)} bytes) at offset {off}"
                )
            path_bytes = _raw(path_len)

            is_array = _u8()
            dtype = _i32()

            path_parts = _split_null_separated(path_bytes)
            parent = tree.root
            for part in path_parts:
                if part:
                    parent = parent[part]

            node = parent[name]
            node.dtype = dtype
            node.is_array = bool(is_array)

            if is_array:
                count = _size_t()
                elem_sz = _element_size(dtype)
                raw = _raw(count * elem_sz)
                node.array_count = count
                node.value = raw
            else:
                if _is_object_type(dtype):
                    node.object_index = _size_t()
                elif dtype == ANARI_STRING:
                    str_len = _size_t()
                    node.value = _raw(str_len).decode("utf-8", errors="replace")
                elif dtype != ANARI_UNKNOWN:
                    sz = _element_size(dtype)
                    if sz > 0:
                        raw = _raw(sz)
                        node.value = _unpack_value(raw, dtype)

        return tree

    @staticmethod
    def _detect_size_t(data: bytes) -> int:
        """Auto-detect whether size_t is 4 or 8 bytes."""
        n8 = struct.unpack_from("<Q", data, 0)[0] if len(data) >= 8 else None
        n4 = struct.unpack_from("<I", data, 0)[0] if len(data) >= 4 else None

        if n8 is not None and 0 < n8 <= len(data) // 10:
            return 8

        if n4 is not None and 0 < n4 <= len(data) // 6:
            return 4

        hexdump = data[:64].hex(" ")
        raise ValueError(
            f"DataTree: cannot detect size_t width.\n"
            f"  buffer size: {len(data)} bytes\n"
            f"  as uint64 LE: {n8}\n"
            f"  as uint32 LE: {n4}\n"
            f"  first 64 bytes: {hexdump}"
        )

    # -- Serialization  (matches DataTree::saveImpl + writeDataNode) ---------

    def to_bytes(self) -> bytes:
        leaves: list[tuple[str, str, DataNode]] = []
        self._collect_leaves(self.root, "", leaves, level=0)

        st = self._size_t_bytes
        st_fmt = "<I" if st == 4 else "<Q"

        parts: list[bytes] = []
        parts.append(struct.pack(st_fmt, len(leaves)))

        for name, path, node in leaves:
            name_b = name.encode("utf-8")
            path_b = path.encode("utf-8") if path else b""

            parts.append(struct.pack(st_fmt, len(name_b)))
            parts.append(name_b)
            parts.append(struct.pack(st_fmt, len(path_b)))
            parts.append(path_b)
            parts.append(struct.pack("B", 1 if node.is_array else 0))

            if node.is_array:
                elem_sz = _element_size(node.dtype)
                parts.append(struct.pack("<i", node.dtype))
                parts.append(struct.pack(st_fmt, node.array_count))
                raw = node.value if isinstance(node.value, (bytes, bytearray)) else b""
                parts.append(raw[: node.array_count * elem_sz])
            else:
                parts.append(struct.pack("<i", node.dtype))
                if _is_object_type(node.dtype):
                    idx = node.object_index if node.object_index is not None else 0
                    parts.append(struct.pack(st_fmt, idx))
                elif node.dtype == ANARI_STRING:
                    s = (node.value or "").encode("utf-8")
                    parts.append(struct.pack(st_fmt, len(s)))
                    parts.append(s)
                elif node.dtype != ANARI_UNKNOWN:
                    parts.append(_pack_value(node.value, node.dtype))

        return b"".join(parts)

    # -- Internals -----------------------------------------------------------

    @staticmethod
    def _collect_leaves(node: DataNode, path: str, out: list, level: int):
        if level == 0:
            for child in node.children.values():
                DataTree._collect_leaves(child, "", out, level + 1)
            return
        if node.is_leaf:
            out.append((node.name, path, node))
        else:
            child_path = path + node.name + "\x00" if path else node.name + "\x00"
            for child in node.children.values():
                DataTree._collect_leaves(child, child_path, out, level + 1)

    def print(self):
        self.root.print_tree()

    def __repr__(self):
        return f"DataTree(root={self.root!r})"
