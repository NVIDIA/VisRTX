# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
ANARI type constants and helpers.

Provides the integer type codes from ``anari_enums.h``, human-readable names,
byte sizes, and pack/unpack utilities used by the DataTree codec.
"""

import struct

# ---------------------------------------------------------------------------
# ANARI type constants (matching anari_enums.h integer values)
# ---------------------------------------------------------------------------

ANARI_UNKNOWN = 0
ANARI_STRING = 101
ANARI_BOOL = 103

# Object handle types (500-519) — treated as object indices in DataTree
_ANARI_OBJECT_RANGE = range(500, 520)

ANARI_INT8 = 1000
ANARI_UINT8 = 1004
ANARI_INT16 = 1008
ANARI_UINT16 = 1012
ANARI_INT32 = 1016
ANARI_UINT32 = 1020
ANARI_INT64 = 1024
ANARI_UINT64 = 1028
ANARI_FLOAT32 = 1068
ANARI_FLOAT32_VEC2 = 1069
ANARI_FLOAT32_VEC3 = 1070
ANARI_FLOAT32_VEC4 = 1071
ANARI_FLOAT64 = 1072
ANARI_FLOAT32_BOX1 = 2008
ANARI_FLOAT32_MAT2 = 2012
ANARI_FLOAT32_MAT3 = 2013
ANARI_FLOAT32_MAT4 = 2014

ANARI_ARRAY = 503
ANARI_ARRAY1D = 504
ANARI_ARRAY2D = 505
ANARI_ARRAY3D = 506
ANARI_CAMERA = 507
ANARI_FRAME = 508
ANARI_GEOMETRY = 509
ANARI_GROUP = 510
ANARI_INSTANCE = 511
ANARI_LIGHT = 512
ANARI_MATERIAL = 513
ANARI_RENDERER = 514
ANARI_SURFACE = 515
ANARI_SAMPLER = 516
ANARI_SPATIAL_FIELD = 517
ANARI_VOLUME = 518
ANARI_WORLD = 519

# ---------------------------------------------------------------------------
# Type name table
# ---------------------------------------------------------------------------

_ANARI_TYPE_NAMES: dict[int, str] = {
    ANARI_UNKNOWN: "unknown",
    ANARI_STRING: "string",
    ANARI_BOOL: "bool",
    ANARI_INT32: "int32",
    ANARI_UINT32: "uint32",
    ANARI_INT64: "int64",
    ANARI_UINT64: "uint64",
    ANARI_FLOAT32: "float",
    ANARI_FLOAT32_VEC2: "vec2f",
    ANARI_FLOAT32_VEC3: "vec3f",
    ANARI_FLOAT32_VEC4: "vec4f",
    ANARI_FLOAT64: "double",
    ANARI_FLOAT32_MAT3: "mat3",
    ANARI_FLOAT32_MAT4: "mat4",
    ANARI_ARRAY: "array",
    ANARI_ARRAY1D: "array1d",
    ANARI_ARRAY2D: "array2d",
    ANARI_ARRAY3D: "array3d",
    ANARI_CAMERA: "camera",
    ANARI_GEOMETRY: "geometry",
    ANARI_GROUP: "group",
    ANARI_INSTANCE: "instance",
    ANARI_LIGHT: "light",
    ANARI_MATERIAL: "material",
    ANARI_RENDERER: "renderer",
    ANARI_SURFACE: "surface",
    ANARI_SAMPLER: "sampler",
    ANARI_SPATIAL_FIELD: "spatialField",
    ANARI_VOLUME: "volume",
    ANARI_WORLD: "world",
}


def anari_type_name(t: int) -> str:
    """Human-readable name for an ANARI type code."""
    return _ANARI_TYPE_NAMES.get(t, f"type({t})")


def is_object_type(t: int) -> bool:
    """True if *t* is an ANARI object handle type (500–519)."""
    return t in _ANARI_OBJECT_RANGE


# ---------------------------------------------------------------------------
# Extended type codes (for byte-size table)
# ---------------------------------------------------------------------------

_ANARI_INT8_VEC2, _ANARI_INT8_VEC3, _ANARI_INT8_VEC4 = 1001, 1002, 1003
_ANARI_UINT8_VEC2, _ANARI_UINT8_VEC3, _ANARI_UINT8_VEC4 = 1005, 1006, 1007
_ANARI_INT16_VEC2, _ANARI_INT16_VEC3, _ANARI_INT16_VEC4 = 1009, 1010, 1011
_ANARI_UINT16_VEC2, _ANARI_UINT16_VEC3, _ANARI_UINT16_VEC4 = 1013, 1014, 1015
_ANARI_INT32_VEC2, _ANARI_INT32_VEC3, _ANARI_INT32_VEC4 = 1017, 1018, 1019
_ANARI_UINT32_VEC2, _ANARI_UINT32_VEC3, _ANARI_UINT32_VEC4 = 1021, 1022, 1023
_ANARI_INT64_VEC2, _ANARI_INT64_VEC3, _ANARI_INT64_VEC4 = 1025, 1026, 1027
_ANARI_UINT64_VEC2, _ANARI_UINT64_VEC3, _ANARI_UINT64_VEC4 = 1029, 1030, 1031
_ANARI_FLOAT16, _ANARI_FLOAT16_VEC2, _ANARI_FLOAT16_VEC3, _ANARI_FLOAT16_VEC4 = 1064, 1065, 1066, 1067
_ANARI_FLOAT64_VEC2, _ANARI_FLOAT64_VEC3, _ANARI_FLOAT64_VEC4 = 1073, 1074, 1075
_ANARI_UFIXED8_R_SRGB, _ANARI_UFIXED8_RA_SRGB, _ANARI_UFIXED8_RGB_SRGB = 2000, 2001, 2002
_ANARI_INT32_BOX1, _ANARI_INT32_BOX2, _ANARI_INT32_BOX3, _ANARI_INT32_BOX4 = 2004, 2005, 2006, 2007
_ANARI_FLOAT32_BOX2, _ANARI_FLOAT32_BOX3, _ANARI_FLOAT32_BOX4 = 2009, 2010, 2011
_ANARI_FLOAT32_MAT2x3, _ANARI_FLOAT32_MAT3x4, _ANARI_FLOAT32_QUAT_IJKW = 2015, 2016, 2017
_ANARI_UINT64_REGION1, _ANARI_UINT64_REGION2, _ANARI_UINT64_REGION3, _ANARI_UINT64_REGION4 = 2104, 2105, 2106, 2107
_ANARI_FLOAT64_BOX1, _ANARI_FLOAT64_BOX2, _ANARI_FLOAT64_BOX3, _ANARI_FLOAT64_BOX4 = 2208, 2209, 2210, 2211
_ANARI_FIXED8, _ANARI_UFIXED8, _ANARI_FIXED16, _ANARI_UFIXED16 = 1032, 1036, 1040, 1044
_ANARI_FIXED32, _ANARI_UFIXED32, _ANARI_FIXED64, _ANARI_UFIXED64 = 1048, 1052, 1056, 1060
_ANARI_DATA_TYPE = 100

_TYPE_SIZE: dict[int, int] = {
    ANARI_UNKNOWN: 4,
    ANARI_BOOL: 1,
    ANARI_INT8: 1, _ANARI_INT8_VEC2: 2, _ANARI_INT8_VEC3: 3, _ANARI_INT8_VEC4: 4,
    ANARI_UINT8: 1, _ANARI_UINT8_VEC2: 2, _ANARI_UINT8_VEC3: 3, _ANARI_UINT8_VEC4: 4,
    ANARI_INT16: 2, _ANARI_INT16_VEC2: 4, _ANARI_INT16_VEC3: 6, _ANARI_INT16_VEC4: 8,
    ANARI_UINT16: 2, _ANARI_UINT16_VEC2: 4, _ANARI_UINT16_VEC3: 6, _ANARI_UINT16_VEC4: 8,
    ANARI_INT32: 4, _ANARI_INT32_VEC2: 8, _ANARI_INT32_VEC3: 12, _ANARI_INT32_VEC4: 16,
    ANARI_UINT32: 4, _ANARI_UINT32_VEC2: 8, _ANARI_UINT32_VEC3: 12, _ANARI_UINT32_VEC4: 16,
    ANARI_INT64: 8, _ANARI_INT64_VEC2: 16, _ANARI_INT64_VEC3: 24, _ANARI_INT64_VEC4: 32,
    ANARI_UINT64: 8, _ANARI_UINT64_VEC2: 16, _ANARI_UINT64_VEC3: 24, _ANARI_UINT64_VEC4: 32,
    _ANARI_FLOAT16: 2, _ANARI_FLOAT16_VEC2: 4, _ANARI_FLOAT16_VEC3: 6, _ANARI_FLOAT16_VEC4: 8,
    ANARI_FLOAT32: 4, ANARI_FLOAT32_VEC2: 8, ANARI_FLOAT32_VEC3: 12, ANARI_FLOAT32_VEC4: 16,
    ANARI_FLOAT64: 8, _ANARI_FLOAT64_VEC2: 16, _ANARI_FLOAT64_VEC3: 24, _ANARI_FLOAT64_VEC4: 32,
    ANARI_FLOAT32_BOX1: 8, _ANARI_FLOAT32_BOX2: 16, _ANARI_FLOAT32_BOX3: 24, _ANARI_FLOAT32_BOX4: 32,
    ANARI_FLOAT32_MAT2: 16, ANARI_FLOAT32_MAT3: 36, ANARI_FLOAT32_MAT4: 64,
    _ANARI_FLOAT32_MAT2x3: 24, _ANARI_FLOAT32_MAT3x4: 48, _ANARI_FLOAT32_QUAT_IJKW: 16,
    _ANARI_INT32_BOX1: 8, _ANARI_INT32_BOX2: 16, _ANARI_INT32_BOX3: 24, _ANARI_INT32_BOX4: 32,
    _ANARI_UFIXED8_R_SRGB: 1, _ANARI_UFIXED8_RA_SRGB: 2, _ANARI_UFIXED8_RGB_SRGB: 3,
    2003: 4,  # ANARI_UFIXED8_RGBA_SRGB
    _ANARI_FIXED8: 1, _ANARI_UFIXED8: 1, _ANARI_FIXED16: 2, _ANARI_UFIXED16: 2,
    _ANARI_FIXED32: 4, _ANARI_UFIXED32: 4, _ANARI_FIXED64: 8, _ANARI_UFIXED64: 8,
    _ANARI_DATA_TYPE: 4,
    _ANARI_UINT64_REGION1: 8, _ANARI_UINT64_REGION2: 16, _ANARI_UINT64_REGION3: 24, _ANARI_UINT64_REGION4: 32,
    _ANARI_FLOAT64_BOX1: 16, _ANARI_FLOAT64_BOX2: 32, _ANARI_FLOAT64_BOX3: 48, _ANARI_FLOAT64_BOX4: 64,
}


def element_size(t: int) -> int:
    """Byte size of one element of type *t*."""
    if t in _TYPE_SIZE:
        return _TYPE_SIZE[t]
    if is_object_type(t):
        return 8
    if 1000 <= t <= 1063:
        base = [1, 1, 2, 2, 4, 4, 8, 8][(t - 1000) // 4 % 8]
        comp = (t - 1000) % 4 + 1
        return base * comp
    if 1064 <= t <= 1075:
        base = 2 if t <= 1067 else (4 if t <= 1071 else 8)
        comp = (t - 1064) % 4 + 1
        return base * comp
    if 2000 <= t <= 2003:
        return t - 1999
    if 2004 <= t <= 2007:
        return 8 * (t - 2003)
    if 2008 <= t <= 2011:
        return 8 * (t - 2007)
    if 2104 <= t <= 2107:
        return 8 * (t - 2103)
    if 2208 <= t <= 2211:
        return 16 * (t - 2207)
    if 100 <= t <= 300:
        return 8
    return 4


def unpack_value(buf: bytes, dtype: int):
    """Unpack a single scalar/vector value from *buf* according to *dtype*."""
    if dtype == ANARI_BOOL:
        return bool(buf[0])
    if dtype == ANARI_INT32:
        return struct.unpack_from("<i", buf)[0]
    if dtype == ANARI_UINT32:
        return struct.unpack_from("<I", buf)[0]
    if dtype == ANARI_INT64:
        return struct.unpack_from("<q", buf)[0]
    if dtype == ANARI_UINT64:
        return struct.unpack_from("<Q", buf)[0]
    if dtype == ANARI_FLOAT32:
        return struct.unpack_from("<f", buf)[0]
    if dtype == ANARI_FLOAT64:
        return struct.unpack_from("<d", buf)[0]
    sz = element_size(dtype)
    n = sz // 4
    if n >= 2 and sz == n * 4:
        return tuple(struct.unpack_from(f"<{n}f", buf))
    return buf[:sz]


def pack_value(val, dtype: int) -> bytes:
    """Pack a single scalar/vector *val* into bytes according to *dtype*."""
    if isinstance(val, (bytes, bytearray)):
        sz = element_size(dtype)
        return bytes(val[:sz])
    if dtype == ANARI_BOOL:
        return struct.pack("b", 1 if val else 0)
    if dtype == ANARI_INT32:
        return struct.pack("<i", int(val))
    if dtype == ANARI_UINT32:
        return struct.pack("<I", int(val))
    if dtype == ANARI_INT64:
        return struct.pack("<q", int(val))
    if dtype == ANARI_UINT64:
        return struct.pack("<Q", int(val))
    if dtype == ANARI_FLOAT32:
        return struct.pack("<f", float(val))
    if dtype == ANARI_FLOAT64:
        return struct.pack("<d", float(val))
    sz = element_size(dtype)
    n = sz // 4
    if dtype in (_ANARI_UINT32_VEC2, _ANARI_UINT32_VEC3, _ANARI_UINT32_VEC4):
        return struct.pack(f"<{n}I", *(int(x) for x in val))
    if dtype in (_ANARI_INT32_VEC2, _ANARI_INT32_VEC3, _ANARI_INT32_VEC4):
        return struct.pack(f"<{n}i", *(int(x) for x in val))
    if n >= 2 and sz == n * 4:
        return struct.pack(f"<{n}f", *val)
    return b"\x00" * sz
