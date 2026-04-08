# Copyright 2025-2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

"""
tsd_client — Python client for TSD (VisRTX) render servers.

Provides the base protocol, connection management, DataTree codec, scene
graph abstraction, interactive viewer widget, and reusable panels for
TSD-based applications.
"""

from .protocol import MessageType, HEADER_FORMAT, HEADER_SIZE, FRAME_CONFIG_FORMAT
from .connection import TSDConnection
from .client import TSDClient
from .datatree import DataNode, DataTree
from .scene import (
    SceneGraph,
    ObjectRef,
    TransformRef,
    ObjectInfo,
    LayerNodeInfo,
    build_parameter_change_payload,
    is_valid_object_pool_index,
)
from .anari_types import (
    ANARI_UNKNOWN, ANARI_STRING, ANARI_BOOL,
    ANARI_INT32, ANARI_UINT32, ANARI_INT64, ANARI_UINT64,
    ANARI_FLOAT32, ANARI_FLOAT32_VEC2, ANARI_FLOAT32_VEC3, ANARI_FLOAT32_VEC4,
    ANARI_FLOAT64, ANARI_FLOAT32_MAT3, ANARI_FLOAT32_MAT4,
    ANARI_ARRAY, ANARI_ARRAY1D, ANARI_ARRAY2D, ANARI_ARRAY3D,
    ANARI_CAMERA, ANARI_FRAME, ANARI_GEOMETRY, ANARI_GROUP, ANARI_INSTANCE,
    ANARI_LIGHT, ANARI_MATERIAL, ANARI_RENDERER,
    ANARI_SURFACE, ANARI_SAMPLER, ANARI_SPATIAL_FIELD,
    ANARI_VOLUME, ANARI_WORLD,
    anari_type_name,
    is_object_type,
    element_size,
    unpack_value,
    pack_value,
)
from .utils import (
    lerp,
    smoothstep,
    clamp,
    normalize3,
    slerp3,
    slerp3_exact,
    resolve_client,
    guess_type,
    parse_1dt,
    parse_tf_json,
    split_null_separated,
    datanode_to_attr,
)
from .viewer import (
    TSDViewer,
    OrbitTSDViewer,
    CameraPose,
    orbit_camera_pose,
    interpolate_camera_path,
    animate_camera_path,
)
from .session import TSDSession, ShowcaseViewer

__version__ = "0.2.0"

__all__ = [
    # Protocol
    "MessageType",
    "HEADER_FORMAT",
    "HEADER_SIZE",
    "FRAME_CONFIG_FORMAT",
    # Connection & Client
    "TSDConnection",
    "TSDClient",
    # DataTree
    "DataNode",
    "DataTree",
    # Scene graph
    "SceneGraph",
    "ObjectRef",
    "TransformRef",
    "ObjectInfo",
    "LayerNodeInfo",
    "build_parameter_change_payload",
    "is_valid_object_pool_index",
    # ANARI types
    "ANARI_UNKNOWN", "ANARI_STRING", "ANARI_BOOL",
    "ANARI_INT32", "ANARI_UINT32", "ANARI_INT64", "ANARI_UINT64",
    "ANARI_FLOAT32", "ANARI_FLOAT32_VEC2", "ANARI_FLOAT32_VEC3", "ANARI_FLOAT32_VEC4",
    "ANARI_FLOAT64", "ANARI_FLOAT32_MAT3", "ANARI_FLOAT32_MAT4",
    "ANARI_ARRAY", "ANARI_ARRAY1D", "ANARI_ARRAY2D", "ANARI_ARRAY3D",
    "ANARI_CAMERA", "ANARI_FRAME", "ANARI_GEOMETRY", "ANARI_GROUP", "ANARI_INSTANCE",
    "ANARI_LIGHT", "ANARI_MATERIAL", "ANARI_RENDERER",
    "ANARI_SURFACE", "ANARI_SAMPLER", "ANARI_SPATIAL_FIELD",
    "ANARI_VOLUME", "ANARI_WORLD",
    "anari_type_name", "is_object_type", "element_size",
    "unpack_value", "pack_value",
    # Utilities
    "lerp", "smoothstep", "clamp",
    "normalize3", "slerp3", "slerp3_exact",
    "resolve_client", "guess_type",
    "parse_1dt", "parse_tf_json",
    "split_null_separated",
    "datanode_to_attr",
    # Viewer
    "TSDViewer",
    "OrbitTSDViewer",
    "orbit_camera_pose",
    "CameraPose",
    "interpolate_camera_path",
    "animate_camera_path",
    # Session
    "TSDSession",
    "ShowcaseViewer",
]
