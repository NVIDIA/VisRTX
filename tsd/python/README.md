<!--
Copyright 2025-2026 NVIDIA Corporation
SPDX-License-Identifier: Apache-2.0
-->

# tsd_client

Python client library for [TSD](https://github.com/NVIDIA/VisRTX) (VisRTX) render servers over TCP.

Provides the base protocol, connection management, DataTree binary codec,
scene graph abstraction, interactive Jupyter viewer widget, and reusable
panels for TSD-based applications.

## Installation

```bash
pip install -e .
```

## Usage

### Base client (any TSD server)

```python
from tsd_client import TSDClient

client = TSDClient("127.0.0.1", port=12345)
client.start_rendering()
client.send_frame_config(1920, 1080)

# Fetch and inspect the scene
sg = client.scene_graph
sg.print_layers()
sg.print_objects()

# Modify a transform and push
xfm = sg.transform_for_object(sg.ANARI_VOLUME, 0)
if xfm:
    xfm.position = (0, 0, 0)
    xfm.scale = (100, 100, 100)
    xfm.commit(client)

client.disconnect()
```

### Extending for a specific application

```python
from enum import IntEnum
from tsd_client import TSDClient, SceneGraph

class MyAppMessageType(IntEnum):
    SET_ANIMATION_TIME = 100
    REQUEST_STATUS = 101
    STATUS_INFO = 102

class MyAppClient(TSDClient):
    def __init__(self, host="127.0.0.1", port=12345, **kwargs):
        super().__init__(host, port, **kwargs)

    def set_animation_time(self, t: float):
        import struct
        self.send(MyAppMessageType.SET_ANIMATION_TIME, struct.pack("<f", t))
```

### DataTree codec

```python
from tsd_client import DataTree

# Deserialize a DataTree from binary (e.g. from a StructuredMessage payload)
tree = DataTree.from_bytes(raw_bytes)
tree.print()

# Build and serialize
tree = DataTree()
tree.root["material"]["roughness"].set_float(0.4)
data = tree.to_bytes()
```

## Architecture

```
tsd_client/
├── __init__.py       # Public API surface
├── protocol.py       # MessageType enum, wire format constants
├── connection.py     # TCP socket, 8-byte framed messaging, recv thread
├── anari_types.py    # ANARI type IDs, sizes, pack/unpack
├── datatree.py       # DataTree/DataNode binary codec (mirrors tsd::core::DataTree)
├── scene.py          # SceneGraph, ObjectRef, TransformRef, LayerNodeInfo
├── client.py         # TSDClient base class with scene graph management
├── viewer.py         # AnyWidget-based Jupyter viewer (orbit, animation, detach)
├── session.py        # TSDSession convenience wrapper (client + viewer + panels)
├── utils.py          # Shared math, TF parsers, Jupyter helpers
└── panels/
    ├── datatree_panel.py          # Hierarchical scene tree inspector
    ├── transfer_function_panel.py # Interactive colour-map & opacity editor
    ├── clip_planes_panel.py       # Volume clip plane controls
    └── light_panel.py             # Light parameter editor
```

## Protocol

The wire format matches the C++ `tsd::network::NetworkChannel`:

- **Header (8 bytes):** `uint8 type | 3 bytes padding | uint32 payload_length` (little-endian)
- **Payload:** variable-length raw bytes

Message type values match `tsd::network::MessageType` in `RenderSession.hpp`. Application-specific extensions use values >= 100.

## License

Apache-2.0
