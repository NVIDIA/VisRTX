## ANARI TSD Device (`anari_tsd`)

`anari_tsd` is an ANARI device implementation that mirrors ANARI object state
into a TSD scene, so ANARI applications can capture their render state for
offline inspection in `tsdViewer`.

### High-Level Concepts

- ANARI objects created on this device are represented as TSD scene objects
  (`Camera`, `Geometry`, `Material`, `Volume`, `Light`, `Renderer`, etc.).
- Instancing (`Group` / `Instance` / `World`) is translated to TSD layers and
  layer nodes.
- ANARI arrays are backed by TSD arrays, including shared-memory array data.
- Each time frame rendering observes new finalized commits, the device saves
  a Scene Archive to `live_capture.tsd`.

### Why Use This Device

- You want to inspect what an ANARI app is actually submitting, independent of
  the app's own UI/debug tooling.
- You want to open captured scene state in `tsdViewer` for object/layer/
  parameter inspection.
- You want a lightweight pass-through path that can still forward rendering to a
  real backend ANARI device.

### Operating Modes

1. Internal-scene pass-through mode (default)

- The device creates its own internal `tsd::scene::Scene`.
- It also creates a surrogate backend ANARI device used for frame rendering.
- Backend library selection comes from `ANARI_TSD_LIBRARY` (default: `helide`).
- Captures are written to `live_capture.tsd` in the process working directory.

2. External-scene capture mode (`scene` device parameter)

- You provide a `tsd::scene::Scene *` via device parameter `"scene"`
  (`ANARI_VOID_POINTER`).
- ANARI state is mirrored into your provided scene.
- Framebuffer channel mapping is not used in this mode; use this when the goal
  is scene capture/inspection.

### Build and Artifacts

- This component is built from `src/anari_tsd/` (added by `src/CMakeLists.txt`).
- Main CMake targets:
  `tsd_device` (core implementation) and `anari_library_tsd` (shared ANARI
  library entrypoint).
- Public C/C++ API header:
  `include/anari_tsd/anariNewTsdDevice.h`.

### Quick Start

Load as a standard ANARI library:

```cpp
anari::Library lib = anari::loadLibrary("tsd", statusFunc);
anari::Device d = anari::newDevice(lib, "default");
```

Or create directly through the helper constructor:

```cpp
#include <anari_tsd/anariNewTsdDevice.h>

anari::Device d = anariNewTsdDevice();
```

Attach to an existing TSD scene for external-scene capture mode:

```cpp
void *scenePtr = &myTsdScene;
anari::setParameter(d, d, "scene", scenePtr);
anari::commitParameters(d, d);
```

### Inspecting Captures in `tsdViewer`

Once `live_capture.tsd` is generated:

```bash
./tsdViewer -tsd live_capture.tsd
```

You can also load the `.tsd` file from the viewer UI.

### Environment Variables

- `ANARI_TSD_LIBRARY`: backend ANARI library used in internal-scene pass-through
  mode (default: `helide`).

### Example in This Repository

- `../../apps/interactive/demos/viskores/tsdDemoViskores.cpp`

This demo uses `anariNewTsdDevice()` and sets the device `"scene"` parameter to
mirror ANARI graph output into a live TSD scene.
