# AGENTS.md

This file provides guidance for coding agents working in the `tsd/` subtree.

## About TSD

TSD ("Testing Scene Description") is an **experimental** C++17 scene graph and testing framework for ANARI devices. It has no API stability guarantees. The parent repository is VisRTX; see `../AGENTS.md` for the full project context.

See [STYLEGUIDE.md](STYLEGUIDE.md) for TSD-specific and project-wide C++ coding conventions.

## Domain Docs

Each area keeps a `CONTEXT.md` glossary. Use these terms in code and docs:

- [src/tsd/io/CONTEXT.md](src/tsd/io/CONTEXT.md) — Archives, serialization verbs, import/export vocabulary
- [src/tsd/app/CONTEXT.md](src/tsd/app/CONTEXT.md) — Application Dumps and app-level state
- [src/tsd/rendering/CONTEXT.md](src/tsd/rendering/CONTEXT.md) — Image Pipeline / Image Pass / render-index vocabulary
- [apps/interactive/scivisStudio/CONTEXT.md](apps/interactive/scivisStudio/CONTEXT.md) — SciVis Studio Projects and shots

## Build

TSD can be built standalone (without the VisRTX devices) or as part of VisRTX.

**Standalone:**

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel
```

Requires ANARI-SDK 0.15.0+ (`find_package(anari)` must succeed).

**Within VisRTX** (from the repo root):

```bash
cmake -DVISRTX_BUILD_TSD=ON -DTSD_BUILD_APPS=ON ...
```

Key optional CMake flags: `TSD_USE_LUA`, `TSD_USE_ASSIMP`, `TSD_USE_HDF5`, `TSD_USE_MPI`, `TSD_USE_NETWORKING`, `TSD_USE_VTK`, `TSD_USE_SILO`, `TSD_USE_USD`.

## Tests

Test sources: `tests/` (one `test_*.cpp` per suite, e.g. `ctest -C Release -R test_Forest`). Run via ctest — see the repo-root [AGENTS.md](../AGENTS.md).

## Architecture

### Library Dependency Layers

```text
tsd_core  ->  tsd_scene  ->  tsd_io  ->  tsd_rendering  ->  tsd_app
                                  \                  /
                               (optional: tsd_ui_imgui, tsd_mpi, tsd_network, tsd_lua)
```

- **`tsd_core`** (`src/tsd/core/`): `Any`, `Token`, `ObjectPool`, `FlatMap`, `Forest`, `DataTree`/`DataStream` (serialization), `TaskQueue`, logging. No scene concepts.
- **`tsd_scene`** (`src/tsd/scene/`): `Scene`, `Object`, `Parameter`, `Layer`/`Forest` (instancing), `Animation`, `UpdateDelegate`. Mirrors ANARI's object hierarchy.
- **`tsd_io`** (`src/tsd/io/`): 30+ file format importers (OBJ, GLTF, PLY, USD, VTK, ASSIMP, etc.), volume importers (RAW, NanoVDB, VTI), procedural generators, and TSD scene serialization.
- **`tsd_rendering`** (`src/tsd/rendering/`): `RenderIndex` (TSD-to-ANARI sync), `ImagePipeline` (composable render passes), camera manipulators.
- **`tsd_app`** (`src/tsd/app/`): `ANARIDeviceManager`, `Context` (bundles scene + render index + pipeline), CLI parsing, `renderAnimationSequence`.
- **`anari_tsd`** (`src/anari_tsd/`): ANARI device implementation that mirrors ANARI state into a TSD scene; writes `live_capture.tsd` on each committed frame.

### Key Design Patterns

**Object model**: `Object` holds typed `Parameter` values. Parameters store `tsd::core::Any` (ANARI-typed). Use-count tracking enables garbage collection. `ObjectPool<T>` manages object lifetime with stable handles.

**Scene graph / layering**: `Scene` owns object databases. A `Layer` is a `Forest` of transform/object nodes enabling instancing. Multiple layers with active/inactive control compose the final scene.

**UpdateDelegate flow**: Mutations to `Scene` fire `BaseUpdateDelegate` callbacks. `MultiUpdateDelegate` fans out to `RenderIndex`, network sync, and UI. Subclass `BaseUpdateDelegate` to intercept changes.

**RenderIndex** (`src/tsd/rendering/index/`): Translates TSD scene state to live ANARI handles. `RenderIndexAllLayers` and `RenderIndexFlatRegistry` are the two strategies. Populated via `populate()`, updated incrementally via delegate callbacks.

**ImagePipeline** (`src/tsd/rendering/pipeline/`): Chain of `ImagePass` objects. The standard chain is `AnariSceneRenderPass` -> `MultiDeviceSceneRenderPass` -> `PickPass` -> `VisualizeAOVPass`.

**anari_tsd device modes**:

1. *Internal scene* (default): creates its own `Scene`, renders via a backend device (controlled by `ANARI_TSD_LIBRARY`, default `helide`), writes `live_capture.tsd`.
2. *External scene*: caller provides a `tsd::scene::Scene*` via the `"scene"` device parameter; ANARI state is mirrored into the caller's scene.

### Tutorial Apps

`apps/tutorial/` contains single-file examples covering specific concepts (render, pipeline, multi-render, forest, DataTree, load/save, USD export). These are the best starting point for understanding how the libraries compose.

### Lua Scripting (`src/tsd/scripting/`, enabled with `TSD_USE_LUA=ON`)

`tsdLua` is a standalone interpreter; `tsdViewer` embeds a Lua terminal. Scripts have a pre-bound `scene` variable. The `scripts/init.lua` populates viewer Actions menus. See `src/tsd/scripting/README.md` for the full API and `scripts/examples/` for worked examples.

Lua module search paths (lowest to highest priority):

1. `<source>/tsd/scripts/` (dev builds)
2. `<install>/share/tsd/scripts/`
3. `~/.config/tsd/scripts/`
4. `TSD_LUA_PACKAGE_PATHS` (`:` separated)

### Environment Variables

| Variable | Purpose |
|---|---|
| `TSD_ANARI_LIBRARIES` | Comma-separated ANARI libraries shown in `tsdViewer` device selector |
| `ANARI_TSD_LIBRARY` | Backend library used by `anari_tsd` device (default: `helide`) |
| `TSD_LUA_PACKAGE_PATHS` | Additional Lua module search paths |
