# AGENTS.md

This file provides guidance for coding agents working in the `devices/rtx/`
subtree.

## What This Is

The `devices/rtx/` subdirectory contains the OptiX-based GPU ray-tracing ANARI device (`libanari_library_visrtx`). See the repo-root `AGENTS.md` for project-wide context.

## Domain Docs

- [`CONTEXT-MAP.md`](./CONTEXT-MAP.md) — bounded contexts and how they relate; each context has its own `CONTEXT.md` glossary ([Frontend](./device/CONTEXT.md), [Render Pipeline](./device/renderer/CONTEXT.md), [World](./device/world/CONTEXT.md), [MDL](./libmdl/CONTEXT.md)). Use these terms in code and docs.
- [`docs/adr/`](./docs/adr/) — architecture decision records (callable-based shading, pipeline-per-renderer, embedded PTX, split surface/volume TLAS). Read before "fixing" any of these designs.

## Build

```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install \
      -DCMAKE_PREFIX_PATH="/path/to/anari-sdk;/path/to/optix" \
      /path/to/visrtx
cmake --build . --parallel
cmake --install .
```

Key device-specific CMake options:

- `VISRTX_ENABLE_MDL_SUPPORT`: NVIDIA MDL materials (requires MDL SDK)
- `VISRTX_ENABLE_NEURAL`: Neural Graphics Primitives (requires OptiX 9.0+)
- `VISRTX_ENABLE_NVTX`: NVTX profiling markers
- `OPTIX_FETCH_VERSION`: Pin OptiX version: `7.7`, `8.0`, `8.1`, or `9.0`

## Tests

```bash
ctest -C Release --output-on-failure
ctest -C Release -R TestSpheres --output-on-failure
```

Test sources: `apps/tests/api/`

## Architecture

### Core Pipeline

**VisRTXDevice** creates all ANARI objects via factory methods and owns the `DeviceGlobalState`.

**DeviceGlobalState** (in `device/optix_visrtx.h`) holds:

- CUDA context/stream
- OptiX device context
- All precompiled PTX modules (one per renderer type, one per material/sampler type)
- `DeviceObjectArray` registries for materials, geometries, samplers, spatial fields

**Frame::renderFrame()** orchestrates each frame:

1. Flush pending `commitParameters()` changes
2. Flush deferred GPU array uploads
3. Rebuild world BVH if needed (`World::rebuildWorld()`)
4. Populate `FrameGPUData` with all GPU pointers/handles
5. `optixLaunch()` runs ray generation, traversal, hit/miss shaders
6. Accumulate, optionally denoise, convert to output format

### GPU Object Registry Pattern

Every ANARI object that needs GPU representation inherits from `RegisteredObject<GPUDataType>` (template). On `commitParameters()`:

- CPU state is parsed and packed into a `GPUDataType` struct
- `upload()` copies it into a slot in the corresponding `DeviceObjectArray`
- The slot index becomes the object's `registryIndex`, used in kernels to look up the object

Arrays (`Array1D`, `Array2D`, `Array3D`) use `HostDeviceArray` or deferred uploads via `UploadableArray` to avoid blocking the CPU during large transfers.

### Shader Binding Table (SBT) and Callables

Each renderer has its own OptiX pipeline. Material and spatial-field shaders are implemented as **OptiX callable programs** so the renderer pipeline does not need to be recompiled when materials change.

SBT callable slots per material type (8 slots each):

- `Initialize`, `EvaluateNextRay`, `EvaluateTint`, `EvaluateOpacity`, `EvaluateEmission`, `EvaluateTransmission`, `EvaluateNormal`, `Shade`

`MaterialGPUData::callableBaseIndex` holds the offset into the callable table. Kernels dispatch shading via `optixDirectCall(callableBaseIndex + SHADE_FN, ...)`.

Spatial field samplers also use callables (`Init`, `Sample`).

### Material Parameter Indirection

Material parameters (color, roughness, etc.) support three source modes encoded in each parameter slot:

- `VALUE`: inline constant (vec4)
- `ATTRIBUTE`: per-vertex geometry attribute (by index 0-3 or `color`)
- `SAMPLER`: texture sampler reference (index into sampler registry)

This is defined in `device/gpu/gpu_objects.h` (`MaterialParameter` struct).

### World / BVH Rebuild

`World::rebuildWorld()` builds:

- **BLAS** (bottom-level AS): one per geometry primitive type, built from its vertex/index arrays
- **TLAS** (top-level AS): one for surfaces, one for volumes; instances reference BLAS handles with their transforms

After rebuild, `OptixTraversableHandle`s are stored in `WorldGPUData` and passed to kernels via `FrameGPUData`.

### Key Files

| File | Purpose |
|------|---------|
| `device/VisRTXDevice.h/.cpp` | Main device class; factory for all ANARI objects |
| `device/optix_visrtx.h` | `DeviceGlobalState` definition; OptiX module/pipeline declarations |
| `device/gpu/gpu_objects.h` | All GPU-side data structs (materials, geometry, lights, samplers, ...) |
| `device/gpu/sbt.h` | SBT entry-point enum |
| `device/gpu/evalShading.h` | Shading evaluation entry points called from hit shaders |
| `device/frame/Frame.cu` | Frame render loop, accumulation, denoising, format conversion |
| `device/renderer/Renderer.h` | Base renderer: pipeline init, SBT construction |
| `device/world/World.h/.cpp` | BVH (TLAS/BLAS) management |
| `device/utility/DeviceObjectArray.h` | Thread-safe GPU slot allocator / registry |

### Renderer Types

| Subtype | Characteristics |
|---------|----------------|
| `fast` | Ambient occlusion, one direct-light sample; real-time |
| `quality` | Full Monte Carlo path tracing, configurable max depth |
| `interactive` | Adaptive sampling with adaptive AO; responsive preview |
| `debug` | Geometry diagnostics (normals, positions, IDs, etc.) |
| `test` | Minimal renderer for validation tests |

### CUDA/OptiX Source Layout

`.cu` files in `device/renderer/` are compiled to PTX and embedded as C++ string resources at build time. Each renderer has its own `*_ptx.cu` (raygen + hit/miss programs). Intersection programs for curves and custom geometry live in `device/geometry/`.
