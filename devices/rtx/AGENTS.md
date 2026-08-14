# AGENTS.md

This file provides guidance for coding agents working in the `devices/rtx/`
subtree.

## What This Is

The `devices/rtx/` subdirectory contains the OptiX-based GPU ray-tracing ANARI device (`libanari_library_visrtx`). See the repo-root `AGENTS.md` for project-wide context.

## Domain Docs

- [`CONTEXT-MAP.md`](./CONTEXT-MAP.md) — bounded contexts and how they relate; each context has its own `CONTEXT.md` glossary ([Frontend](./device/CONTEXT.md), [Render Pipeline](./device/renderer/CONTEXT.md), [World](./device/world/CONTEXT.md), [MDL](./libmdl/CONTEXT.md)). Use these terms in code and docs.
- [`docs/adr/`](./docs/adr/) — architecture decision records. Scan the directory and read the relevant one before "fixing" a design it locks in.

## Build

Built as part of the repository build — see the repo-root [AGENTS.md](../../AGENTS.md). `CMAKE_PREFIX_PATH` must locate the ANARI-SDK and OptiX.

Key device-specific CMake options:

- `VISRTX_ENABLE_MDL_SUPPORT`: NVIDIA MDL materials (requires MDL SDK)
- `VISRTX_ENABLE_MATERIALX_SUPPORT`: MaterialX materials
- `VISRTX_ENABLE_NEURAL`: Neural Graphics Primitives (requires OptiX 9.0+)
- `VISRTX_ENABLE_NVTX`: NVTX profiling markers
- `OPTIX_FETCH_VERSION`: Pin OptiX version: `7.7`, `8.0`, `8.1`, or `9.0`
- `VISRTX_MIN_ARCH`: Minimum compute capability (default `50`, or `75` under
  CUDA 13, which dropped pre-Turing codegen — an explicit floor below `75` there
  is a configure error). Single source for the host kernel PTX target, the OptiX
  module PTX target, and the MDL backend's `sm_version`. The host kernels and
  the OptiX module both target `compute_<MIN_ARCH>` (PTX, JIT-compiled by the
  driver/OptiX to the actual GPU at launch); the MDL backend instead picks the
  highest entry its fixed `sm_version` set offers at or below the floor, so a
  floor with no exact match (e.g. `89` → `86`) rounds down — safe, since lower
  SM PTX runs on newer GPUs. `VISRTX_ENABLE_NEURAL` raises the floor to at least
  `89` (cooperative vectors need Ada). Override the host side with
  `-DCMAKE_CUDA_ARCHITECTURES=<...>` (e.g. `all-major`) to bake per-arch cubins
  instead of shipping PTX; the OptiX module target ignores that override and
  stays at `compute_<MIN_ARCH>`.

## Tests

Test sources: `apps/tests/api/` (e.g. `ctest -C Release -R TestSpheres`). Run via ctest — see the repo-root [AGENTS.md](../../AGENTS.md).

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

SBT callable slots per material type (one per `SurfaceShaderEntryPoints` value; `device/gpu/sbt.h` is authoritative):

- `Initialize`, `EvaluateNextRay`, `EvaluateTint`, `EvaluateOpacity`, `EvaluateEmission`, `EvaluateTransmission`, `EvaluateNormal`, `EvalBsdf`, `EvaluatePdf`

`MaterialGPUData::callableBaseIndex` holds the offset into the callable table. Kernels dispatch shading via `optixDirectCall(callableBaseIndex + SHADE_FN, ...)`.

Spatial field samplers also use callables (`SpatialFieldSamplerEntryPoints`): every field family provides the Woodcock-loop bodies `SampleDistance`, `RatioTrackTransmittance`, `RayMarchVolume`; custom fields add `Init`/`SampleValue`/`SampleNormal`, dispatched callable-in-callable from those loop bodies (built-in families sample inline via ADL instead).

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
| `interactive` (aka `default`) | Adaptive sampling with adaptive AO; responsive preview |
| `debug` | Geometry diagnostics (normals, positions, IDs, etc.) |
| `test` | Minimal renderer for validation tests |

### CUDA/OptiX Source Layout

`.cu` files in `device/renderer/` are compiled to PTX and embedded as C++ string resources at build time. Each renderer has its own `*_ptx.cu` (raygen + hit/miss programs). Intersection programs for curves and custom geometry live in `device/geometry/`.
