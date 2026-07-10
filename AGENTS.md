# AGENTS.md

This file provides repository-wide guidance for coding agents working in this
repository.

## Project Overview

**VisRTX** is an experimental NVIDIA implementation of the [Khronos ANARI](https://www.khronos.org/anari) (Analytic Rendering Interface) standard, focused on scientific visualization. It ships two ANARI device implementations and an optional testing/demo framework (TSD).

All development should occur on top of the `next_release` branch, unless otherwise directed.

## Build

### Build Scope

The `tsd/` subproject and the individual devices (`devices/rtx/` and
`devices/gl/`) are independently buildable. When making changes, pay attention
to whether the work is isolated to one subproject or device, and prefer the
smallest relevant configure/build/test cycle before falling back to a full
repository build.

### Requirements

- CMake 3.17+, C++17 compiler, NVIDIA Driver 530+
- CUDA 12+ (for RTX device), ANARI-SDK 0.15.0+

### Basic Build

```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install /path/to/visrtx
cmake --build . --parallel
cmake --install .
```

### Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `VISRTX_BUILD_RTX_DEVICE` | ON | OptiX/CUDA ray-tracing device |
| `VISRTX_BUILD_GL_DEVICE` | ON | OpenGL 4.3/GLES 3.2 device |
| `VISRTX_BUILD_TSD` | OFF | TSD testing applications |

Component-specific options (MDL, Neural, NVTX, OptiX pinning, `TSD_USE_*`) are
documented in the subtree AGENTS.md files: [devices/rtx/AGENTS.md](devices/rtx/AGENTS.md),
[tsd/AGENTS.md](tsd/AGENTS.md).

## Tests

Run from the build dir:

```bash
ctest -C Release --output-on-failure
# or a single test
ctest -C Release -R <test_name> --output-on-failure
```

Test sources: `tsd/tests/` (TSD unit tests), `devices/rtx/apps/tests/api/` (RTX API tests). `ctest -N` lists the current set.

## Code Style

Google-style C++ formatting via `.clang-format`. C++17 throughout. Format with:

```bash
clang-format -i <file>
```

See [STYLEGUIDE.md](STYLEGUIDE.md) for detailed C++ and CUDA coding conventions.

## Architecture

### ANARI Devices (`devices/`)

Both devices implement the ANARI C API and are installed as shared libraries loadable by any ANARI application.

- **`devices/rtx/`**: OptiX-based GPU ray tracer (`libanari_library_visrtx`). Implements 30+ Khronos and NVIDIA ANARI extensions including CUDA array/framebuffer interop, NanoVDB volumes, and MDL materials. The heavy lifting is in OptiX launch parameters, hit programs, and CUDA kernels.

- **`devices/gl/`**: OpenGL rasterizer (`libanari_library_visgl`). Simpler implementation; supports shadow mapping and ambient occlusion.

### TSD: Testing Scene Description (`tsd/`)

TSD is explicitly **experimental**. It has no API stability guarantees. It is educational and for testing ANARI devices, not a production pipeline. It maintains its own scene graph that mirrors ANARI object state (editing, serialization, networking without device coupling), layered libraries from `tsd_core` up to `tsd_app`, and apps (`tsdViewer`, `tsdRender`, `tsdLua`, …).

Full library layering, design patterns, env vars: [tsd/AGENTS.md](tsd/AGENTS.md).
