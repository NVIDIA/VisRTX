# TSD Library Collection

This directory contains the main TSD libraries used by applications, tools, and
ANARI integration layers.

## Library Index

| Library | Directory | Description | Build Control |
| --- | --- | --- | --- |
| `anari_tsd` (`tsd_device`, `anari_library_tsd`) | [../anari_tsd/](../anari_tsd/) | ANARI device implementation backed by TSD, useful for capturing ANARI state in a Scene Archive for offline inspection in `tsdViewer`. | Built from `src/CMakeLists.txt` |
| `tsd_algorithms` | [algorithms/](algorithms/) | Image-processing kernels (tone mapping, auto exposure, AOV visualization, outline, buffer ops) with CPU and optional CUDA backends. | Always enabled (CUDA backend requires `TSD_USE_CUDA=ON`) |
| `tsd_app` | [app/](app/) | Application-facing glue for command-line import setup, ANARI device management, selection utilities, and offline sequence rendering. | Always enabled |
| `tsd_core` | [core/](core/) | Foundational utilities including typed value storage, tokens, containers, serialization trees, logging, timing, and task queue support. | Always enabled |
| `tsd_io` | [io/](io/) | Foreign-format importers/exporters, native TSD Archives, component serialization, and procedural scene generators. | Always enabled (some importers depend on optional build flags) |
| `tsd_scene` | [scene/](scene/) | Core scene representation with ANARI-like object hierarchy, parameters, layers, and animation support. | Always enabled |
| `tsd_rendering` | [rendering/](rendering/) | Render index and render pipeline layers that synchronize TSD scenes to ANARI worlds and render passes. | Always enabled |
| `tsd_mpi` | [mpi/](mpi/) | MPI helper types for rank-replicated state used in distributed workflows. | `TSD_USE_MPI=ON` |
| `tsd_network` | [network/](network/) | Boost.Asio messaging and scene synchronization primitives for networked client/server workflows. | `TSD_USE_NETWORKING=ON` |
| `tsd_ui_imgui` | [ui/](ui/) | ImGui-based UI framework and reusable viewer/editor windows and dialogs. | `TSD_BUILD_UI_LIBRARY=ON` |
| `tsd_lua` | [scripting/](scripting/) | Lua bindings and scripting runtime for scene authoring, automation, and viewer action extensions. | `TSD_USE_LUA=ON` |

## Notes

- Each library directory has a dedicated `README.md` with more detail.
- Application examples that consume these libraries are under [../../apps/](../../apps/).
