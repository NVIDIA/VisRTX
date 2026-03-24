## tsdViewer

`tsdViewer` is the main interactive TSD application.

It is used to:

- Load scene data from many supported file formats.
- Inspect and edit objects, layers, parameters, animations, and transfer
  functions.
- Compare ANARI devices and renderer behavior in one UI.
- Save and reload full TSD sessions (`.tsd`) for offline inspection.

## Command-Line Usage

```bash
./tsdViewer [viewer-options] [scene-options]
```

CLI parsing is split into two layers:

- Viewer options are consumed by `tsd::ui::imgui::Application`.
- Scene/import options are consumed by `tsd::app::Context`.

### Parsing Rules

- If no importer flag is active, a positional filename is treated as a state
  file (`.tsd`) to load.
- If an importer flag is active, following filenames are loaded with that
  importer until another importer flag appears.
- `-l` or `--layer` changes the target layer for subsequent imports.
- If no files are provided and no state file is loaded, `tsdViewer` generates a
  default material-orb scene.

## Viewer Options

| Option | Description |
| --- | --- |
| `--noDefaultLayout` | Do not apply built-in ImGui dock layout at startup. |
| `--noDefaultRenderer` | Do not auto-select a default renderer in the viewport. |
| `--secondaryView <library>` / `-sv <library>` | Enable secondary viewport and set its ANARI library. |

## Scene and Import Options

| Option | Description |
| --- | --- |
| `-v`, `--verbose` | Enable verbose logging. |
| `-e`, `--echoOutput` | Enable log echo output mode. |
| `-l <name>`, `--layer <name>` | Set destination layer for following imports. |
| `-camera <file>`, `--camera <file>` | Provide camera file path (if used by workflow). |
| `-ensight_fields <f1,f2,...>` | Field selection list used with `-ensight`. |
| `-xf <file>`, `--transferFunction <file>` | Load transfer function file for subsequent `-volume` import(s). |
| `-blank` | Start with an empty scene (no generated default scene). |

## Importer Flags

Use one of these flags before filenames to select how those files are loaded:

- `-tsd`
- `-agx`
- `-assimp`
- `-assimp_flat`
- `-axyz`
- `-dlaf`
- `-e57xyz`
- `-ensight`
- `-gltf`
- `-hdri`
- `-hsmesh`
- `-nbody`
- `-obj`
- `-pdb`
- `-ply`
- `-pointsbin`
- `-pt`
- `-silo`
- `-smesh`
- `-smesh_animation`
- `-swc`
- `-trk`
- `-usd`
- `-vtp`
- `-vtu`
- `-xyzdp`
- `-volume`

Note: Availability of some importers depends on CMake feature options and
optional dependencies.

## Examples

Load a saved TSD session:

```bash
./tsdViewer state.tsd
```

Load geometry and an HDRI environment:

```bash
./tsdViewer -obj model.obj -hdri studio.exr
```

Load multiple files into explicit layers:

```bash
./tsdViewer -l points -ply cloud.ply -l mesh -obj part.obj
```

Load a volume with a transfer function:

```bash
./tsdViewer -xf fire.xf -volume density.nvdb
```

Start secondary viewport on another ANARI library:

```bash
./tsdViewer -sv visgl -gltf scene.gltf
```

## Environment

`TSD_ANARI_LIBRARIES` controls the ANARI library list shown in the UI (comma-
separated names). If unset, `tsdViewer` falls back to built-in defaults.
