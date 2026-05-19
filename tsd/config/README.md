# TSD User Color Maps

This directory contains example user color-map files for TSD applications.

To make TSD pick up a color map, copy a `.1dt` file into your user config
color-map directory:

- Linux/macOS: `~/.config/tsd/colormaps/`
- Windows: `%APPDATA%/tsd/colormaps/`

TSD loads these files once at application startup. The filename stem becomes
the transfer-function editor dropdown name, so `Sunset Test.1dt` appears as
`Sunset Test`.

Each non-comment line should contain whitespace-separated `r g b a` values.
For auto-loaded color-map presets, TSD uses only RGB values; alpha is ignored.
Manual transfer-function load/save still treats `.1dt` as full RGBA transfer
function state.
