# USD MaterialX import — known gaps

Observations from importing MaterialX USD Stages that are understood well enough
to record but not yet diagnosed. Measured against
`OpenPBRShaderPlayground-1.0/ShdrPlygrnd/ShdrPlygrnd_OpenPBR.usda` — 55
MaterialX materials, 117 texture inputs.

**UDIM tile sets are reported, not imported.** 89 of the 117 texture inputs.
This one *is* decided; see
[ADR 0019](adr/0019-report-udim-tile-sets-as-unsupported.md).

**One material fails to transcode.** The device logs
`MaterialX: failed to transcode '<inline document>': Could not find a nodedef
for node 'Surface'`, then an MDL compile error, then falls back to the default
material. Exactly one material out of 55. The error comes from MaterialX's own
`ShaderGraph.cpp`, so shader generation inside the device is what fails, not
TSD's emission. No prim named `Surface` exists on the composed Stage and none of
the 55 `.mtlx` files contains a node by that name, so the name is synthesized
somewhere between the Hydra material network and MaterialX codegen. Diagnosing
it needs the failing inline document dumped from the device.

**hdMtlx validation warning on every material.** `Input 'geometry_opacity'
doesn't match declaration: <open_pbr_surface ...>`. OpenPBR nodedef version skew
between the asset (MaterialX 1.39) and the locally built MaterialX. Non-fatal:
documents still generate and render.

**MDL logs a resolve failure for every texture, including bound ones.**
`Failed to resolve texture resource <abs path>`. MDL treats a leading `/` as
root-relative to a registered resource search root rather than as a host path,
and TSD never sets the device's `mdlResourceSearchPaths` parameter
(`ANARIDeviceManager::initialDeviceParams` is the existing seam, with no caller
populating it). Cosmetic for any input that has a sampler bound, since the
sampler supplies the texels — but it is noise that hides real failures.

## Measuring

`tsdOffline` defaults to the `visrtx_matx` library and prompts for a camera:

```bash
F=.../ShdrPlygrnd/ShdrPlygrnd_OpenPBR.usda
echo 8 | ./tsdOffline -usd_matx $F -o /tmp/out.png -s 1 -w 128 -h 96 > /tmp/run.log 2>&1
```

Redirect with `> log 2>&1`, not `2>&1 > log`, or the ANARI errors miss the file.
`-s 1 -w 128 -h 96` collects device errors without waiting on a real render.
Grep the log for `sampler not bound`, `failed to transcode`,
`Failed to resolve texture resource`, and `texture load failed`.

Renders are a poor oracle here — the asset's default lighting is dark and noisy,
so "did the texture bind" cannot be judged by eye. Assert in the suite instead;
`tests/test_UsdImport.cpp` and `tests/test_Importers.cpp` carry the fixture
patterns, both hand-writing decodable 1x1 files because the decoders need real
ones.
