# USD MaterialX import — known gaps

Observations from importing MaterialX USD Stages that are understood well enough
to record but not yet diagnosed. Measured against
`OpenPBRShaderPlayground-1.0/ShdrPlygrnd/ShdrPlygrnd_OpenPBR.usda` — 55
MaterialX materials, 117 texture inputs.

**UDIM tile sets are reported, not imported.** 89 of the 117 texture inputs.
This one *is* decided; see
[ADR 0019](adr/0019-report-udim-tile-sets-as-unsupported.md).

**One material fails to transcode.** *Diagnosed and closed.* The device logged
`MaterialX: failed to transcode '<inline document>': Could not find a nodedef
for node 'Surface'`, then an MDL compile error, then fell back to the default
material, for exactly one material out of 55.

`Surface` turned out to be a red herring: hdMtlx names the surface node after
the shader prim, so all 54 emitted documents contain a node by that name. The
material is `/World/Looks/OJfoam`, and the defect is in the asset.
`materials/OJfoam.mtlx` connects `geometry_opacity` — declared `float` on
`open_pbr_surface` — to `mtlxcolorcorrect2`, a `color3` node. MaterialX matches
a node to its definition on category, type *and* the exact set of inputs, so
the mistyped input leaves the surface node resolving to no nodedef at all.
MaterialX 1.39.6 rejects the source `.mtlx` standalone with `Mismatched types in
port connection`, so nothing between the Stage and codegen introduced it.

TSD cannot fix the asset, but it no longer emits a document the device cannot
compile. `documentResolves` in `UsdMaterials.cpp` checks every node against the
standard libraries before emission; a document that fails is reported as
`MATERIAL_RESOLUTION_FAILED` naming the offending port, and the material falls
back to the portable preview-surface mapping. Regression coverage is the
`color3`-into-`float` scenario in `tests/test_UsdImport.cpp`.

The check sits *after* the texture pass and *before* sampler creation, which is
load-bearing: the fallback mapping reads the network by UsdPreviewSurface
names, so it reports none of the tile sets the MaterialX path found. Checking
first silently dropped `OJfoam`'s two tile sets from the report — 89 texture
load failures became 87 — which is the ADR 0019 failure mode arriving by the
back door. The reported counts are unchanged from baseline; what changed is 6
fewer ANARI errors and one material that now says why it fell back. The render
is byte-identical, since `OJfoam` is not visible from `renderCam_CU_meetMATandTube`.

Setting `TSD_USD_MATERIALX_DUMP_DIR` writes each generated document there,
named after the material prim. That is how the above was isolated, and it is
the tool to reach for when a device rejects a document.

**hdMtlx validation warning on every material.** `Input 'geometry_opacity'
doesn't match declaration: <open_pbr_surface ...>`. Not version skew, as
previously recorded, and not benign — it is MaterialX reporting the mistyped
port above, and the warning fires for the materials that author
`geometry_opacity`, not for all 55. Three of the four author it as `float` and
generate fine; `OJfoam` is the one that does not.

**MDL logs a resolve failure for every texture, including bound ones.**
`Failed to resolve texture resource <abs path>`. MDL treats a leading `/` as
root-relative to a registered resource search root rather than as a host path,
and TSD never sets the device's `mdlResourceSearchPaths` parameter
(`ANARIDeviceManager::initialDeviceParams` is the existing seam, with no caller
populating it). Cosmetic for any input that has a sampler bound, since the
sampler supplies the texels — but it is noise that hides real failures.

Setting that parameter was tried and is **blocked**, not merely undone. Measured
on the reference asset, by setting `mdlResourceSearchPaths` on the device right
after `anari::newDevice` in `tsdOffline`:

| value | resolve failures | distinct unresolved (UDIM) | render |
|---|---|---|---|
| unset | 109 | 103 (85) | baseline |
| `.../ShdrPlygrnd/textures` | 109 | 103 (85) | identical to baseline |
| `/` | 0 | 0 (0) | differs from baseline |

Two conclusions, and they close off the obvious fix from both ends.

Anchoring on the directories the importer already knows is a no-op, and no set
of per-texture anchors can ever be anything else. Root-relative means resource
`/a/b/c.tif` under registered root `R` is looked for at `R/a/b/c.tif`, so with
the fully absolute host paths `UsdMaterials.cpp` writes into the document, the
only root that can match is `/` itself.

Registering `/` does silence every failure, and regresses
[ADR 0019](adr/0019-report-udim-tile-sets-as-unsupported.md) while doing it. The
85 UDIM paths resolve too; `libmdl::Core::resolveResource` returns
`get_element(0)->get_filename(0)`, which is tile 1001, and
`SamplerRegistry::loadFromImage` binds it. The render visibly changes. TSD's own
`texture load failed` count stays at 90, so the import report would go on
claiming the tile sets were skipped while the device quietly draws one tile of
each — the exact "reported gap becomes a silently incorrect render" outcome that
ADR 0019 rejects.

So silencing the noise needs a decision that has not been made. Three routes are
open: stop writing the absolute path into the document for UDIM inputs, which
buys `/` at the cost of the well-formed-path property ADR 0019 deliberately
keeps; or fix it device-side, by not attempting resolution for an input that
already has a sampler bound, which is where the noise is genuinely cosmetic; or
accept the noise and record why. The first two both need an ADR.

**Setting `mdlResourceSearchPaths` as a string array segfaults the device.**
It is a `:`-separated `ANARI_STRING`; passing it via
`anari::setParameterArray1D` with `ANARI_STRING` crashes in
`helium::BaseDevice::unmapArray` rather than being rejected. Found while probing
the above. Unrelated to USD import, and in `devices/`, not TSD.

## Measuring

Reproducing any of this needs three things that are not in the repo: the
reference asset, a TSD build with `TSD_USE_USD=ON` and `TSD_USE_OIIO=ON` (the
asset's textures are TIFF), and a VisRTX build with both MDL and MaterialX
enabled, installed where `anari::loadLibrary` finds `visrtx_mtlx`. A standalone
TSD build is enough for TSD's own side; the device comes from the parent repo.

`tsdOffline` defaults to the `visrtx` library, which has no MaterialX shader
generation and so reports none of the failures above -- pass `--lib
visrtx_mtlx` explicitly. It also prompts for a camera:

```bash
F=.../ShdrPlygrnd/ShdrPlygrnd_OpenPBR.usda
echo 8 | ./tsdOffline --lib visrtx_mtlx -usd_mtlx $F -o /tmp/out.png -s 1 -w 128 -h 96 > /tmp/run.log 2>&1
```

Redirect with `> log 2>&1`, not `2>&1 > log`, or the ANARI errors miss the file.
`-s 1 -w 128 -h 96` collects device errors without waiting on a real render.
Grep the log for `sampler not bound`, `failed to transcode`,
`Failed to resolve texture resource`, and `texture load failed`.

Renders are a poor oracle here — the asset's default lighting is dark and noisy,
so "did the texture bind" cannot be judged by eye. Byte-comparing two PNGs from
otherwise identical runs is a usable oracle for "did anything change at all",
which is how the UDIM regression above was caught. Assert in the suite instead;
`tests/test_UsdImport.cpp` and `tests/test_Importers.cpp` carry the fixture
patterns, both hand-writing decodable 1x1 files because the decoders need real
ones.
