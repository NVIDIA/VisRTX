# Geometry attribute names: initial investigation

Status: prototype implemented and validated in the `geom_attrib_names` worktree.
Baseline: next_release at 51d32f5f. Scope: RTX native MDL scene-data bindings.

## Intended application behavior

One MDL material can request `temperature` while different geometries store
that quantity in different existing ANARI attribute slots. Applications should
be able to share the material instead of cloning it solely to change bindings.

A minimal API consistent with the existing material feature is:

```cpp
anariSetParameter(device, geometryA, "sceneData.temperature",
    ANARI_STRING, "attribute0");
anariSetParameter(device, geometryB, "sceneData.temperature",
    ANARI_STRING, "attribute2");
// Supply vertex.attribute0 / vertex.attribute2 through the normal array API.
// Both surfaces can reference the same mdl material.
anariCommitParameters(device, geometryA);
anariCommitParameters(device, geometryB);
```

Provisional extension name: `ANARI_VISRTX_GEOMETRY_MDL_SCENE_DATA`.
This is an MDL-specific name-to-slot alias map, not additional data channels.
The initial scope keeps attribute0..3 and color, existing interpolation and
existing float/color lookup support. It does not add arbitrary named arrays,
integer/matrix lookups, instance naming, or shared world values.

An alternative is `attribute0.name = "temperature"`. That reads naturally as
naming a slot and could become a general geometry naming extension, but limits
each slot to one name and differs from the existing name-to-slot material API.
Decide the intended broader scope before promising a generic extension name.
No collision is apparent in the inspected VisRTX implementation; external
ANARI extension naming and specification review remains to be done.

## Proposed precedence

Use an explicit geometry binding first, then the existing material binding
when the geometry does not bind that name. This lets geometry describe its own
layout while material bindings remain useful defaults. Existing applications
without geometry aliases retain their behavior.

Resolve the name to a slot before resolving the slot's data. A geometry alias
that selects an absent attribute should produce MDL's default value, not silently
fall back to a different slot selected by the material. Unsetting the geometry
alias restores material fallback. Reject invalid target strings with a warning
and omit them from the committed alias snapshot; document this separately from
a valid binding whose data happens to be missing. Empty names should be rejected.

Material-first precedence is also feasible, but would prevent geometry aliases
from overriding a shared material's existing binding. This is an API decision,
not an ANARI attribute-data precedence rule imposed by the current code.

Once a slot is selected, reuse `resolveAttribute` and the existing attribute
cache. Geometry/instance constants, instance arrays, primitive, vertex and
face-varying sources retain their current precedence and uniform eligibility.
Names do not make a varying attribute uniform. A name declared on geometry can
therefore still obtain an instance value when normal slot resolution selects it.

## Current implementation and the missing bridge

- `MDL::syncSceneData()` scans material `sceneData.*` string parameters.
- `libmdl::SceneDataDescriptor` owns target-code strings at their original IDs.
- `mdl::SceneDataInstance` converts material aliases into an ID-indexed slot table.
- `MaterialGPUData::materialData.mdl` owns the GPU table reference.
- `MDLShader_ptx.cu` places that table in the per-hit TextureHandler.
- `MDLTexture_ptx.cu::sceneDataSlot()` selects the slot; `sceneDataValue()`
  checks cached presence, uniform eligibility and finite values. `data_isvalid`
  uses the same slot resolution but intentionally does not read values.
- `Geometry` currently stores attribute data but no name map.

The MDL string ID is local to the target code, not a device-global identifier.
The same geometry can be used by materials with different ID assignments.
Copying a material's numeric IDs into GeometryGPUData would therefore be wrong.

## Preferred prototype: device-local interned name tokens

Intern exact name strings on the host, giving each distinct string a stable
device-local token (reserve zero as invalid). Keep tokens stable for the device
lifetime initially; do not treat an unchecked string hash as identity. Account
for token exhaustion and the memory cost of applications generating many names.

Store committed geometry aliases as a compact GPU list of token/slot pairs.
Allow multiple names to refer to a slot. Add a token alongside each entry in
the material's ID-indexed scene-data table. Material binding updates continue
to populate its fallback slot. Target-code descriptor creation supplies the
tokens even for names that have no explicit material binding.

At shading initialization, pass the hit geometry's alias list to TextureHandler.
For a lookup, use its material-local ID to fetch a token, search the geometry
list for that token, and otherwise use the material fallback slot. Both
`sceneDataValue` and `scene_data_isvalid` must use this single resolution helper.
An empty geometry list preserves the existing constant-time material-only path.

Advantages: no per-surface material clones, no material recompilation for a
geometry-name edit, and no geometry/material cross-product cache. Costs: extra
GPU metadata and an alias-list search per lookup. Measure this cost with a
realistic number of names; choose a different lookup representation if needed.
Interning must be synchronized or confined to a known host owner: the existing
MDL coordinator/worker architecture must not gain unsynchronized shared state.

Alternative: resolve a dense ID-to-slot table per surface (geometry/material
pair) on the host and pass that at each hit. This keeps GPU lookup cheap but
requires invalidation when either object changes, more storage proportional
to surfaces and descriptor sizes, and wiring through synthetic geometry-light
hits. Prefer it only if measurement justifies that extra lifetime machinery.

## Implementation checkpoints

1. Finalize extension name, alias syntax, precedence, invalid-input semantics,
   and initial geometry subtype coverage. Add extension JSON, dependencies,
   extension-query exposure and README examples using existing conventions.
2. Parse geometry aliases in the common Geometry commit path as a complete
   snapshot so unset parameters disappear. Own the strings and uploaded list.
   Audit each subtype's GPU-data population to initialize the new fields.
3. Add host name-token ownership and descriptor-to-token conversion without
   changing MDL backend string IDs or compiled-material cache identity.
4. Extend shared GPU structs and TextureHandler; resolve geometry override
   then material fallback in the common scene-data slot helper.
5. Audit edit propagation: GPU upload, accumulation reset, resource lifetime,
   and geometry destruction. Name-only edits should not require PTX rebuilds;
   inspect whether current Geometry finalization unnecessarily rebuilds BLAS.
6. Audit all MDL evaluation paths, especially emission on synthetic light hits,
   so direct lighting and visible emission resolve the same names. Preserve
   emission classification's treatment of scene-dependent expressions.

## Validation required before publishing an implementation

- One shared material on two geometries with different name-to-slot maps.
- One shared geometry with two materials whose target-code string IDs differ.
- Conflicting geometry/material bindings; missing geometry alias; unsetting;
  valid alias with missing data; invalid strings; multiple aliases for one slot.
- Existing material-only results unchanged, including data_isvalid and missing,
  nonfinite, uniform versus varying, and default-return behavior.
- Primitive, vertex and triangle face-varying values; geometry constants and
  instance fallback. Test indexed data where the selected source needs it.
- Retained edits after rendering, geometry/material replacement and destruction,
  multiple devices, descriptor rebuild, and existing parallel MDL compilation.
- Named emission through visible hits and geometry-light sampling where supported.
- Existing MDL scene-data and rendering regressions; build with MDL disabled.
- Performance comparison for no geometry aliases and representative alias counts.

No source implementation or build was performed during the initial investigation.
Keep future build output under this worktree (for example `_build/mdl`); never
reuse the parent checkout's CMake cache or build directories.

## Implementation decisions (2026-11-17)

The prototype adopts `ANARI_VISRTX_GEOMETRY_MDL_SCENE_DATA`, the
`sceneData.<name>` syntax and geometry-first precedence described above.
Initial advertised subtypes are triangle, quad, sphere, cylinder, cone, curve,
SDF and isosurface. It does not extend the ANARI attribute channel count.

- `SceneDataNames` owns exact, case-sensitive strings per device, synchronizes
  interning with a mutex, reserves token zero for the empty name and checks
  32-bit token exhaustion. Tokens and their strings live until device teardown.
  Descriptor conversion happens in material source synchronization on the commit
  thread, after collecting the compile result. No worker or compiler-cache
  identity changes are required.
- `SceneDataInstance` retains a name token for every target-code string-table
  entry even without a material binding. Updating the material snapshot replaces
  fallback slots while retaining tokens. Geometry commits upload their complete
  alias snapshots through the common Geometry path; invalid entries warn and
  are omitted, and unsetting the last alias releases its GPU buffer.
- Each geometry GPU record carries its alias pointer/count. Shading initialization
  passes these to the MDL TextureHandler. Float/color lookups and `data_isvalid`
  share `resolveSceneDataAttribute`, which searches tokens before selecting the
  material fallback. Attribute presence, finite-value handling and uniform
  eligibility remain in the existing cache lookup.
- Synthetic geometry-light hits already point at the real geometry GPU record.
  They use the same shading initialization, so this path needs no special alias
  cache or changes to emission classification/registration policy.
- Alias edits retain compiled target code and renderer pipelines. Current Geometry
  finalization still marks the surface BLAS dirty for all parameter edits; removing
  that rebuild requires broader geometry dirty-state tracking and is deferred.
- Expanding material table entries exposed an upload call passing byte size to
  the element-count overload. It now uploads `data.size()` entries.

## Validation results (2026-11-17)

Environment: Windows, Visual Studio 2022, CUDA 13.2, OptiX 7.7, ANARI 0.16.0,
MDL SDK 2025.0.5, NVIDIA RTX A3000 Laptop GPU (driver 595.97). Builds and logs
are isolated under this worktree's `_build/` directory.

- MDL-enabled Release library and selected MDL regression targets built under
  `_build/mdl`; MDL-disabled Release library and rendering smoke test built under
  `_build/no-mdl`.
- `TestMdlSceneData`: passed through CTest. Covers shared materials/geometries,
  target-code replacement, geometry/material precedence, unsetting, missing and
  invalid aliases, multiple names per slot, color, nonfinite values, presence,
  uniform eligibility, indexed vertex/primitive/face-varying data, instance-array
  fallback, curves, retained edits, accumulation and destruction. Alias-only edits
  preserve the material PTX fingerprint. Existing material-only scene-data and
  animation-time checks also pass.
- Named emission through synthetic geometry-light hits: the camera sees only a
  diffuse receiver, with the emitter outside its view. The material uses
  `math::saturate` so the existing nonnegative-emission policy admits it as a
  Geometry Light. Changing its geometry alias from the slot holding 0.8 to the
  slot holding 0.4 changes receiver radiance from 0.036212 to 0.018106. The
  existing emission registration policy is unchanged.
- `rtx::MdlSceneDataInstance`, `rtx::MdlSceneDataDescriptor` and
  `rtx::MdlEmissionClassifier`: passed. The instance test includes reordered
  target-code IDs, device independence and concurrent interning of 8,000 names
  across eight host tasks.
- MDL-disabled `TestDenoiserReuse`: passed through CTest.
- `git diff --check`: passed.

The optional `TestMdlSceneData --benchmark` measures the median of 16 GPU frame
durations after four warm-up frames, at 256x256 pixels and 64 samples per pixel:

| Geometry aliases | GPU frame duration | Relative to zero aliases |
| --- | --- | --- |
| 0 | 53.183 ms | baseline |
| 8 | 53.729 ms | +1.0% |
| 32 | 56.540 ms | +6.3% |

This is one microbenchmark on a laptop GPU, not a general throughput guarantee
or a comparison against the pre-feature binary. The linear list is acceptable
for the prototype; larger alias maps may justify a different representation.

## Remaining limitations and coverage gaps

- The existing `rtx::MdlMaterialRegistry` test fails its Windows canonical-module
  recovery fixture (`source acquire failed (no fallback permitted)`). The same
  failure was reproduced using the untouched parent checkout's registry test,
  MaterialRegistry, TaskExecutor and PTX source files, linked against a fresh
  build of unchanged libmdl code. This feature does not change that path.
- Validation uses normal Release compiler settings. An initial failed compiler
  detection left an empty flag cache; an assertion-enabled attempt then hit the
  existing World light/surface-array cursor check before the surface array is
  rebuilt. Restoring normal Release flags resolved that validation setup issue;
  assertion-enabled World ordering remains outside this feature's scope.
- External extension-name/specification review, MaterialX-specific coverage,
  neural geometry, every individual geometry subtype, and broad application
  performance testing remain follow-ups. Common Geometry setup and GPU-data
  population were audited across implemented subtypes.
- Geometry finalization still rebuilds BLAS for alias-only edits, and distinct
  interned names remain allocated until device destruction.
