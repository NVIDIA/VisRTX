# Sampleable MDL emission via a general, limitation-agnostic path

Any MDL material whose emission is **not provably zero** becomes an Emissive
Surface — and thus a next-event-sampled Geometry Light. Emissiveness is
determined host-side, **per material kind, from the value each kind actually
holds** — no recompilation, no re-resolution of class-compiled expressions
(the dynamic recipe below only reads values the host already holds: the live
argument block and bound samplers):

- **Raw `mdl`** (author-supplied source): a compile-time introspection of the
  compiled material, run while it is alive in the material registry and cached
  per-uuid. If `surface.emission.emission` is `df::diffuse_edf` (radiant-exitance
  mode) and `surface.emission.intensity` folds to a **constant** (dereferencing
  `let`-block temporaries), the host-known emission radiance is that constant
  **divided by PI** — the device emission callable returns `edf * intensity`, and
  a diffuse EDF's value is `1/PI` — sampleable iff nonzero. If the EDF is diffuse
  but the intensity does **not** fold (texture / procedural / parameter-driven —
  note a parameter with a default stays symbolic under class compilation, so only
  body-literal intensities fold), it is textured emission: sampleable, radiance
  evaluated on the device. When the symbolic intensity is a SINGLE argument- or
  texture-driven factor times folded constants — a color/float parameter, or
  the parameter `tex` of a `tex::lookup_color` (body-literal textures fold and
  fail the walk); e.g. the `value * PI` authoring idiom — the
  classification additionally records a **dynamic recipe** — the argument name
  plus the folded radiance-domain scale — so the host resolves a live mean from
  the current argument block (or bound sampler mean) at light-build time.
  Anything outside that shape keeps the unit-luminance proxy as the selection
  weight. Otherwise (no emission, or a non-diffuse EDF) it is not sampleable.
- **Wrapper materials** (`PhysicallyBasedMDL`, which maps an ANARI `emissive`
  input): read the committed parameter from its **post-translate keys**
  (`emissive.value` / `emissive.texture`, which persist in parameter storage; the
  pre-translate `emissive` key is consumed by the translation at first commit, so
  reading it would silently zero on a later unrelated commit). A bound
  `emissive.texture` sampler gives sampleability with the LIVE sampler mean as
  Pick Power (the host light build queries it live — the sampler may finalize
  after the material's commit; the hit-side MIS pdf reads the same mean from
  the material's finalize-time GPU snapshot, equal in steady state and at most
  one flush apart transiently, as in native PBR; variance-only, the compiled
  EDF at the synthetic hit supplies the true per-point radiance);
  otherwise a nonzero `emissive.value` gives sampleability, and that value IS the
  emitted radiance (the wrapper authors `intensity = emissive * PI`, and the
  callable's `1/PI` cancels it) — the exact average, matching native PBR. The
  wrapper does **not** introspect the compiled DAG, and this is not a workaround:
  under class compilation `emissive` is a symbolic argument, so the wrapper's
  compiled intensity — a `texture_isvalid(emissive.texture) ? … : emissive.value`
  ternary times PI — never folds to a constant, and even `emissive=0` would read
  as "textured, hence a light." The host already holds the wrapper's value, so it
  reads it rather than reverse-engineering it out of a class-compiled DAG.
  Sampleability is gated on the `emissive` binding only — opacity/baseColor
  texturing does not affect it (they never enter the emission intensity), matching
  native PBR.

Both are the same rule ("emission not provably zero, from the material alone")
applied with the strongest evidence each kind has. The compiled emission EDF is
evaluated at the synthetic next-event hit through the same callable the path-hit
deposit uses (a pure `optixDirectCall`, correct since the `edf * intensity`
radiance fix), so next-event and deposit radiance agree and MIS stays unbiased.
The guiding constraint is unchanged: **current MDL-integration limitations
restrict only which emission is faithfully evaluable (hence testable), never
whether a material is treated as an emitter.**

## Considered options

- **Wrapper emissiveness — how.** Read the committed ANARI `emissive` parameter
  (chosen). Rejected: unifying the wrapper into the raw-`mdl` DAG classifier — a
  class-compiled wrapper's intensity is never a constant, so the classifier would
  mark every wrapper (including `emissive=0`) as textured-emissive, reopening
  over-inclusion. Making the DAG path work for the wrapper would require resolving
  the class-compiled `emissive.value`/`emissive.texture` arguments and folding the
  `ResolveColorInput` `texture_isvalid` ternary — disproportionate to reading the
  value we already hold. Instance-compiling with injected arguments, and
  resolving the argument block against the intensity sub-expression, were both
  rejected for the same reason (net-new plumbing for a value the wrapper has in
  hand).
- **Pick Power.** For a constant emitter the average is the emission radiance:
  raw-`mdl` folded intensity **divided by PI** (the device emission callable
  returns `edf * intensity`, and a diffuse EDF's value is `1/PI` — the earlier
  "return intensity directly" convention was the pre-fix PI-too-bright bug);
  the wrapper's `emissive.value` is that radiance already (its `* PI` authoring
  cancels the `1/PI`). No scaling choice here can bias MIS — the average feeds
  only the Light Pick and is read identically on both estimator sides — but
  `intensity/PI` is the *exact* weight, keeping raw-`mdl` constants comparable to
  native/wrapper emitters in a mixed scene. For non-constant emission the
  dynamic recipe (see the classification above) resolves a live mean from the
  current argument value or bound sampler mean; only expressions outside the
  single-factor shape fall back to the unit-luminance proxy (`vec3(1)`) — as
  do runtime resolution failures (argument missing or wrong-typed, sampler
  unbound or default-textured, non-finite product).
  The proxy is variance-only in exact arithmetic but NOT harmless in practice:
  under-picking a bright emitter in a mixed scene concentrates its energy in
  rare overweighted picks, which the firefly clamp then cuts and the last-depth
  MIS truncation cannot repay — the raw-`mdl`-dimmer-than-PBR regression the
  multi-light parity section pins. The wrapper's bound sampler mean IS
  host-known, so that is its weight (matching native PBR). A device numerical
  estimate for arbitrary raw-`mdl` expressions is deferred.
- **Sidedness.** Triangle Geometry Lights are double-sided (the deposit orients
  the shading normal toward the incoming ray; the triangle sampler pdf uses
  `|cos|`), so the synthetic next-event hit's normal is oriented toward the
  receiver for the triangle path. Analytic sphere/cylinder/cone lights already
  emit an outward, single-sided normal and cull far-side samples, and are
  untouched. True MDL front/back sidedness is deferred.
- **EDF fidelity scope.** The synthetic next-event hit carries the geometric face
  normal, a synthesized tangent frame, and object id 0. Next-event equals the
  deposit exactly only for emission invariant to shading normal, tangent, and
  object id (the diffuse-like case, which is also the only EDF the raw-`mdl`
  classifier accepts). Direction-, tangent-, or object-id-dependent emissive EDFs
  are out of scope; enriching the synthetic hit is a follow-up.

## Consequences

- No recompilation and no argument-block resolution beyond reading the live
  block. Raw-`mdl` emissiveness is a compile-time classification cached per-uuid
  (constant radiance, a textured-diffuse flag, and the optional dynamic recipe —
  instance-agnostic; each material instance plugs its own argument values in),
  evicted alongside the registry's material release; the wrapper reads its ANARI
  parameters. Both are known by finalize; the light-set
  refresh runs from `MDL::finalize` (after `syncSource`/`syncParameters`), and the
  wrapper's existing `commitParameters` refresh call is removed — refreshing
  before the classification exists reads a stale flag.
- Raw `mdl` emissive materials, previously deposit-only, now also drive
  next-event estimation and engage MIS on the deposit — strictly lower variance,
  unbiased, but a rendering change for existing MDL-emissive scenes.
- `emissionIsConstant` stays `false` for raw `mdl` (its EDF path is always
  taken; the resolved value supplies only the Pick Power average). The wrapper
  keeps its constant fast path (`emissionIsConstant = true` for a nonzero inline
  `emissive`): its baked radiance equals its EDF evaluation exactly, so either
  path is MIS-consistent.
- The framebuffer test seam confirms emissive materials **light** correctly but
  **cannot** observe that a non-emissive material is **excluded** from the light
  set (a zero-radiance light is pixel-identical to no light). That exclusion is
  guarded by a device-level assertion on the light count in a no-authored-light
  scene (`numLightInstances` equals the synthesized count there; exposing
  `countGeometryLights` is an optional sharper seam). The count is valid only
  after the light build.
- Conservative over-inclusion is possible and harmless: a raw `mdl` whose diffuse
  intensity is argument-driven from zero, or an all-black bound emissive texture,
  resolves as textured -> a zero-radiance Geometry Light (unbiased, wasted pick
  slot). The wrapper's `emissive=0` case is caught exactly by its parameter read,
  and its all-black texture by the zero sampler mean.
- Testable now: constant and textured MDL emission (including an away-facing
  triangle emitter). Not testable until `scene_data` is wired: attribute-driven
  emission.

## Follow-ups

1. Device numerical estimate for non-constant Pick Power (power-weighted CDF).
2. Wire `scene_data_lookup_*` -> `readAttributeValue` (MDL attribute probing).
3. True MDL front/back sidedness (`backface.emission`, `thin_walled`).
4. `intensity_power` mode area-normalization; consume `mdlEmissionMode`.
5. Feed interpolated normal + authored tangent + real object id into the
   synthetic hit, widening fidelity past diffuse-like EDFs.
6. Recover opacity/cutout in next-event emission (pre-existing gap for native
   constant emitters too).
