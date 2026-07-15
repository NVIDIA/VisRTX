# Emission descriptor and faithfulSet registration policy

An MDL material's emission is analyzed into an **immutable descriptor** that
*describes* — it never decides registration. A thin, renderer-side **policy**
decides which described slots become next-event-sampled Geometry Lights. This
splits a previously entangled classifier into three concerns:

1. **Classifier** (MDL-pure, in `libmdl`): walks the compiled material's emission
   DAG, emits an owned IR, and folds a descriptor `{verdict, edfKinds, magnitude,
   mode}` per slot plus the argument/resource dependencies the emission reads. It
   carries no renderer knowledge and retains no MDL-SDK expression pointers.
2. **Contract** (the seam): the descriptor shape plus a consumer-exported
   `faithfulSet` — the EDF kinds the renderer can evaluate faithfully on its
   synthetic next-event hit.
3. **Policy** (renderer-side): `register(slot)` iff the slot is consumed **and**
   `verdict ≠ ProvablyNull` **and** `edfKinds ⊆ faithfulSet` **and** the intensity
   mode is faithfully handled.

This supersedes the classification rule of
[ADR-0006](0006-sampleable-mdl-emission.md): the "emission not provably zero →
Emissive Surface" premise and the diffuse-EDF fidelity scope are preserved, but
the *decision* of whether to register moves out of the classifier into the
policy, and the classifier's output becomes a complete, honest descriptor rather
than a sampleability verdict. ADR-0006's synthetic-hit fidelity limits
(§EDF fidelity scope) and Pick-Power semantics remain in force.

## Emission IR (lowering the MDL expression DAG)

The classifier does **not** fold the MDL SDK's `IExpression` DAG directly. It
first lowers the compiled material's emission expressions (surface/backface ×
`{EDF, intensity, mode}` plus `thin_walled`) into an owned **emission IR** — a
one-time projection walked by `IExpression::get_kind()`:

| MDL SDK expression | IR node |
|---|---|
| `EK_TEMPORARY` (`IExpression_temporary`) | dereferenced and memoized, so a CSE-shared temporary collapses to a single node |
| `EK_CONSTANT` (`IExpression_constant` → `IValue`) | `Constant` — the value is copied out |
| `EK_PARAMETER` (`IExpression_parameter`) | `Parameter` |
| `EK_DIRECT_CALL` (`IExpression_direct_call`) | `Call` / `Texture`, tagged with `IFunction_definition::get_semantic()` |

Each node stores a `Semantic` enum, operand indices, copied constant values, and
resource names — and **no `IExpression` handle**. Three properties motivate the
lowering rather than folding `IExpression` in place:

- **Lifetime**: the device does not retain the compiled material, so any held
  `IExpression*` would dangle. The IR owns everything the fold reads and outlives
  the SDK objects.
- **Stable keying**: nodes key on `get_semantic()`, never DB names, so no user
  module can masquerade as a `df::`/`tex::` intrinsic.
- **SDK-runtime-free fold**: `foldEmissionDescriptor` interprets the IR with no
  MDL-SDK *runtime* calls or objects — link-level SDK-free. It still includes SDK
  headers for the `Semantic` type (`IFunction_definition::Semantics`) the IR keys
  on. What crosses into the renderer is the descriptor, which is fully SDK-free
  (no `mi::` types), letting the device consume it without the SDK.

Temporary memoization also makes the `−` exact-identity test decidable: shared
subexpressions become the same node, so the lattice's subtraction rule
(below) can prove `ProvablyZero` by IR-node ref-compare.

## Error model

The default renderer is a Quality path tracer whose forward estimator deposits
emission on any BSDF closest-hit at MIS weight 1 when the surface is **not**
registered as a light. This asymmetry drives the whole design:

- **Miss (false negative) = variance, not bias**, in unbounded Quality: an
  unregistered emitter still deposits via the forward path, unbiased, just noisier
  (no next-event contribution). Miss is *bias* (dark) only where no forward
  estimator survives — the Interactive/Matte/finite-depth modes below. (Even the
  default Quality path is finite — `maxRayDepth=5` — so a miss leaves a small
  final-vertex darkening; strict "variance, converges" holds only at unbounded
  depth. The renderer work item that closes the finite-depth gap removes this.)
- **Over-register (false positive) = BIAS**, not free perf: the next-event
  synthetic hit is fidelity-limited (geometric normal, synthesized tangent,
  object id 0, forced front — ADR-0006:102). Registering an EDF the renderer
  cannot evaluate faithfully makes next-event repay a *wrong* integrand while the
  correct forward deposit is MIS-downweighted — a systematic error the sample
  count never removes.

Therefore the policy registers **only** what is both non-null and faithfully
evaluable. An unfaithful or unknown EDF kind is *described, never registered* —
its light still arrives via the forward path, unbiased.

### Render-mode matrix (why a miss matters where)

| Mode | A missed (unregistered) emitter |
|---|---|
| Quality, unbounded depth | variance (converges) |
| Quality, `maxRayDepth=1` / final path vertex | dark |
| Matte receivers | dark |
| Interactive | dark receivers (no hit-side geometry-light MIS) |
| Fast | emission invisible regardless |

The dark cases are pre-existing forward-estimator gaps (today's classifier
already rejects every non-diffuse EDF, so those emitters are already
forward-only). Collapsing this matrix to "variance everywhere" is tracked as an
independent renderer work item, not a prerequisite of this policy.

## Lattice and op table

Analysis is a three-valued abstract interpretation over the **current immutable
snapshot** (argument block + resource table); the reactive re-fold owns future
writes. `ProvablyZero` means identically zero over all uv/state/time at the
current snapshot; any doubt joins to `Unknown`.

Scalar/color lattice `{ProvablyZero, ProvablyNonZero, Unknown}`; EDF lattice
`{ProvablyNull, ProvablyEmissive, Unknown}`.

- `·` is zero-absorbing; `+` is zero iff both operands are zero.
- `−`: `ProvablyZero` **only** on exact IR-node identity (CSE temporaries make
  ref-compare decidable; distinct-but-equal nodes fall to `Unknown`). Otherwise
  `Unknown`, never `ProvablyNonZero` (`1 − w` with `w` unknown can be zero).
- `?:`: fold if the condition folds to a constant at the current snapshot, else
  **join = least-upper-bound with `Unknown` as top**: `Zero ⊔ Zero = Zero`;
  `Zero ⊔ NonZero = Unknown`; any `Unknown ⇒ Unknown`. EDF lattice likewise:
  `Null ⊔ Null = Null`, `Null ⊔ Emissive = Unknown`, else `Unknown`.
- Mixes are analyzed conservatively: the fold unions the EDF kinds of every
  reachable component regardless of weight (a zero-weight component still
  contributes its kind) and never proves a mix `ProvablyNull`. Weight-based
  pruning — dropping a `ProvablyZero`-weighted component (cross-channel for
  color weights) — is a sound refinement, not yet implemented.
- `/`, `pow`, `exp`, and any **unmodeled node ⇒ `Unknown`**.
- Color zero test is **cross-channel**: `ProvablyZero` iff the max over *all*
  channels is zero — never a luminance/scalar reduction (`(+1,−1,0)` is not
  provably zero).
- Proofs assume finite values; a non-finite texel flags `Unknown`.

### Texture reductions

Per canvas, one pass at load / content-change yields
`{maxAbs, meanPositive, minValue}` per channel (memoized on the image's
content-version stamp):

- `maxAbs == 0 ⇒ ProvablyZero` — exact, no epsilon, over all texels and canvases.
- The gate is in **sampler-output space**: a transfer `T` with `T(0) ≠ 0`
  (LUT / ICC / nonzero border) breaks the stored-texel bound ⇒ `Unknown`. The
  standard MDL transfers (`tex::gamma_*`) and `wrap_clip` satisfy `T(0) = 0`.
- The reduction also yields a per-channel **`minValue`**: `minValue ≥ 0` over all
  texels proves the texture's `sign` contribution is `ProvablyNonnegative`; a
  negative texel makes it `Unknown`.
- The magnitude proxy is `meanPositive` — the per-channel mean of `max(texel, 0)`
  — never mean-absolute (see the negative-emission decision). It never gates
  zero; it is non-negative by construction so a CDF can represent it. This one
  magnitude sizes the emissive Pick Power for **every** emissive material type —
  native PBR and MDL alike read it, not a separate signed mean. The two coincide
  for any registerable emitter (whose texels are all ≥ 0), and `meanPositive`
  stays CDF-valid even for the signed emitters that never register.

## Descriptor and contract

```
SlotDesc = { verdict: Null|Emissive|Unknown,
             edfKinds: set<diffuse|spot|directional|measured|unknown>,
             magnitude: meanPositiveRadianceProxy (per channel, >= 0),
             mode: radiant_exitance|power,
             dependsOnGeometricState: bool,   // intensity/EDF reads normal/tangent/objectId/position
             sign: ProvablyNonnegative|Unknown }

EmissionDescriptor = {
  surface: SlotDesc, backface: SlotDesc,   // each folded from its own sub-expressions
}
```

The emission's argument/resource dependencies are **not** on the descriptor —
they live on the owned IR (`EmissionIR`), computed by `collectDeps`:

```
emissionDeps: set<argBlockSlot>    // structural — every branch of every ?:
resourceDeps: set<resourceSlot>
```

The descriptor is **complete and honest**: it describes the backface slot and
unfaithful EDF kinds even though today's consumer ignores them. The classifier
needs no change when the renderer's fidelity grows — only `faithfulSet` grows.

Two per-slot flags make the faithfulness gate *sufficient*, not just necessary
(an EDF-kind check alone is not enough — see Considered options):

- **`dependsOnGeometricState`** — set when the emission (EDF **or** intensity)
  reads a geometric-state quantity the synthetic next-event hit fabricates:
  shading normal (`state::normal`, bump), tangent (`state::texture_tangent_*`),
  object/instance id, or position. Such emission evaluates a *different*
  integrand at the synthetic hit than at the real forward hit, so registering it
  biases even when the EDF kind is faithful. The fold detects it structurally
  from the IR (any reachable `state::` intrinsic in the diffuse-fidelity set).
- **`sign`** — `ProvablyNonnegative` when no reachable emission value can be
  negative (constants with all channels ≥ 0; textures whose reduction proves a
  non-negative minimum; otherwise `Unknown`). Signed emission renders correctly
  via the forward path (the device does not clamp), but it cannot be *registered*
  faithfully: an all-negative next-event contribution is dropped by the shadow
  ray's positive-contribution epsilon gate while the forward deposit is
  MIS-downweighted, under-applying the light. So only `ProvablyNonnegative`
  emission is registerable; the rest stays forward-only, unbiased.

`faithfulSet` is a single consumer-exported constant. Today it is `{diffuse}`
(ADR-0006: emission must be invariant to normal/tangent/object-id). It lives in
one header (`devices/rtx/device/material/EmissionPolicy.h`) and is cross-
referenced from the two GPU-side sites that encode the same diffuse assumption
(`lightPickPower.h` double-sided Lambertian flux, `sampleLight.h` double-sided
normal orientation) so the assumption has a single source of truth.

### Registration policy

```
register(slot) iff consumed(slot)                       // call-site: the caller passes only consumed slots (surface today)
               and slot.verdict ≠ ProvablyNull          // ┐ isRegisterable(slot) — the five
               and slot.edfKinds ⊆ faithfulSet          // │ faithfulness conjuncts, in
               and slot.mode is faithfully handled       // │ EmissionPolicy.h (radiant_exitance today)
               and not slot.dependsOnGeometricState      // │ synthetic hit fabricates it
               and slot.sign == ProvablyNonnegative      // ┘ else forward-only, unbiased
```

`consumed(slot)` is not a term inside `isRegisterable`; it is expressed by which
slots the caller folds and tests (surface only today — backface is described but
not yet consumed).

`Unknown` *verdict* with faithful kinds (and the two faithfulness flags
satisfied) ⇒ register: the worst case is a spurious zero-radiance diffuse light,
genuinely perf-only and unbiased, because the magnitude proxy is non-negative and
the emission is state-invariant. Any unfaithful/unknown *kind*, a
geometric-state dependence, a possibly-negative value, or a power mode ⇒ do not
register — the forward path keeps that light unbiased; registering would bias.

## Considered options

- **Descriptor vs verdict.** The classifier emits a descriptor and a separate
  policy decides (chosen). Rejected: the classifier deciding sampleability
  directly (ADR-0006's model) — it entangles renderer fidelity into MDL-pure
  analysis, so every fidelity gain (a new faithful EDF, backface consumption)
  requires editing the classifier. A describing classifier plus a thin policy
  restores renderer-agnosticism: `faithfulSet` grows, classifier unchanged.
- **`faithfulSet` export.** A static `constexpr` set in one renderer header,
  cross-referenced from the GPU lockstep sites (chosen). Rejected: a queried
  runtime capability — the renderer owner is the same team, and a compile-time
  constant makes the three lockstep sites (classifier gate, Pick-Power flux,
  synthetic-normal orientation) impossible to desynchronize silently.
- **Negative-emission policy / magnitude proxy (#6).** The device does **not**
  clamp negative emission (`MDLShader_ptx.cu`; the host mirrors this deliberately,
  as ADR-0006 established), so signed emission must keep rendering correctly.
  Chosen: **signed emission renders via the forward path; only
  `ProvablyNonnegative` emission is registered, with a non-negative
  `meanPositive` magnitude proxy.** This is the only coherent choice: a
  mean-**absolute** proxy (an earlier draft) gives a negative emitter a *positive*
  Pick Power, so next-event selects it, but its all-negative shadow-ray
  contribution is then dropped by the positive-contribution epsilon gate while
  the forward deposit is MIS-downweighted — the light is under-applied, a bias
  the sample count never removes. Making the magnitude non-negative *and* gating
  registration on `sign == ProvablyNonnegative` keeps negative emission on the
  unbiased forward path (a purely-negative emitter folds to magnitude 0 → zero
  Pick Power → never NEE-selected → forward deposit at weight 1). Rejected:
  clamping negative emission to zero **on the device** — it would let any signed
  emitter register but silently changes rendered output; a device clamp is a
  separate change (renderer ticket) that, if made, would let `sign` treat clamped
  emission as non-negative. The host magnitude and the device hit-side pNee must
  read the **same** non-negative proxy (consumer obligation below), or a CDF
  built from `meanPositive` disagrees with a hit-side pNee derived from signed
  radiance — MIS bias. Today's constant path feeds signed radiance to the hit-side
  pdf, which this ADR flags as a renderer work item.
- **Emission `.mode`.** The descriptor records `radiant_exitance` vs `power`;
  only `radiant_exitance` is faithfully handled today, so `power` is
  described-but-not-registered (forward-path only), identical to current behavior
  but now principled rather than a hard classifier rejection. Power-mode
  area-normalization is a follow-up.
- **Reactive re-fold and atomic publish.** The descriptor is re-folded whenever
  the emission's argument/resource dependencies change, which happens on the
  device's existing commit path (all argument writes flow through
  `syncParameters` inside `finalize`); the existing light-set refresh
  (`refreshEmissionLightSet` → `lastLightSetChange`) then republishes light data,
  CDF, and sampleability flags together in one host-serialized pre-launch rebuild.
  No new double-buffer machinery: "atomic publish" reduces to the invariant *every
  descriptor change affecting verdict or magnitude bumps `lastLightSetChange`
  before launch*, enforced by test.

## Consequences

- The classifier becomes MDL-pure and independently unit-testable without a GPU:
  `libmdl` is a standalone static library, so the static pass, IR, and fold are
  exercised device-free against inline `.mdl` snippets.
- ADR-0006's `EmissionClassification` (constant radiance + textured-diffuse flag
  + single-factor dynamic recipe) and its string-prefix DAG walk are replaced by
  the owned IR + descriptor fold, keyed on inlined `df::` **semantics**
  (`IFunction_definition::get_semantic`) rather than DB-name prefixes.
- `PhysicallyBasedMDL` continues to read its post-translate keys (it does not
  introspect the DAG) but now publishes the same descriptor type, so both raw and
  wrapper materials feed one policy. Its live-sampler-mean Pick Power is preserved.
- `Unknown`-verdict diffuse emission now **registers** (worst case a spurious,
  unbiased zero-radiance light) where the old classifier's non-fold fell back to a
  unit proxy; spot/measured/power emission is **described but not registered**,
  identical rendered result to today (forward-only) but now via an explicit
  fidelity gate rather than an EDF-kind rejection.
- Volume emission (`SLOT_VOLUME_EMISSION_INTENSITY`) is out of scope; the surface
  and backface slots are the analysis subject.

## Follow-ups

1. Synthetic-hit fidelity (real normal/tangent/object-id, angular EDF eval) →
   grows `faithfulSet` to spot/directional/measured.
2. Consume the descriptor's `backface` slot (side-aware hit/PDF/Pick-Power).
3. Collapse the render-mode matrix: forward-estimator deposits in
   Interactive/Matte/finite-depth so a miss is variance in every mode.
4. Float-CDF dim-light bias: preserve every positive mass or derive hit-side
   `pNee` from the quantized CDF delta.
5. `intensity_power` mode area-normalization; register power-mode emitters.
6. Hit-side pNee must read the **same** non-negative `meanPositive` magnitude the
   Pick-Power CDF is built from; today the constant path feeds signed radiance to
   the hit-side geometry-light pdf, which would disagree with a `meanPositive` CDF.
   (A device negative-emission clamp would resolve this and would let `sign` treat
   clamped emission as non-negative.)

These are filed as independent renderer tickets; none blocks the classifier.
