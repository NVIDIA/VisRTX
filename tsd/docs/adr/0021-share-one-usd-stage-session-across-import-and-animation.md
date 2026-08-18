# Share one USD Stage Session across an import and its animations

An imported USD Stage is held open by a **Stage Session**: the `UsdStage`, the
Hydra resolution chain built on it, and the Time Code both are currently
evaluated at, owned together. `import_USD` acquires one for the file it is
importing and consumes its chain instead of building a second one beside it;
every animation binding that import creates holds the same Session. A Session
is acquired from a process-wide registry keyed by absolute file path, held
weakly, so the last holder to let go closes the Stage. A fully static import
therefore retains nothing — the last reference drops when `import_USD` returns
— while an animated import pins the file for as long as its bindings live.

This extends ADR 0015's "Hydra owns resolution" from import time to every time.
It also amends ADR 0018, which
described per-scene stage retention and claimed a retained scene index the code
did not in fact keep: `UsdGeometryFileBinding` used to re-open its own raw
`UsdStage`, so a 1.6 GB stage was opened twice.

Both bindings resolve through the Session's chain rather than off the Stage's
schemas, so neither can drift from what the import converted: the instancer
binding re-runs the import's own `readInstancerPlacements()`, and the geometry
binding re-runs the import's own `resolveGeometry()` (ADR 0022). The Stage
itself is still read directly for the things Hydra does not model — authored
time samples, the `anari:` and `tsd:io:` vocabularies — which is what it was
always retained for (ADR 0015).

Keying by path rather than by Scene or by AnimationManager keeps USD knowledge
out of `tsd_scene` and `tsd_animation`, which sit below `tsd_io`, and makes
deserialization trivially correct: a binding stores only the file path and
rejoins by path, with no ordering dependency on anything else in the archive.

Two things follow from the Session owning the chain. Building the filter chain
moved into the Session, so an import and a scrub that both read the resolved
scene provably resolve identically; and dialect pruning moved *out* of the chain
into the import, so the Session carries no trace of any one import's options and
two imports of one file with different options can still share it. Pruning was a
filtering scene index; it is now a question the Import Context answers
(`ImportContext::isClaimed`), which every walk over the resolved scene asks.

The Session also owns the mapping from TSD's normalized animation time onto the
Stage's own Time Code, and USD evaluates continuously there rather than snapping
to the nearest authored sample. That is what usdview shows, and it fixes a real
aliasing bug: `example_granular_collision_sdf.usd` authors samples at time codes
1…400 on a stage range of 0…400, so under snapping every frame of a 400-frame
grid mapped to `400*i/399` and never landed on an authored code. Because the
mapping is a Stage-level fact living once in the Session, the `sampleTimes` and
`timeBase` caches every binding used to carry are deleted rather than
maintained: they were derived data, and re-deriving them from the Stage is
strictly more correct than trusting a possibly-stale copy. Archives written
before this omit them on write and ignore them on read, and silently gain the
continuous-time behavior. Versioning with a legacy snapping path was rejected:
it would keep behavior we decided was wrong alive forever.

One case has no Stage-level answer to re-derive. Nothing obliges a Stage to
author a `startTimeCode`/`endTimeCode` range, and USD reports zero for both ends
when it has not — which would map every animation time onto one Time Code and
freeze the scene. Animated prims therefore tell the Session what range their own
samples cover as they are bound, and the Session widens a fallback range with
them; a Stage that authored a range of its own remains the authority and is
never widened.

The costs are accepted knowingly. An animated import holds its file open —
1.6 GB for `example_apic_fluid.usd` — for the bindings' lifetime. Interpolating
between samples of a particle simulation, where index *i* is a different
particle from frame to frame, is physically meaningless; it is visually
harmless, it is what usdview does, and USD's own `interpolation` stage metadata
is the correct lever if it ever matters. And the shared Session makes threading
*sharper*, not safer: `setTime` is a global mutation every binding reads, so the
Session is a single serialization point by construction, and nothing here makes
a TSD scene safe to use across threads while USD is being read. ADR 0018 already
listed thread safety as unresolved and it stays that way.
