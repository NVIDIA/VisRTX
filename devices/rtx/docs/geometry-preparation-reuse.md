# Preparation-preserving geometry updates

## Terminology

A **geometry-wide attribute** supplies one value for the entire geometry:
`attribute0`–`attribute3` or `color`, rather than a per-vertex or per-primitive
attribute array. The storage/shading representation may still call these values
*uniform attributes*; that term describes their scope, not an update strategy.

A **preparation-preserving update** reuses successfully prepared geometry because
its preparation inputs are unchanged. Geometry-wide attribute edits can take
this path, but not if coalesced with an input change or applied to a failed
preparation. A scalar parameter such as `radius` is still a preparation input.

## Scope

Triangle, quad, sphere, curve, cone, cylinder, and SDF geometries opt into a
preparation-preserving finalization path. Changing only geometry-wide attributes
republishes the geometry registry slot without regenerating tangents, indices,
radii, bounds, or area CDFs, and without invalidating the surface BLAS timestamp.
Normal object-finalization timestamps still advance, so automatic accumulation
reset remains intact. Before replacing parameter references, the input-retirement
check compares the declared committed preparation inputs and referenced-object
versions with the last successful preparation. Unchanged inputs skip the work-
stream wait; first preparation, changed inputs, failed preparation, and non-opt-in
types retain it. This check performs no GPU uploads. Parameter lookup names are
cached once per geometry rather than rebuilt on every finalization. Finalization
still revalidates values, object versions, and backing pointers before reusing
derived data: dependencies can be updated between commit and finalization, so the
earlier retirement verdict is not a reusable preparation verdict.

This is deliberately not general fine-grained invalidation. Any changed
preparation input takes the existing full preparation path, including shading
arrays and scalar/string parameters that could be narrowed further later.
Isosurfaces, optional neural geometry, and unknown geometries retain their
existing full-preparation behavior; they do not opt in.

## Reuse contract

`Geometry::beginPreparation(parameters)` is the opt-in seam. A subtype must list
**every consumed parameter except geometry-wide attributes**, including inputs
read lazily by tangent generation or `gpuData()`. These are the subtype's
preparation inputs. The base adds primitive and vertex
attribute arrays and primitive IDs. When adding a subtype parameter, update its
list or keep that subtype on the default conservative path. The declaration is
fixed per subtype; do not conditionally add/remove names based on current values.

The helper compares committed parameter values, not live staging iterators.
For referenced arrays it also records object update/finalization timestamps,
data modification timestamp, and device backing pointer. Consequently:

- Changed handles, values, logical ranges, or backing pointers invalidate reuse.
- A geometry-wide attribute commit coalesced with an array unmap or
  privatization cannot hide it.
- Observer-only finalization always follows the full preparation path.
- Missing/invalid initial data and previously failed preparation cannot reuse
  uninitialized or failed derived state.
- Reuse becomes eligible only after a valid, successful geometry upload.

Only present parameters occupy snapshot storage. The cached `AnariAny` values
retain internal references to the current input objects until replacement or
geometry destruction; they do not copy authored data. This also prevents cached
parameter identity from becoming a dangling object pointer. The extra reference
lifetime is intentional; old snapshot references are dropped on the next full
preparation after the retirement check.

Geometry-wide attribute values are excluded because they do not change geometric
area or bounds.
Geometry Light synthesis uses the material's emission estimate, while sampled
attribute-driven radiance reads the updated geometry registry data at rendering
time. Material changes continue to use their existing light-set invalidation.

## Correctness checks

`TestGeometryPreparationReuse` exercises seven types with host and shared-device
inputs. It verifies all five geometry-wide attributes and their removal against
the initial full-preparation rendering, stable bounds, and no preparation/BVH-build diagnostic
messages. Debug rendering modulates attributes by geometric facing, so the initial
full-preparation image is the exact per-shape reference, with a separate nonzero
control. Scalar/string edits, array ranges, in-place host unmaps, and device-array
privatization act as positive controls requiring preparation and BVH rebuilding.

The test also stages an input removal **after** a geometry-wide attribute commit:
that uncommitted edit must not leak into the committed update. Emissive materials request area data
where supported. Existing numeric-failure tests additionally verify that a
geometry-wide attribute commit cannot rehabilitate a failed generation.

## Local performance checks

Use the [benchmark skill](../../../.agents/skills/benchmark-geometry-preprocessing/SKILL.md)
to compare against a revision before the optimization with matched builds.
Measure `small-recommit` for bounds/BVH readiness and add
`--extra-case reuse-render:32:64:recommit:1` for registry publication and a
completed render. Keep replacement and unchanged-frame cases as guards.
The [benchmark guide](geometry-preprocessing-performance.md) defines the timed
boundaries and local result interpretation; no historical baseline is required.

## Preparation stream and retirement

Checked geometry preprocessing (bounds, area CDFs, generated quad/curve indices
and radii, and SDF alignment) and OptiX BVH build/compaction use one device-owned
nonblocking preparation stream. `utility/PreparationStream.h` owns its lifetime
and the event used to import producer dependencies. Scene-update submission is
serialized; each event wait captures the corresponding record before the event
is reused.

- Full preparation imports prior render work before overwriting derived buffers.
- Legacy array uploads are imported through an event on the default stream.
- The BVH builder imports both render-stream producers (including legacy field
  and isosurface work) and default-stream input uploads. Its build, readbacks,
  and compaction execute on the preparation stream.
- Host bounds/area summaries and BVH publication still require preparation-stream
  completion. BVH scratch survives through that completion; handles/bounds are
  published only on success. Group/World failure state prevents failed builds
  from being treated as ready and permits a later retry.
- Parameter-reference replacement and last-public-release privatization retain
  host waits for old render consumers. Error handling drains participating work
  before storage can be retired or retried.

This is not concurrent in-place scene replacement. Legacy default-stream events
can include preceding work on blocking streams; tangent generation and field-grid
preparation retain their legacy synchronization, and `cudaMalloc`/`cudaFree` can
still serialize work. Removing those dependencies requires stream-aware upload
and allocation ownership. Overlapping replacement with active rendering also
requires separate storage generations and last-use tracking for all consumers,
including shared geometry and registries.

`TestPreparationStream` checks explicit producer ordering and independence from
an unrelated nonblocking stream. The shared-cudart failure tests additionally
cover preparation dependency/completion failures and BLAS/TLAS readback/completion
failures, publication suppression, and retry.

## Remaining work

Broader array/shading invalidation precision, reducing the extra CPU snapshot
comparison cost, batched readback/dispatch, and fine-grained area-vs-bounds
invalidation are separate improvements. This change does not repack or download
inputs, introduce CPU preprocessing, or weaken the guarantee that changed inputs
are prepared on the GPU before their derived data is published.
