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

`Geometry::declarePreparationParameters(parameters)` is the opt-in seam, called
once in the subtype constructor before its first commit. A subtype must list
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
lifetime is intentional. For a multi-geometry input batch, obsolete snapshot
references are dropped after commit-time retirement checks and parameter reads,
before replacement input uploads are allocated. This avoids retaining a whole
cohort of obsolete device copies through the replacement batch. Single-geometry
and observer-only updates retain the existing snapshot replacement order during
full preparation.

Geometry-wide attribute values are excluded because they do not change geometric
area or bounds.
Geometry Light synthesis uses the material's emission estimate, while sampled
attribute-driven radiance reads the updated geometry registry data at rendering
time. Material changes continue to use their existing light-set invalidation.

## Correctness checks

`TestGeometryPreparationReuse` exercises seven types with host and shared-device
inputs. It verifies all five geometry-wide attributes and their removal against
the initial full-preparation rendering, stable bounds, and no preparation/BVH-build
diagnostic messages. Debug rendering modulates attributes by geometric facing, so the initial
full-preparation image is the exact per-shape reference, with a separate nonzero
control. Scalar/string edits, array ranges, in-place host unmaps, and device-array
privatization act as positive controls requiring preparation and BVH rebuilding.

The test also stages an input removal **after** a geometry-wide attribute commit:
that uncommitted edit must not leak into the committed update. Emissive materials
request area data where supported. Existing numeric-failure tests additionally verify that a
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
- For the seven opt-in geometry types, parameter commits capture references to
  dirty declared host arrays. The first geometry finalizer submits the captured
  inputs across geometries on the preparation stream and waits once before any
  finalizer consumes them. Shared arrays are deduplicated by handle and data
  version. Sources remain retained through completion; only successful geometry
  submissions mark their still-current array versions uploaded. A failed
  geometry remains unprepared even if another geometry successfully shares one
  of its arrays. Legacy consumers can safely use completed copies afterward.
- Observer-only updates, or arrays modified after submission, retain the local
  per-geometry upload path. A later geometry commit or observer update may retry
  a failed input generation; unchanged failed inputs cannot silently become
  prepared. Preparation reuse does not capture or submit an upload batch.
- These checked geometry kernels no longer import the default stream: authored
  inputs are ready before they run. Other array consumers retain their existing
  upload path.
- The BVH builder imports both render-stream producers (including legacy field
  and isosurface work) and default-stream input uploads. Its build, readbacks,
  and compaction execute on the preparation stream.
- OptiX instance arrays and proxy AABBs are uploaded on the preparation stream
  before their builds. World tracing tables (surface/volume/light records, proxy
  records, and Light Pick arrays) are batched after their final CPU edits; proxy
  flags no longer require a separate re-upload.
- All dirty object registries are uploaded in one preparation-stream batch before
  tracing. Registry pointer access no longer performs implicit uploads. Stable
  host snapshots survive object creation or edits during the batch; only dirty
  slots are staged unless registry growth requires a full copy. Edits made after
  a snapshot remain dirty for the next batch. No registry lock is held during
  CUDA waits, so completion callbacks cannot be blocked by that lock.
- Host bounds/area summaries are collected after batched submission (see below).
  BVH publication still requires preparation-stream completion. BVH scratch
  survives through that completion; handles/bounds are
  published only on success. Group/World failure state prevents failed builds
  from being treated as ready and permits a later retry.
- Parameter-reference replacement and last-public-release privatization retain
  host waits for old render consumers. Error handling drains participating work
  before storage can be retired or retried.

This is not concurrent in-place scene replacement. Legacy default-stream events
can include preceding work on blocking streams; tangent generation and field-grid
preparation retain their legacy synchronization, and `cudaMalloc`/`cudaFree` can
still serialize work. Frame/renderer payloads and group index tables still use
legacy uploads. Removing those dependencies requires further producer migration
and allocation ownership changes. Overlapping replacement with active rendering also
requires separate storage generations and last-use tracking for all consumers,
including shared geometry and registries.

`TestPreparationStream` checks explicit producer ordering and independence from
an unrelated nonblocking stream. The shared-cudart failure tests additionally
cover preparation dependency/completion failures and BLAS/TLAS readback/completion
failures, publication suppression, and retry. Host-upload cases also check copy
stream selection, allocation/copy/completion failures, partial-batch retry, and
an in-place unmap coalesced with a geometry-wide attribute commit. Scene-upload cases cover
instance allocation, instance/world copy and completion errors, registry copy
and completion errors, partial registry batches, and retry without recommitting
already-finalized objects. `TestRegistrySnapshot` verifies stable copy sources
across host-vector growth and preservation of newer dirty edits.

## Geometry submission and collection

Checked bounds and area-CDF preparation queue kernels and asynchronous readbacks
into geometry-owned pinned storage. A successful submission is not publication:
scalar outputs, numeric validation, and geometry registry updates happen only
after collection. The device queue retains each submitted geometry, and thereby
its arrays and derived buffers, until those operations finish.

Group/World rebuilds collect queued finalizations before constructing BVHs. Lazy
Geometry Light area requests are submitted for all candidate surfaces first,
then collected together before configuring lights. Shared geometry is queued
once. A numeric failure invalidates its geometry only; a stream-completion error
invalidates every pending generation, including errors observed by an intervening
upload or legacy completion wait. Failed generations cannot take preparation reuse.

A recommit or observer-only re-finalization drains the geometry's previous queued
work before reusing its storage. Last-public-release privatization of shared CUDA
arrays also drains pending preparation readers, not just rendering. Device
teardown collects the queue before releasing objects or CUDA/OptiX resources.
Status callbacks cannot reenter device property queries or rendering while
collection is resolving results; those calls are refused instead of publishing
a partially collected scene.

### Small-geometry computation batches

For 1–256 primitives, bounds and area requests are held as device-owned batches
of descriptors. One CUDA block handles each geometry in a single batch launch:
block-local reductions produce its bounds summary, and a double-precision
block scan produces its area CDF. Bounds and area requests for the same geometry
share a descriptor when both arrive before the completion boundary. Requests
above 256 primitives retain the general multi-block/CUB path. Zero-count
handling is unchanged. SDF bounds retain their zero epsilon-scale convention.

Every `finishGeometryPreparation()` flushes these descriptors before waiting,
including intervening upload waits and shared-array privatization. This is
necessary because a queued descriptor still references its original inputs even
before GPU work has been launched. Collection retains owners through descriptor
upload, dispatch and readback completion. Descriptors remain stable through
fallback drains; dispatch or transfer failure invalidates the pending batch,
while numeric failures remain local to their geometry.

Small jobs write both result slots into a contiguous device-owned result buffer.
The batch performs one asynchronous transfer into a reusable pinned host buffer,
then scatters only completed, requested slots into the geometries' summaries.
Even unused slots are initialized before transfer. A bounds-only flush followed
by a later area request preserves the earlier bounds summary. Failed transfers
or completion do not scatter partial results. Both batch buffers grow to their
high-water size and remain alive through fallback drains; the per-geometry
pinned leases and general-path readbacks retain their existing ownership.

`TestSmallGeometryBatch` exercises mixed sphere/cone/cylinder/triangle batches
at 1, 31, 32, 33, 255, 256 and 257 primitives with host and device-backed logical
array slices. It checks independent bounds, zero-area light removal, recovery,
and transitions between small and general paths. Failure injection covers the
descriptor upload, dispatch-status check, readback and batch device/pinned
allocation, including batch-wide invalidation and retry. With shared CUDA runtime
interposition, the mixed-size test also checks one summary copy per collection
(two initial collections for bounds and lazy areas, then one for combined jobs).

Pinned readback storage is leased from the device's `GeometryReadbackPool`.
Each page holds 256 geometry leases, each containing two result slots (a 4 KiB
page with the current result layout). A geometry keeps its lease across updates;
returning it after collection makes the slot reusable without a CUDA allocation.
The pool caches pages at its high-water mark until device teardown rather than
trimming on every geometry destruction. A lease retains its page independently
of the pool's lifetime, so cleanup cannot free storage while a lease still exists.
This reduces allocation count, not necessarily requested bytes for tiny scenes:
a partially occupied page still reserves the whole page.

Pool/page locks protect slot accounting but are never held across CUDA allocation
or free. A failed page allocation leaves existing leases intact and clears CUDA
last-error state before retry. Concurrent growth may allocate an extra page;
source/result lifetime ordering remains the collection queue's responsibility.
`TestGeometryReadbackPool` covers first-page/growth failure and retry, slot reuse,
move/reset, pool-before-lease teardown, concurrent leases, and asynchronous DMA
into separate slots on a page.

Committed host inputs now complete in a cross-geometry batch before finalizers
run. Observer-only input updates remain conservative. Last-public-release of
shared host arrays during an active upload joins the preparation stream before
the application may free its source memory. Quad/curve generated-data completion,
tangent generation, and field/isosurface
preparation retain their current synchronous paths. This does not introduce
independent scene generations or concurrent in-place replacement during tracing.

`TestDeferredGeometryPreparation` covers recommitting a queued geometry, retained
captured-device inputs, last-reference queue ownership through teardown, unrelated
numeric failures, and collection-time callback reentry. Injection cases also
cover pinned-readback allocation failure, completion failure across multiple
queued geometries, an intervening upload completion failure, and shared CUDA
array retirement while preparation remains pending. `TestGeometryInputBatch`
checks submission-before-wait across multiple geometries, shared-array dedup,
preparation-preserving and observer-only controls, coalesced unmaps/commits, and host-source
retirement during an active copy. Input failure cases cover batch-wide completion
failure and recovery through an array unmap without a geometry recommit.

## Remaining work

Broader array/shading invalidation precision, reducing the extra CPU snapshot
comparison cost, batched readback/dispatch, and fine-grained area-vs-bounds
invalidation are separate improvements. Registry, bounds, and area-result
collection and committed input uploads now span geometries. Observer-only input
batching, legacy finalizers, and allocation-related synchronization require
further producer migration while preserving source lifetimes and failure
propagation. This change does not repack or download
inputs, introduce CPU preprocessing, or weaken the guarantee that changed inputs
are prepared on the GPU before their derived data is published.
