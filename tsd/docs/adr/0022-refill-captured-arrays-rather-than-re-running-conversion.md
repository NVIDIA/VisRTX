# Re-fill captured Arrays rather than re-running conversion per frame

A USD animation binding re-fills the TSD Arrays the import created. It does not
re-run the converter that created them. The converter that *creates* a target
Array is the same code that *constructs the binding* and hands it that Array
plus the prim path, so there is no second inference pass and nothing has to find
the array again by name.

Re-running conversion per frame was the obvious alternative and is the wrong
one. It allocates new Arrays every frame, rebinds parameters, and churns object
identity — which forces the render index to tear down and recreate ANARI handles
instead of updating buffers — and it re-creates the Materials and Surfaces that
did not change. The scenes that motivated this work make the scale plain:
`example_apic_fluid.usd` scatters 531,441 instances, about 34 MB of matrices per
frame. Nothing about that survives a per-frame rebuild. The part that varies
over time is the contents of a handful of Arrays; it is not the Surface and
Material graph around them.

Concretely, a point instancer becomes one transform-array node per
`(instancer, prototype)` pair, and one binding per pair re-reads the instancer
from the Stage Session at the current Time Code, re-applies the same
per-prototype instance-index selection and visibility mask the importer applied
— through the same `readInstancerPlacements()` the importer calls, so the two
cannot drift — and writes through `Array::setData()`. Deforming geometry works
the same way on `vertex.position` and `vertex.normal`.

A TSD `Array` has no resize, so an element count that moves mid-sequence
allocates a right-sized Array and rebinds: `setAsTransformArray` for an
instancer, a parameter rebind for geometry. That costs nothing on the common
path and pays handle churn only on the frames where the count actually moves.
Detecting constant counts at import time was rejected: proving it means reading
every sample — all 1.6 GB of `apic_fluid` — and sampling first/mid/last is a
heuristic that would have passed on all three motivating scenes while still
being wrong in general.

The one case this deliberately does not handle is a mesh whose vertex count and
topology both move. Points, indices and primvars are then one consistent set
that has to be re-pulled together, and re-pulling them together through
triangulation and per-Surface attribute binding (ADR 0020) *is* re-running the
converter. The binding detects that case, warns once, and leaves the frame as
imported rather than writing a mesh whose positions and indices disagree.
Nothing in the available data exercises it; a fuller "split converters into
resolve → plain data / data → TSD objects" refactor is what would fix it
properly, and that is a refactor project with a feature attached rather than
part of this one.

Because a scrub writes one Array per animated instancer and
`RenderIndexAllLayers` rebuilds its world on every `ANARI_FLOAT32_MAT4` array
unmap, batching is not optional: `AnimationManager` brackets a time change in
`Scene::beginUpdateBatch()`/`endUpdateBatch()`, and a render index coalesces the
rebuilds it owes until the batch ends. The existing `beginLayerEditBatch` could
not be reused — it batches *structural* layer changes, and an array unmap is not
one. Without the new bracket, a stage with several animated instancers would pay
a full world rebuild per instancer per frame. None of the three motivating
scenes would have caught that, since each has exactly one.
