# Re-fill captured Arrays rather than re-running conversion per frame

Geometry conversion is split in two. **Resolving** a gprim — reading its
topology and primvars, refining it, triangulating it, expanding and gathering
its attributes — produces plain data and is the half that changes over time.
**Building** turns that data into Surfaces, Geometries, Materials and Arrays,
and is the half that does not. A USD animation binding re-runs the resolve half
and writes the result over the objects the import built; it never re-runs the
build half.

`resolveGeometry()` touches no Scene and returns a `ResolvedGeometry`: a list of
Parts, one per Surface the prim yields, each holding its attributes as inert
buffers. `refillGeometry()` writes one Part over one existing Geometry. The
import calls both in sequence; a scrub calls only the second.

Re-running conversion per frame was the obvious alternative and is the wrong
one. It allocates new Arrays every frame, rebinds parameters, and churns object
identity — which forces the render index to tear down and recreate ANARI handles
instead of updating buffers — and it re-creates the Materials and Surfaces that
did not change. The scenes that motivated this work make the scale plain:
`example_apic_fluid.usd` scatters 531,441 instances, about 34 MB of matrices per
frame. Nothing about that survives a per-frame rebuild. The part that varies
over time is the contents of a handful of Arrays; it is not the Surface and
Material graph around them.

Point instancers get the same treatment without needing the split, because a
prototype's placements are already one flat buffer: an instancer becomes one
transform-array node per `(instancer, prototype)` pair, and one binding per pair
re-reads the instancer from the Stage Session at the current Time Code,
re-applies the same per-prototype instance-index selection and visibility mask
the importer applied — through the same `readInstancerPlacements()` the importer
calls, so the two cannot drift — and writes through `Array::setData()`.

A TSD `Array` has no resize, so an element count that moves mid-sequence
allocates a right-sized Array and rebinds: `setAsTransformArray` for an
instancer, a parameter rebind for geometry. That costs nothing on the common
path and pays handle churn only on the frames where the count actually moves.
Detecting constant counts at import time was rejected: proving it means reading
every sample — all 1.6 GB of `apic_fluid` — and sampling first/mid/last is a
heuristic that would have passed on all three motivating scenes while still
being wrong in general.

A mesh whose vertex count and topology both move is handled by construction,
which is the main thing the split buys. Points, indices and primvars are one
consistent set: they come out of a single resolve and go in through a single
refill, so the Geometry is never left describing half of one frame and half of
another. Re-pulling only points — which is what a binding without the split can
do — would have described a mesh that never existed.

What a binding still cannot do is change how many Parts a prim has. Parts appear
and disappear when a mesh's material subsets change, and that means new Surfaces
and new Materials — conversion, not animation. The binding writes the Parts it
still recognizes, warns once, and leaves the rest as imported. It also replays
rather than recomputes everything the import decided that does not vary with
time: which primvar each Part's material reads as texture coordinates (which is
what assigns every other primvar its attribute slot), whether the mesh refines,
and what transform is baked in. Re-deriving those would mean resolving materials
again, which is exactly the object churn this ADR exists to avoid.

Because a scrub writes one Array per animated instancer and
`RenderIndexAllLayers` rebuilds its world on every `ANARI_FLOAT32_MAT4` array
unmap, batching is not optional: `AnimationManager` brackets a time change in
`Scene::beginUpdateBatch()`/`endUpdateBatch()`, and a render index coalesces the
rebuilds it owes until the batch ends. The existing `beginLayerEditBatch` could
not be reused — it batches *structural* layer changes, and an array unmap is not
one. Without the new bracket, a stage with several animated instancers would pay
a full world rebuild per instancer per frame. None of the three motivating
scenes would have caught that, since each has exactly one.
