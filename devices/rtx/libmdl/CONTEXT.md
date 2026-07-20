# MDL

Compiles MDL material definitions into GPU code and editable argument data.
Standalone from the device; `device/mdl/` is a consumer.

## Language

**MDL Material Definition**:
A material as authored in MDL source (a `.mdl` module).
_Avoid_: material (alone), shader

**Compiled Material**:
An MDL Material Definition compiled with a specific parameter set; the input
to code generation.

**Target Code**:
The generated PTX implementing a Compiled Material's functions on the GPU.

**Class Compilation**:
The compilation mode where parameter values stay editable through an Argument
Block, so edits do not require recompiling the material.

**Argument Block**:
The GPU blob holding a Compiled Material's runtime-editable parameter values.
A descriptor defines its layout; an instance holds actual values.

**Texture Runtime**:
The texture access machinery Target Code calls at shading time.
_Avoid_: sampler (see the Frontend context)

**Emission Classifier**:
The MDL-pure analysis that lowers a Compiled Material's emission expressions
into Emission IR and folds them into an Emission Descriptor. It describes;
it never decides light registration (that is the renderer-side policy, see
ADR-0007).
_Avoid_: deciding registration in the classifier

**Emission IR**:
The owned lowering of the MDL emission expression DAG (constants, parameters,
calls, textures) the classifier folds over. Retains no MDL-SDK expression
pointers.

**Emission Descriptor**:
The immutable per-material output of the Emission Classifier: per-slot
`{verdict, edfKinds, magnitude, intensity mode}` plus the argument/resource
dependencies the emission reads.

**Faithful Set**:
The consumer-exported set of EDF kinds a renderer can evaluate faithfully on
its synthetic next-event hit (`kFaithfulSet`). A described slot registers as a
Geometry Light only when its EDF kinds are a subset of it.

**Compile Job**:
One worker-pool task producing a Target Code from an already-preloaded module:
own transaction opened *after* the Coordinator commits the module, own cloned
execution context, `create_compiled_material` through PTX generation. Module
load, resolution, and registry insert are the Coordinator's, not the job's.
_Avoid_: loading modules or mutating the registry inside a job

**Uuid Dedup**:
Two levels of de-duplication for compiled materials. The name cache maps a full
material name to a compiled-material Uuid (a cheap hit that skips compilation
entirely). The Uuid cache maps that Uuid to a registry slot: two *different*
requests whose compiled materials hash to the same Uuid share one slot. Two
concurrent requests for the same uncached material both compile (redundant
work), then collapse to one slot when `insertCompiled` dedups by Uuid — the
result is correct, the refcount accounts for every waiter.

**Coordinator**:
The single dedicated thread that owns the MaterialRegistry, and the request
queue, for their whole device lifetime. Being single-threaded, it mutates all
cache/dedup/slot/refcount state lock-free and by construction. It also owns the
serial Module Preload, then dispatches Compile Jobs to the worker pool. The
front door that hides the pool's concurrency. (The SamplerRegistry cache is
guarded by its own mutex instead, because sampler release runs on arbitrary
app threads.)

**Module Preload**:
The serial Coordinator phase that resolves, loads, and *commits* every module
a request needs before any Compile Job runs. Committing up front makes modules
globally visible, so parallel worker transactions only read shared modules —
never re-load overlapping imports into conflicting uncommitted transactions.
Replaces the discarded per-worker load-and-lock idea.

**Compile Backend**:
The per-Compile-Job `IMdl_backend` (`get_backend(MB_CUDA_PTX)` news a fresh
instance each call), carrying its own options and link unit. Options are
per-instance, so PTX generation parallelizes; the residual shared state is the
`IMDL` compiler and JIT `ICode_cache` beneath all backends — the real
concurrency surface to validate.

**Staged Decode**:
A texture decoded to a host buffer on a worker thread (tail of a Compile Job),
consumed by the commit-thread sampler acquire at finalize. A missing stage
falls back to inline decode; a wasted stage is never an error.
_Avoid_: uploading to CUDA or creating helium objects off the commit thread
