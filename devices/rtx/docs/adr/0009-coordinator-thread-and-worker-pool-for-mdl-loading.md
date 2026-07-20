# Coordinator thread and worker pool for MDL loading

MDL material loading — module load, class compilation, PTX generation, texture
decode, and OptiX module build — runs on a dedicated **Coordinator** thread that
owns a **worker pool**, instead of serialized inline on the ANARI commit-flush
thread. The goal is scene-load throughput: the LLVM-heavy `translate_link_unit`,
the CPU texture decode, and the OptiX module build of many materials run in
parallel. `finalize` still blocks on each material's result, so ANARI sync
semantics are unchanged — this is a pure speedup, not an async/latency change.
Measured: 64 compile-heavy materials compile 7.1× faster at 8 threads than
serial.

## The model

- **Coordinator** — one dedicated thread, device-lifetime owner of the
  `MaterialRegistry`. It is the *only* thread that mutates registry state
  (`m_targetCodes`, the uuid/name caches, refcounts, the update timestamp), so
  dedup and slot assignment are lock-free by confinement rather than by mutex.
  Both `acquire` and `release` are posted to it. Slots are assigned as each
  worker's compile completes, so the slot index (the dispatch-time
  `callableBaseIndex`) follows compile-completion order and is not stable across
  runs. That is intentional: consumers look a material up by uuid, and the PTX
  fingerprint XORs slots (order-independent), so nothing depends on slot order.
- **Module Preload** — the Coordinator resolves, loads, and *commits* every
  module a request needs before dispatching any compile. Committing up front
  makes modules globally visible, so parallel worker transactions only ever
  *read* shared modules. This is what makes concurrent compilation sound: no
  two workers race to load the same overlapping import into conflicting
  uncommitted transactions.
- **Worker pool** — each compile worker opens its own transaction *after* the
  module commit, re-accesses DB elements by name, and runs
  `create_compiled_material` + `translate_link_unit` (own fresh backend, own
  cloned execution context), producing a transaction-independent result. The
  registry insert (slot assignment, PTX blob, emission IR) is posted back to the
  Coordinator; the request's promise carries only the compiled-material uuid and
  the argument-block descriptor. `finalize` blocks on that promise, turning a
  worker exception into a failed result rather than propagating it (the
  default-material fallback then runs).
- **Split-phase over the two-pass flush.** helium flushes all `commitParameters`
  before all `finalize`. `commitParameters` starts the compile and holds the
  future; `finalize` collects it. So a whole flush of materials is compiling in
  parallel before the first `finalize` blocks.
- **Texture decode fans out too.** A single material can reference many texture
  files. `finalize` submits each material's stb image decodes to the pool and
  waits on the commit thread (never on a pool worker, so a task never waits on
  its own pool); decoded pixels are staged and the commit thread creates the
  samplers. BSDF-data and `.dds` textures are handed to the device as-is. The
  batch skips already-cached textures and de-duplicates keys, so a texture
  referenced N times is decoded once.
- **OptiX module build fans out.** The renderer creates each material's OptiX
  module on the pool (`optixModuleCreate` on one context is thread-safe); the
  SBT is then assembled serially, so callable order — and every material's SBT
  offset — is exactly the serial layout.
- **Registry confinement is split, and asymmetric.** The `MaterialRegistry` is
  Coordinator-confined (lock-free by construction). The `SamplerRegistry` is
  *not*: `acquireSampler` creates helium objects + uploads to CUDA on the commit
  thread, but `releaseSampler` runs whenever a material's last reference drops —
  any app thread under `khr_device_synchronization` — so its cache carries its
  own mutex. Confinement dissolves the MaterialRegistry races; a targeted lock
  handles the one path (sampler release) that confinement cannot reach.

## The accepted risk: undocumented SDK concurrency

Beneath the per-worker backends, the MDL SDK shares one `IMDL` compiler and one
JIT `ICode_cache` per `INeuray`. The SDK documents multiple parallel
*transactions* (since 2023.1.0) but says **nothing** about concurrent
`translate_link_unit` / `create_compiled_material` on one instance. We are
betting these tolerate concurrent access. (The shared entity resolver is *not*
part of this bet — module load and texture-URL resolution are serialized on the
Coordinator precisely because concurrent resolver access is also undocumented.)
The failure mode of the bet is not a crash — it is *silently miscompiled PTX*,
which ThreadSanitizer cannot detect (the SDK is an opaque binary we suppress).

Two things make the bet defensible rather than reckless:

1. **A PTX-byte-identity gate.** The device exposes an `mdlPtxFingerprint`
   property — an order-independent hash of every compiled material's PTX — and a
   test (`TestMdlPtxIdentity`) renders the same scene under
   `VISRTX_MDL_COMPILE_THREADS=1` and `=8` and asserts the fingerprints match.
   This is the one check that catches a code-cache corruption; image parity and
   TSAN cannot. It passes on the current SDK.
2. **A fallback to serial compilation.** Setting `VISRTX_MDL_COMPILE_THREADS=1`
   forces one worker, so compilation runs one material at a time and the shared
   SDK state is never touched concurrently. If the field ever shows corruption
   the parallel path can be switched off outright without a redesign. (A finer
   Coordinator-held mutex around only the shared `translate` — "decode-only
   parallelism" — was considered but not built; the identity gate plus this
   escape hatch were judged sufficient.)

## Considered options

- **Pervasive registry locking (rejected).** The first design locked
  `load_module` only and ran everything else on parallel transactions under
  registry mutexes. Two review panels found it unsound: a function-static
  target-function array raced, the entity resolver and shared scope raced, and
  the registry critical sections were a TOCTOU minefield. Thread-confining the
  registry to one Coordinator dissolves that whole class instead of guarding
  each site with a lock.
- **Single-thread confinement of the whole SDK (rejected).** Putting *all* MDL
  work on one thread removes every concurrency question but delivers no compile
  throughput — the LLVM stage, the dominant cost, stays serial. The
  Coordinator+Pool split keeps that stage parallel.
- **Decode-only parallelism by default (rejected).** The correct-by-default
  posture serializes compilation until NVIDIA documents thread-safety,
  parallelizing only texture decode. Rejected in favor of the bet: full compile
  parallelism now, gated by PTX-identity and reversible via
  `VISRTX_MDL_COMPILE_THREADS=1`. The larger win was judged worth the managed
  risk.

## Consequences

- Correctness rests on an undocumented SDK property. The PTX-identity gate and
  the serial-compilation fallback (`VISRTX_MDL_COMPILE_THREADS=1`) are
  load-bearing, not optional test niceties — a future SDK bump must re-run the
  gate before trusting parallelism.
- Module load and texture-URL resolution stay serial on the Coordinator (both
  touch shared, un-preloadable SDK state), so import-parse-bound scenes are
  Amdahl-capped; class compilation, PTX generation, texture decode, and OptiX
  module build are what parallelize.
- Teardown is ordered by construction: the Coordinator (with its worker pool) is
  declared last in the MDL state struct, so it is destroyed — and its threads
  drained and joined — before the registry and Core it serializes. The workers
  are joined before the Coordinator thread (a worker's bookkeeping reaches back
  into it), and a `run()`/`submit()` issued after stop executes inline, so a
  material released during `commitBuffer.clear()` never deadlocks on a
  thread that is already gone.
- `finalize` blocking on a promise cannot hang or crash the commit thread: a
  worker exception is caught and turned into a failed compile that falls back to
  the default material.
- `finalize` semantics are unchanged for applications: a committed material is
  ready when `finalize` returns, exactly as before.
