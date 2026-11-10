# Camera and transport sampling

VisRTX uses a bounded hybrid: **Halton for the camera, Sobol for fixed-size
surface transport decisions, and PCG for variable-length or fallback work**.
The purpose is predictable sample addressing and low-discrepancy surface
sampling without forcing variable-length tracking loops into a fixed dimension
budget. No universal convergence improvement or statistical independence of
hashed streams is claimed.

## Camera: Halton

Pixel jitter uses bases 2/3 and aperture uses bases 5/7. The camera index is the
completed-sample ordinal plus a per-pixel hash, with uint32 wrapping. Camera and
transport share an ordinal but not their sampling sequence or dimension space.

Both progressive raygen loops use `completedSamples + iteration`; the frame
counter already counts samples, not launches. Positive `pixelSamples` therefore
uses a contiguous prefix across batches. Checkerboarding uses effective spp=1
and advances the ordinal once per four-tile cycle. The existing signed frame
counter must not overflow; this is not an unlimited accumulation contract.

The first accumulated sample centers pixel jitter but still samples aperture.
Ray-buffer cameras bypass jitter and aperture. These are per-pixel integration
samples, not frame-wide reconstruction jitter or temporal-upscaler inputs.
Changing batching does not promise byte-identical general rendered images:
PCG initialization and floating-point accumulation can also affect results.

## Surface transport: Sobol

Quality uses the binary sample ordinal with pbrt-style **FastOwenScrambler**.
This is a hash-based fast approximation, not pbrt's separate OwenScrambler or a
proof of independent nested random permutations. A pixel key is mixed with
bounce and cumulative transparency depth before each surface vertex's draws.

| Sobol dimensions (zero-based) | Role |
| --- | --- |
| 0–1 | Ambient hemisphere and HDRI cosine strategy |
| 2–3 | Analytic-light coordinates or HDRI CDF row/column |
| 4–5 | HDRI within-texel jitter |
| 6–7 | Light category and instance selection |
| 8–11 | First four sequential material requests |

The raw table supports 16 dimensions; 12–15 are unassigned capacity, not camera
reservations. Distinct dimensions or keys do not prove independence. Ambient
and HDRI cosine terms can intentionally use the same direction at a vertex;
that correlates their estimates but does not by itself establish biased MIS.
The 32-bit depth-key mixer does not promise collision-free arbitrary tuples.

`sobolUInt` evaluates the binary index directly, not a Gray-code iterator index.
The float conversion is exactly `(scrambled >> 8) * 2^-24`: a 24-bit grid in
`[0,1)`, including zero and excluding one. There is no global clamp or implicit
open-interval conversion for exponential sampling.

`owenPixelSeed` accepts an internal optional scramble seed. Zero preserves the
pixel-key hash; nonzero seeds provide reproducible sampler experiments. Renderer
callers use zero. This is not public full-render replicate control: that would
also need a PCG policy and accumulation-reset semantics.

## PCG and material consumption

Volume free flights, volume light/phase sampling, ratio-tracking shadows,
geometry-light coordinates, surface roulette and coverage rejection remain on
PCG. Interactive's material and light paths retain their PCG defaults.

`NextRayQmc` supplies up to four uniforms. Native material lobe choices consume
this same branch-dependent sequence; it is not a fixed semantic role assignment
across every material configuration. Draws are ordered and remain inside their
original branches and early exits. A fifth request falls back to PCG.

A non-null overlay requires signed `0 <= n <= 4` and `0 <= cursor <= n`.
Supplied requests advance the cursor without touching PCG. Exhausted requests
advance PCG once and leave the cursor at n. A null overlay requires cursor=0,
advances PCG once per request and leaves the cursor unchanged. PCG state must be
valid whenever fallback is reached. Unsupported dimensions and invalid overlay
bounds abort on the host or trap on CUDA, including NDEBUG builds; this is a
programmer-error contract, not a recoverable frame-failure mechanism.

## Shared implementation and test coverage

Native continuation sampling and its required pure helpers live in
`material/shaders/physicallyBasedSampling.h`. The OptiX callable delegates to
`samplePhysicallyBased`; the host branch tests call that same implementation.
The existing PDF helper is shared with sampling, while parameter initialization,
texture/normal-map access and the NEE evaluation callable remain in the shader.
Shared basis/hemisphere math lives in `gpu/samplingMath.h`.

MDL assigns `sample_data.xi` using `materialUniform4` in `gpu/shadingState.h`.
This helper orders four fixed component requests; native PBR does not eagerly
use it because its draw consumption depends on branches and early exits.
No C++ source-text extraction or generated shader-expression header is used.
The shared functions compile as host inline functions with the C++ compiler and
as device functions with NVCC through the existing `VISRTX_DEVICE` macro.
The tests still require CUDA/OptiX headers; host-compilable does not mean
CUDA-toolkit-free.

| Test | Observable contract |
| --- | --- |
| `rtx::Sobol`, `rtx::SobolCUDA` | Independent direction recurrence, integer/grid vectors, dyadic 1D strata, seed vectors, Halton rational reference, host/device behavior |
| `rtx::SobolTableHeader` | Offline complete-header byte identity, including notices and guard |
| `rtx::SobolReferencePublication` | Validation, input-read and staging failures preserve existing fixtures |
| `rtx::SamplerContractsAssert`, `rtx::SamplerContractsNDEBUG` | Complete valid/invalid overlay matrix, exact PCG state and host SIGABRT |
| `rtx::SamplerContractsCUDAAssert`, `rtx::SamplerContractsCUDANDEBUG` | Same production contracts and isolated device failure outcomes |
| `rtx::DrawOrder` | Native directions, branch-local fallback consumption, early exits, and ordered full/mixed MDL components through shared functions |
| `rtx::LightDrawOrder` | Real rectangle/HDRI/hemisphere sampling and PCG consumption on CUDA |
| `rtx::SampleOrdinal`, `TestSampleAddressing` | Ordinal prefix, batching/reset/checkerboarding and actual camera mapping, including perspective aperture |
| `TestMaterialSampling` | Quality/Interactive callable integration for Matte, native PBR and available MDL materials |

The exhaustive n/cursor matrix belongs in SamplerContracts, not each material
test. Light ordering is tested through the real CUDA samplers rather than
repeated host-extracted coordinate expressions. This does not independently
qualify those light expressions under a second compiler. Native host tests use
asymmetric supplied values and independent analytic directions; CUDA builds and
API tests remain necessary to verify device compilation and callable wiring.

The negative-process harness is UNIX-only. Each CUDA child first establishes a
successful ordinary launch, then requires an assert/illegal-instruction/launch-
failure synchronization error and normal expected-error exit. Unrelated errors,
crashes and timeouts fail. Each child owns a disposable context; none is reused
after a trap. This is not Windows failure-policy qualification.

## Reference provenance

- [pbrt-v4 direction matrices](https://github.com/mmp/pbrt-v4/blob/b0bdf461ba5dd6504b70ebf0932acc0fe7aa744f/src/pbrt/util/sobolmatrices.cpp):
  rows 0..15 and columns 0..31, with source row stride **52**.
- [Joe–Kuo primitive parameters](https://github.com/joe-kuo/sobol_data/blob/835a7d7b1ee3bc83e575e302a985c66ec4b65249/new-joe-kuo-6.21201):
  the unchanged header and one-based dimension rows 2..16, with attribution.
- [pbrt-v4 FastOwenScrambler](https://github.com/mmp/pbrt-v4/blob/b0bdf461ba5dd6504b70ebf0932acc0fe7aa744f/src/pbrt/util/lowdiscrepancy.h):
  retained integer operations and a vendored reference excerpt.

`LICENSE.pbrt`, `LICENSE.joe-kuo` and `NOTICE` retain source attribution and are
installed for binary distributions even when tests are disabled. There is no
pbrt runtime dependency and ordinary tests need no reference downloads.

SHA-256 digests:

- Table: exactly 512 dimension-major/column-minor uint32 words serialized
  little-endian, without header:
  `357f257f40c9f9b7efb784bdf355b543f456984987e8519686fc978cf319be96`.
- Exact attributed parameter fixture bytes:
  `d155655216d8f60bac3d2b34f2926cf03b64e288c7df74d6c25c13cccce3b26f`.
- `vectors.txt`:
  `a25db35646e5f5efc17500bff4f6d957102d2c5ea78cd3baee8813919eb779f9`.

The parameter oracle regenerates columns independently of production data.
Dimension zero uses `1u << (31-i)`; other rows use their primitive-polynomial
recurrence. Raw Sobol tests use binary indexing. Halton expectations use integer
digit reversal and a double rational division, allowing absolute error `2^-21`
for float reciprocal/FMA differences; Halton CPU/CUDA bit identity is not
required. Sobol integer/grid comparisons require exact agreement.

## Regeneration

Download pinned inputs independently, then invoke the generator from the
workspace root. Source hashes are verified before processing:

```sh
pbrt=b0bdf461ba5dd6504b70ebf0932acc0fe7aa744f
joe=835a7d7b1ee3bc83e575e302a985c66ec4b65249
curl -fL https://raw.githubusercontent.com/mmp/pbrt-v4/$pbrt/src/pbrt/util/sobolmatrices.cpp -o /tmp/sobol-matrices.cpp
curl -fL https://raw.githubusercontent.com/mmp/pbrt-v4/$pbrt/src/pbrt/util/lowdiscrepancy.h -o /tmp/sobol-lowdiscrepancy.h
curl -fL https://raw.githubusercontent.com/mmp/pbrt-v4/$pbrt/LICENSE.txt -o /tmp/sobol-pbrt-license
curl -fL https://raw.githubusercontent.com/joe-kuo/sobol_data/$joe/new-joe-kuo-6.21201 -o /tmp/sobol-parameters
curl -fL https://raw.githubusercontent.com/joe-kuo/sobol_data/$joe/LICENSE -o /tmp/sobol-joe-license
python3 -B devices/rtx/apps/tests/unit/sobol/reference.py \
  /tmp/sobol-matrices.cpp /tmp/sobol-parameters \
  --fast-owen-source /tmp/sobol-lowdiscrepancy.h \
  --pbrt-license /tmp/sobol-pbrt-license --joe-license /tmp/sobol-joe-license \
  --table devices/rtx/device/gpu/sobolDirections.h
python3 -B devices/rtx/apps/tests/unit/sobol/seed_reference.py
```

Regeneration prepares all requested outputs and checks oracle agreement before
publishing any files. It stages all outputs before atomic per-file replacements,
preserving existing files on validation, input-read or staging failure. This is
not a multi-file filesystem transaction: a replacement failure midway through
publication can still require recovery. A disagreement must be investigated,
not resolved by modifying the oracle to match production.

## Running the tests

Enable `VISRTX_BUILD_TESTS=ON`; Python 3 is required for offline fixture tests.
Use native-PBR and MDL-backed-PBR configurations, rebuild affected PTX, and
ensure the tests load that build's device library rather than an installed copy.
For a single-config native build, for example:

```sh
cmake -S . -B _out/native -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_BUILD_TESTS=ON \
  -DVISRTX_ENABLE_MDL_SUPPORT=OFF
cmake --build _out/native --parallel
LD_LIBRARY_PATH="$PWD/_out/native:${LD_LIBRARY_PATH:-}" \
  ctest --test-dir _out/native --output-on-failure \
  -R '^(rtx::(Sobol.*|SamplerContracts.*|DrawOrder|LightDrawOrder|SampleOrdinal)|TestSampleAddressing|TestMaterialSampling)$'
```

For MDL-backed PBR enable both `VISRTX_ENABLE_MDL_SUPPORT` and
`VISRTX_USE_MDL_FOR_PHYSICALLY_BASED`, then also run `TestEmissiveMaterialParity`.
For multi-config builds select the configuration with `--config` / `ctest -C`
and put that configuration's library directory on the loader path.

These are correctness and integration checks, not convergence/timing claims.
Variable-length volume sampling, geometry-light QMC, public replicate controls,
non-positive pixelSamples, CDF zero-mass ties and signed frame-counter overflow
are outside this implementation's guarantees.
