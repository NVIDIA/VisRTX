# Local geometry-preprocessing benchmarks

Use this tool to assess whether a development change improves end-to-end latency
on your machine. It is a manual before/after helper, not a performance regression
CTest or a historical baseline. Performance evolution tracking belongs in a
separate controlled CI job; this tool does not establish that infrastructure.

## Quick comparison

Build `BenchmarkGeometryPreprocessing` in the candidate and `anari_library_visrtx`
in a separate control workspace, with matching compiler, CUDA, OptiX, architecture,
optional-component settings and library suffix. The same benchmark executable
loads either ABI-compatible device library. The control needs no benchmark code.
Use the project [benchmark skill](../../../.agents/skills/benchmark-geometry-preprocessing/SKILL.md)
for workspace/build preparation and interpretation.

```bash
python3 devices/rtx/apps/tests/api/benchmark_preprocessing.py \
  CANDIDATE_BUILD/BenchmarkGeometryPreprocessing \
  CONTROL_LIBRARY_DIR CANDIDATE_LIBRARY_DIR
```

Pass the directories **containing the shared libraries**, not an install prefix.
For a multi-configuration build these may be `BUILD/RelWithDebInfo`. Verify with
`LD_LIBRARY_PATH=LIBRARY_DIR ldd BENCHMARK` that each side loads the intended
library and a real CUDA driver, rather than a toolkit link-time driver stub.

The runner prints before/after warm medians, trial-median ranges, and ratios.
Defaults cover small recommits, small/large input replacement, unchanged rendering,
and two completed frames after replacement. It alternates order across fresh
process trials. Run `--help` for controls. Examples:

```bash
# Focus on a preparation-reuse change.
python3 devices/rtx/apps/tests/api/benchmark_preprocessing.py \
  BENCHMARK BEFORE_LIB AFTER_LIB --case small-recommit

# Include a completed frame and use another geometry family.
python3 devices/rtx/apps/tests/api/benchmark_preprocessing.py \
  BENCHMARK BEFORE_LIB AFTER_LIB --case unchanged-frame \
  --extra-case reuse-render:32:64:recommit:1 --geometry triangle

# Longer comparison with optional local raw output.
python3 devices/rtx/apps/tests/api/benchmark_preprocessing.py \
  BENCHMARK BEFORE_LIB AFTER_LIB /tmp/preprocessing-comparison.json \
  --trials 6 --repeats 101
```

The optional JSON contains commands, library paths, stdout/stderr and trial
medians. It is written incrementally: a failed command leaves partial results,
not a successful comparison. Keep results and profiler captures in local scratch.
Record revision IDs and environment alongside local results when needed to
interpret them; do not add one-off evidence files or result tables to the repo.

## Workload and timing contract

The executable can also run one workload directly:

```text
BenchmarkGeometryPreprocessing TYPE COUNT DEVICE REPEATS \
  [SHARED_SURFACES] [DISTINCT_GEOMETRIES] \
  [recommit|replace|unchanged|overlap] [FRAMES:0=bounds] [EMISSIVE:0|1]
```

- COUNT is primitives **per geometry**. The paired runner uses emissive geometry,
  one surface per geometry and host inputs by default. `--device` requires device
  input support on both revisions; unsupported input paths are not comparisons.
- `recommit` changes a geometry-wide attribute; `replace` replaces inputs, including
  allocation, host staging, application H2D copy and parameter installation in
  the measured interval; `unchanged` retains the initial inputs.
- FRAMES=0 ends at completed bounds/BVH readiness. FRAMES>0 includes submission
  and completion of all frames, registry publication and tracing, then bounds.
  Multiple frames share one device/world; this is not a concurrency benchmark.
- `overlap` requires FRAMES=1 and uses a 1024x1024 frame with 32 pixel samples.
  It submits rendering, commits geometry-wide attribute updates, then queries bounds before
  waiting for the frame. `overlap ... query_ms` measures the bounds query alone;
  `sample ... ms` still includes completed rendering. This isolates CPU blocking
  on an outstanding frame, not a rendering-throughput improvement. Run it
  directly (the paired runner does not summarize query latency):
  `BENCHMARK cylinder 32 0 11 1 64 overlap 1 1`.
- Initial setup is outside each iteration's timer. `setup_to_ready_ms` also
  includes fixture generation and initial input setup, but not device creation.
  Iteration zero is reported separately and excluded from warm statistics.
- The runner reports the median of per-process warm medians and their range.
  Ranges are not confidence intervals. Thermal state, clocks and system load
  affect results; rerun suspicious cases before attributing a delta to code.
- DEVICE=1 stages host fixtures to the GPU. It does not model an upstream GPU
  producer or prove that real GPU-produced input needs an H2D copy.
- The harness checks finite bounds/images, sample counts and frame completion
  outside the timed interval. These guard against skipped work, not radiometric
  equivalence. Run the affected correctness tests separately.

## Optional investigation

Profile only when timing leaves an explanatory question:

```bash
nsys profile --trace=cuda --sample=none -o /tmp/preprocessing-trace \
  env LD_LIBRARY_PATH=LIBRARY_DIR \
  BENCHMARK cylinder 32 0 6 1 64 recommit 1 1
nsys stats --report cuda_api_sum,cuda_gpu_mem_size_sum /tmp/preprocessing-trace.nsys-rep
```

Compare work counts when claiming work elimination. Profiled whole-process totals
include initialization and teardown and are not comparable to unprofiled warm
latencies. `BenchmarkCudaOrdering` is an optional stream-ordering probe, not an
end-to-end timing or correctness test. Synchronization changes must still protect
input retirement, host readbacks and publication; timing cannot establish safety.
