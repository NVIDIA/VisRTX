---
name: benchmark-geometry-preprocessing
description: Compare local before/after performance for RTX geometry preparation, preprocessing, BVH readiness or preparation-preserving updates. Use when assessing whether a development change improves end-to-end latency.
---

# Local geometry-preprocessing comparison

Produce a terminal before/after comparison and a short assessment of whether the
change helps its intended workload. Results are local development feedback;
historical performance tracking is a separate CI concern. Run affected correctness
tests independently of timing.

Repository paths below are relative to the checkout root. Timing definitions,
custom workload examples and optional profiling commands live in
`devices/rtx/docs/geometry-preprocessing-performance.md`; read it before choosing
workloads or interpreting numbers.

## 1. Identify before and after

Use the user's control revision; ask if unspecified. Keep the candidate intact
and create a separate control workspace using the checkout's VCS (`jj workspace
add` in a jj workspace, otherwise `git worktree add`). Record the control revision
and candidate revision/working-copy state in the local report.

## 2. Build matched libraries

Use identical compiler, CUDA, OptiX, architecture, build configuration, optional
components and library suffix on both sides. Prefer the host's existing build
recipe. A minimal single-configuration starting point, with dependencies already
available, is:

```bash
cmake -S SOURCE -B BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVISRTX_BUILD_GL_DEVICE=OFF -DVISRTX_ENABLE_MDL_SUPPORT=OFF \
  -DVISRTX_ENABLE_MATERIALX_SUPPORT=OFF -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
```

Enable `VISRTX_BUILD_TESTS` in the candidate and build
`BenchmarkGeometryPreprocessing`; build `anari_library_visrtx` in the control.
One candidate executable runs against either ABI-compatible library, so the
control need not contain benchmark sources. If their ABIs differ, establish a
compatible harness/build pair before comparing.

Locate the executable and directories containing the built shared libraries.
For multi-configuration builds include the configuration directory. Check both
loader resolutions, substituting each library directory:

```bash
LD_LIBRARY_PATH=LIBRARY_DIR ldd BENCHMARK
```

Proceed only when each resolves the intended device library and a real CUDA
driver, not a toolkit link-time stub. Keep the remaining runtime environment the
same on both sides.

## 3. Compare

Start with the default cases, or a focused case when the user requests one:

```bash
python3 devices/rtx/apps/tests/api/benchmark_preprocessing.py \
  BENCHMARK CONTROL_LIBRARY_DIR CANDIDATE_LIBRARY_DIR
```

Use `--help` for workload selection, geometry, device inputs and run budget.
The tool prints warm medians, trial ranges and candidate/control ratios.
A trailing JSON path before the options retains raw results in local scratch;
no output file is required. A nonzero exit or incomplete run is a failed
comparison. Resolve the failure rather than reporting its partial timing.

## 4. Assess and stop

Report the selected workloads, before/after medians, trial ranges and ratios.
Describe what improved, regressed or remains inconclusive, and the geometry/input
and rendering paths actually measured. Overlapping ranges warrant a focused
rerun with more trials if the distinction matters; they neither prove equality
nor establish a statistically significant effect. Avoid universal percentage
thresholds for noise on an uncontrolled host.

For an apparent regression, rerun that case before drawing a conclusion. For a
claimed work-elimination speedup, use the optional trace recipe to check work
counts. Pair synchronization or memory-lifetime changes with relevant failure
and retirement tests; use Compute Sanitizer when those risks warrant it.

Finish with a concise local assessment and any remaining uncertainty. Keep raw
results and captures outside the repository; this workflow does not create
versioned baselines, performance-history tables or CI acceptance thresholds.
