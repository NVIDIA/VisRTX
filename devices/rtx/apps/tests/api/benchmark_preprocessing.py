#!/usr/bin/env python3
"""Compare local before/after geometry-preparation latency (not a CI assertion).

Run one harness against two ABI-compatible device libraries. Only the device
library path is injected; the environment must resolve a real CUDA driver.
Each trial is a fresh process; iteration zero is excluded from warm statistics.
Optional JSON output is local scratch, not a versioned performance baseline.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("executable")
p.add_argument("base_library_dir", help="before/control device library directory")
p.add_argument("branch_library_dir", help="after/candidate device library directory")
p.add_argument("output", type=Path, nargs="?", help="optional local JSON results path")
p.add_argument("--trials", type=int, default=3)
p.add_argument("--repeats", type=int, default=21)
p.add_argument("--geometry", default="cylinder",
               choices=["triangle", "quad", "sphere", "curve", "cone", "cylinder", "sdf"])
p.add_argument("--device", action="store_true",
               help="use device-backed inputs; both libraries must support them")
p.add_argument("--case", action="append", default=[], metavar="NAME",
               help="run only the named default case; repeatable")
p.add_argument("--extra-case", action="append", default=[],
               metavar="NAME:COUNT:GEOMETRIES:MODE:FRAMES",
               help="append a custom case; repeatable")
a = p.parse_args()
if a.trials < 2 or a.repeats < 3:
    p.error("need at least two trials and three repetitions")
# name, primitive count per geometry, distinct geometries, update mode, frames
cases = [
    ("small-recommit", 32, 64, "recommit", 0),
    ("small-replace", 32, 64, "replace", 0),
    ("large-replace", 262144, 1, "replace", 0),
    ("unchanged-frame", 32, 64, "unchanged", 1),
    ("two-frames-replace", 32, 64, "replace", 2),
]
if a.case:
    known = {c[0] for c in cases}
    unknown = [n for n in a.case if n not in known]
    if unknown:
        p.error(f"unknown case(s) {unknown}; known: {sorted(known)}")
    cases = [c for c in cases if c[0] in a.case]
for spec in a.extra_case:
    try:
        name, count, geometries, mode, frames = spec.split(":")
        case = (name, int(count), int(geometries), mode, int(frames))
        if (not name or case[1] < 1 or case[2] < 1 or case[4] < 0
                or mode not in {"recommit", "replace", "unchanged", "update"}
                or (mode == "update" and (a.device or a.geometry == "sdf"))
                or name in {c[0] for c in cases}):
            raise ValueError()
        cases.append(case)
    except ValueError:
        p.error(f"bad --extra-case {spec!r}; use a unique NAME, positive COUNT/GEOMETRIES, "
                "recommit|replace|unchanged|update, and nonnegative FRAMES; "
                "update requires host inputs and non-SDF geometry")
records = []


def save():
    if a.output:
        a.output.write_text(json.dumps(records, indent=2) + "\n")


print("Warm latency (ms): median of trial medians [min..max trial median]. "
      "After/before < 1 is faster; ranges are not confidence intervals.", flush=True)
for name, count, geometries, mode, frames in cases:
    for trial in range(a.trials):
        order = ["base", "branch"] if trial % 2 == 0 else ["branch", "base"]
        for revision in order:
            env = os.environ.copy()
            lib = a.base_library_dir if revision == "base" else a.branch_library_dir
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                filter(None, [lib, env.get("LD_LIBRARY_PATH")]))
            command = [a.executable, a.geometry, str(count), str(int(a.device)),
                       str(a.repeats), "1", str(geometries), mode, str(frames), "1"]
            run = subprocess.run(command, env=env, capture_output=True, text=True)
            samples = [(int(i), float(ms)) for i, ms in re.findall(
                r"sample iteration=(\d+) ms=([\d.]+)", run.stdout)]
            record = dict(case=name, trial=trial, revision=revision,
                          command=command, library_dir=lib, returncode=run.returncode,
                          stdout=run.stdout, stderr=run.stderr)
            records.append(record)
            save()
            if (run.returncode or len(samples) != a.repeats
                    or {i for i, ms in samples} != set(range(a.repeats))
                    or any(not math.isfinite(ms) or ms < 0 for i, ms in samples)):
                raise SystemExit(f"failed {name} {revision} (exit {run.returncode}):\n"
                                 f"{run.stderr}\n{run.stdout}")
            record["warm_median_ms"] = statistics.median(ms for i, ms in samples if i)
            save()
    medians = {rev: [r["warm_median_ms"] for r in records
                     if r["case"] == name and r["revision"] == rev]
               for rev in ["base", "branch"]}
    before, after = (statistics.median(medians[rev]) for rev in ["base", "branch"])
    ratio = f"{after / before:.3f}" if before else "n/a"
    ranges = [f"{statistics.median(values):.4f} [{min(values):.4f}..{max(values):.4f}]"
              for values in medians.values()]
    print(f"{name}: before={ranges[0]} after={ranges[1]} after/before={ratio}", flush=True)
