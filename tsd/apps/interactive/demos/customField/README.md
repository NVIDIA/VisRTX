# Custom Spatial Field Demo

This demo showcases the **custom spatial field** extension mechanism in VisRTX,
using a *weighted points field* as a concrete example. It demonstrates how
external applications can define entirely new volume representations without
modifying the core rendering engine.

## Custom Fields in VisRTX

The ANARI specification defines a fixed set of spatial field subtypes
(`structuredRegular`, `amr`, etc.). VisRTX extends this through a
**registration-based plugin system** that allows new field types to be added at
build time:

1. **Define the GPU data**: A plain struct (`WeightedPointsFieldData`) that
   fits within the 256-byte `CustomData::fieldData` buffer. This struct is
   uploaded to the GPU and made available to the OptiX sampler.

2. **Implement the host object**: A class derived from `CustomField` that
   handles parameter parsing (`commitParameters`), host-to-device data transfer
   (`finalize`), and spatial metadata (`bounds`, `stepSize`).

3. **Write the GPU sampler**: A `__host__ __device__` function that evaluates
   the field at an arbitrary 3D point, dispatched via the
   `VISRTX_CUSTOM_SAMPLE_DISPATCH` macro.

4. **Supply a conservative value range** (optional but recommended): A function
   returning a `{lo, hi}` interval that bounds the field value, used by VisRTX
   to build a per-macrocell space-skipping grid for delta tracking. Two hooks
   are available — define exactly one:

   | Macro | Signature | Trade-off |
   |-------|-----------|-----------|
   | `VISRTX_CUSTOM_VALUE_RANGE_DISPATCH(data, boxLo, boxHi)` | `box1` over an object-space AABB | Tight per-cell bounds → best space skipping |
   | `VISRTX_CUSTOM_GLOBAL_VALUE_RANGE_DISPATCH(data)` | `box1` over the whole domain | Constant bound, zero extra cost → no space skipping |

   Without either hook the engine falls back to point-supersampling each cell,
   which can mis-bound the field (the volume may render too dark or vanish). See
   *Space skipping & majorants* below.

5. **Register at static init**: A small registration file calls
   `visrtx::registerCustomField("subtypeName", factory)`, which inserts the
   type into the `SpatialFieldRegistry`. The ANARI device discovers it at
   runtime when `anariNewSpatialField(device, "subtypeName")` is called.

The sample and value-range macros, plus their `__device__` helpers, live in a
single dispatch header (`WeightedPointsFieldDispatch.h`) that CMake wires into
the core library via the `VISRTX_CUSTOM_FIELD_DATA_HEADER` and
`VISRTX_CUSTOM_SAMPLERS_HEADER` compile definitions. All pieces are compiled
into `libanari_library_visrtx.so` via CMake `target_sources`, keeping the core
VisRTX codebase unchanged.

## The Weighted Points Field

The `weightedPoints` field represents a continuous scalar field defined by a
set of 3D points, each carrying a scalar weight. The field value at any
position **p** is the sum of Gaussian contributions from nearby points:

```
f(p) = Σ  w_i · exp( -|p - x_i|² / (2σ²) )
```

### Octree acceleration

Naively evaluating all N points per sample is O(N): far too slow for
real-time volume ray marching on the GPU. Instead, the demo builds a
**bounding-volume octree** on the CPU:

- **Leaf nodes** store the weighted centroid of a small cluster of points.
- **Internal nodes** aggregate the total weight and centroid of their subtree.
- The GPU sampler **traverses the octree** with a distance-based LOD criterion:
  if a node's spatial extent is small relative to its distance from the sample
  point, its aggregate contribution is used directly instead of descending
  into children.

This gives O(log N) sampling cost with controllable quality via two parameters:

| Parameter | Description |
|-----------|-------------|
| `sigma` | Gaussian kernel width (Å). Controls how "blobby" each atom appears. Auto-computed from median nearest-neighbor distance. |
| `cutoff` | LOD distance threshold (Å). Nodes farther than this from the sample point are approximated by their aggregate. Auto-computed from domain diagonal. |

### Space skipping & majorants

VisRTX renders volumes with delta tracking, which needs a conservative `{lo,
hi}` value interval over each region it steps through. For custom fields this
comes from one of the two value-range hooks described in step 4:

- **`VISRTX_CUSTOM_VALUE_RANGE_DISPATCH`** — per-AABB interval, enables tight
  space skipping. Used by this demo via `ValueRangeWeightedPoints.cuh`.
- **`VISRTX_CUSTOM_GLOBAL_VALUE_RANGE_DISPATCH`** — single domain-wide interval,
  simpler to implement but provides no per-cell empty-space culling.

The weighted points field implements the per-AABB variant by summing, over *all*
octree nodes, each node's largest Gaussian contribution within the AABB
(evaluated at the box point closest to the node center).

This is provably conservative — the sampler's value at any point is a sum over
an LOD cut, which is a *subset* of all nodes, so the all-nodes sum can only
over-estimate. The result is a per-macrocell `{0, hi}` grid that lets the
renderer skip empty space without ever under-bounding the field. See
`fields/samplers/ValueRangeWeightedPoints.cuh` for the derivation.

### GPU data layout

The octree is serialized into two flat arrays for GPU consumption:

- **values** (`float×4 per node`): `[x, y, z, weight]`: position and
  aggregate weight of each node.
- **indices** (`int32×2 per node`): `[childBegin, childEnd)`: index range of
  children in the values/indices arrays. Leaves have `(0, 0)`.

The `WeightedPointsFieldData` struct also carries the precomputed `1/(2σ²)`
factor and the conservative global `maxValue` used as the value-range fallback
when the octree is empty.

## PDB File Support

The demo can load atomic coordinates from **Protein Data Bank (PDB)** files,
the standard format for macromolecular structures. Each `ATOM` / `HETATM`
record provides 3D coordinates (in Ångströms) and an occupancy factor used as
the point weight.

### Usage

```bash
tsdDemoCustomField --pdb /path/to/structure.pdb
```

Or without arguments for a random point cloud.

The ImGui controls panel exposes `sigma`/`cutoff`, the transfer function, and an
**Animation** section that perturbs the points over time (amplitude expressed in
multiples of the median nearest-neighbor distance, so motion scales with the
data). The octree is rebuilt each frame via a fast path that keeps blob size
stable across the animation.

### Why weighted points for molecular data?

Traditional molecular viewers render atoms as discrete spheres or stick models.
The weighted points field offers a complementary visualization:

- **Electron density approximation**: The Gaussian sum produces a smooth
  scalar field reminiscent of electron density maps, revealing the molecular
  *envelope* rather than individual atoms.
- **Transfer function control**: By mapping different density thresholds to
  distinct colors and opacities, specific structural features (surface, buried
  cavities, dense core) can be selectively highlighted or hidden.
- **Hardware-accelerated ray marching**: VisRTX renders the volume using
  OptiX, achieving interactive frame rates even for large structures
  (e.g., 23,000+ atoms in the SARS-CoV-2 spike protein 6VXX).
- **Level-of-detail**: The octree naturally provides multi-resolution
  rendering: zoomed-out views use coarse aggregates while close-ups resolve
  individual atomic contributions.

## File Overview

```
customField/
├── fields/
│   ├── WeightedPointsFieldData.h        # GPU data struct (shared CPU/GPU)
│   ├── WeightedPointsFieldDispatch.h    # Sample + value-range dispatch macros
│   ├── samplers/
│   │   ├── SampleWeightedPoints.cuh     # __host__ __device__ field evaluator
│   │   └── ValueRangeWeightedPoints.cuh # Conservative per-AABB value interval
│   ├── WeightedPointsField.h/cpp        # Host-side ANARI object
│   └── RegisterWeightedPointsField.cpp  # Static registration
├── WeightedPointsOctree.h/cpp           # CPU octree builder
├── WeightedPointsControls.h/cpp         # ImGui UI panel + PDB loader + animation
├── tsdDemoCustomField.cpp               # Application entry point
├── CMakeLists.txt                       # Build configuration
└── README.md                            # This file
```
