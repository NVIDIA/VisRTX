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

3. **Write the GPU sampler**: A `__device__` function that evaluates the field
   at an arbitrary 3D point, dispatched via the
   `VISRTX_CUSTOM_SAMPLE_DISPATCH` macro.

4. **Register at static init**: A small registration file calls
   `visrtx::registerCustomField("subtypeName", factory)`, which inserts the
   type into the `SpatialFieldRegistry`. The ANARI device discovers it at
   runtime when `anariNewSpatialField(device, "subtypeName")` is called.

All four pieces are compiled into `libanari_library_visrtx.so` via CMake
`target_sources`, keeping the core VisRTX codebase unchanged.

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

### GPU data layout

The octree is serialized into two flat arrays for GPU consumption:

- **values** (`float×4 per node`): `[x, y, z, weight]`: position and
  aggregate weight of each node.
- **indices** (`int32×2 per node`): `[childBegin, childEnd)`: index range of
  children in the values/indices arrays. Leaves have `(0, 0)`.

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
│   ├── WeightedPointsFieldDispatch.h    # OptiX sampler dispatch macro
│   ├── WeightedPointsField.h/cpp        # Host-side ANARI object
│   └── RegisterWeightedPointsField.cpp  # Static registration
├── WeightedPointsOctree.h/cpp           # CPU octree builder
├── WeightedPointsControls.h/cpp         # ImGui UI panel + PDB loader
├── tsdDemoCustomField.cpp               # Application entry point
├── CMakeLists.txt                       # Build configuration
└── README.md                            # This file
```
