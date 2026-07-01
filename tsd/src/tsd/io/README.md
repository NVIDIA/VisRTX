## IO Library (`tsd_io`)

`tsd_io` handles foreign-format conversion, native TSD Archives, procedural
scene generation, and component serialization.

### High-Level Concepts

- File importers for full scenes and geometry datasets:
  AGX, ASSIMP, AXYZ, DLAF, E57, ENSIGHT, GLTF, HDRI, HSMESH, NBODY, OBJ, PDB,
  PLY, POINTSBIN, PT, SILO, SMESH, SWC, TRK, USD, VTP, VTU, XYZDP.
- Volume/spatial-field importers (`import_RAW`, `import_NVDB`, `import_VTI`,
  etc.) and `import_volume()` dispatch helpers.
- Procedural generators for test and demo scenes (`generate_randomSpheres`,
  `generate_material_orb`, `generate_default_lights`, and others).
- Native Scene, Object, Layer Subtree, Camera, Renderer, Animation, and
  Animation Manager Archives, exposed through `archives.hpp`.
- Reusable component serialization between TSD objects and
  `tsd::core::DataTree` nodes, exposed through `serialization.hpp`.
- Foreign export helpers for scene-to-USD and structured-volume-to-NanoVDB,
  exposed through `exporters.hpp`.

### Why Use This Library

- You want a single API surface for importing many scene and volume formats.
- You need deterministic generated content for tests, demos, or device bringup.
- You want to save/load native TSD Archives or export scene data to other
  tools.

### Build Notes

- Optional importer backends are controlled by CMake options such as
  `TSD_USE_ASSIMP`, `TSD_USE_HDF5`, `TSD_USE_USD`, `TSD_USE_VTK`,
  `TSD_USE_SILO`, and `TSD_USE_TORCH`.
