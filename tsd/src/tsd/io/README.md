## IO Library (`tsd_io`)

`tsd_io` handles scene import/export, procedural scene generation, and
serialization workflows.

### High-Level Concepts

- File importers for full scenes and geometry datasets:
  AGX, ASSIMP, AXYZ, DLAF, E57, ENSIGHT, GLTF, HDRI, HSMESH, NBODY, OBJ, PDB,
  PLY, POINTSBIN, PT, SILO, SMESH, SWC, TRK, USD, VTP, VTU, XYZDP.
- Volume/spatial-field importers (`import_RAW`, `import_NVDB`, `import_VTI`,
  etc.) and `import_volume()` dispatch helpers.
- Procedural generators for test and demo scenes (`generate_randomSpheres`,
  `generate_material_orb`, `generate_default_lights`, and others).
- Serialization between TSD scene objects and `tsd::core::DataTree` nodes.
- Export helpers:
  scene-to-USD and structured-volume-to-NanoVDB.

### Why Use This Library

- You want a single API surface for loading many scene and volume formats.
- You need deterministic generated content for tests, demos, or device bringup.
- You want to save/load TSD scene state or export scene data to other tools.

### Build Notes

- Optional importer backends are controlled by CMake options such as
  `TSD_USE_ASSIMP`, `TSD_USE_HDF5`, `TSD_USE_USD`, `TSD_USE_VTK`,
  `TSD_USE_SILO`, and `TSD_USE_TORCH`.
