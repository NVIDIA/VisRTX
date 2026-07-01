# Context Map

## Contexts

- [TSD I/O](./src/tsd/io/CONTEXT.md) — moves data between TSD and external
  representations and persists native TSD state
- [TSD App](./src/tsd/app/CONTEXT.md) — composes reusable application state
  around TSD scenes, animations, rendering, and interaction
- [SciVis Studio](./apps/interactive/scivisStudio/CONTEXT.md) — organizes
  scientific-visualization assets into projects and shots

## Relationships

- **SciVis Studio → TSD I/O**: SciVis Studio uses generic TSD I/O mechanisms,
  but owns its application-specific project, dataset, rig, and shot language.
  SciVis Studio terminology does not define the generic TSD I/O vocabulary.
- **TSD App → TSD I/O**: TSD App composes Archives produced by TSD I/O with
  application-level state to create Application Dumps. TSD I/O does not depend
  on or create Application Dumps.
