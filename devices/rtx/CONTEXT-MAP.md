# Context Map

## Contexts

- [Frontend](./device/CONTEXT.md) — implements the ANARI object model and moves
  committed object state into GPU residency
- [Render Pipeline](./device/renderer/CONTEXT.md) — turns committed GPU state
  into images via per-renderer OptiX pipelines and callable-based shading
  (covers `device/frame/`, `device/renderer/`, `device/gpu/`)
- [World](./device/world/CONTEXT.md) — maintains the acceleration structures
  that make committed scene content traceable
- [MDL](./libmdl/CONTEXT.md) — compiles MDL material definitions into GPU code
  and editable argument data (`device/mdl/` is a consumer)

## Relationships

- **Frontend → Render Pipeline**: Frontend commits produce packed GPU object
  state in registries; the Render Pipeline consumes it by registry index at
  launch. The Render Pipeline never parses ANARI parameters.
- **Frontend → World**: Frontend commits mark scene content out of date; the
  World rebuilds acceleration structures before the next launch.
- **World → Render Pipeline**: the World hands traversables to the Render
  Pipeline; the pipeline traces against them but never builds them.
- **Frontend → MDL**: the `mdl` material subtype delegates compilation to MDL;
  the Frontend owns the ANARI-facing Material, MDL owns everything from
  definition to target code.
- **MDL → Render Pipeline**: Compiled Materials link into renderer pipelines
  as PTX callable modules, indistinguishable from built-in Material Shaders at
  dispatch time.
