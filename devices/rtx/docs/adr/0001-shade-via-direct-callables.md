# Shade via OptiX direct callables

Material and spatial-field shading is implemented as OptiX direct callables
dispatched by index, not compiled into the renderer pipelines. The driver is
commit-time latency: adding or changing a material must never trigger a
renderer pipeline rebuild, which would stall interactive applications for
seconds. The direct-callable mechanism arrived with MDL support: runtime-compiled
MDL materials cannot be baked into a renderer pipeline, so built-in materials were
moved onto the same callable dispatch to unify all material handling under one path.

## Consequences

- Every shading evaluation pays `optixDirectCall` overhead, and the compiler
  cannot inline or optimize across the call boundary.
- Pipeline register pressure is sized by the fattest callable linked in.
- New material types only need to provide the fixed set of shading entry
  points; no renderer code changes.
