# One OptiX pipeline per renderer

Each renderer subtype (fast, quality, interactive, debug, test) gets its own
OptiX pipeline and shader binding table, instead of one uber-pipeline that
branches on a launch parameter. Each renderer wants its own payload layout,
trace depth, and compile options; an uber-raygen would pay worst-case register
pressure and divergence for all renderers even when running the cheapest one.
The debug renderer's validation and exception flags would also poison a shared
pipeline's performance.

## Consequences

- N pipelines compile at startup and N sets of SBT bookkeeping exist.
- Renderers evolve independently: changing one renderer's programs or compile
  options cannot regress another's performance.
