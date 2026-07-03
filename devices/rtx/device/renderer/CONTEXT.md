# Render Pipeline

Turns committed GPU state into images: per-renderer OptiX pipelines trace rays
and dispatch shading through callables. Covers the frame loop (`../frame/`),
renderers (`./`), and GPU-side shading structures (`../gpu/`).

## Language

**Renderer Pipeline**:
The compiled OptiX pipeline plus its shader binding table for one renderer
subtype. Each renderer subtype has its own.
_Avoid_: renderer (the ANARI object; see the Frontend context)

**Launch**:
One GPU execution of a Renderer Pipeline over the frame.

**Accumulation Frame**:
One Launch's worth of samples, blended into the accumulated result. A
converged image is many Accumulation Frames.
_Avoid_: sub-frame, iteration, pass

**Callable**:
A GPU function linked into a pipeline but dispatched by index at runtime. The
unit that lets shading change without rebuilding the pipeline.

**Material Shader**:
The set of Callables and packed parameters that implement shading for one
material. Built-in and MDL-compiled Material Shaders are dispatched
identically.
_Avoid_: material (alone)

**Shading Entry Point**:
One of the fixed Callable roles every Material Shader provides (initialize,
shade, evaluate tint/opacity/emission/transmission/normal, next ray).

**Field Sampler**:
The Callable pair (init, sample) through which a spatial field is evaluated
during volume integration.

**Parameter Source**:
Where a material parameter gets its value: an inline constant, a geometry
attribute, or a sampler.
