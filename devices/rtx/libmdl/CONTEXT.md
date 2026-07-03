# MDL

Compiles MDL material definitions into GPU code and editable argument data.
Standalone from the device; `device/mdl/` is a consumer.

## Language

**MDL Material Definition**:
A material as authored in MDL source (a `.mdl` module).
_Avoid_: material (alone), shader

**Compiled Material**:
An MDL Material Definition compiled with a specific parameter set; the input
to code generation.

**Target Code**:
The generated PTX implementing a Compiled Material's functions on the GPU.

**Class Compilation**:
The compilation mode where parameter values stay editable through an Argument
Block, so edits do not require recompiling the material.

**Argument Block**:
The GPU blob holding a Compiled Material's runtime-editable parameter values.
A descriptor defines its layout; an instance holds actual values.

**Texture Runtime**:
The texture access machinery Target Code calls at shading time.
_Avoid_: sampler (see the Frontend context)
