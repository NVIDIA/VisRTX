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

**Emission Classifier**:
The MDL-pure analysis that lowers a Compiled Material's emission expressions
into Emission IR and folds them into an Emission Descriptor. It describes;
it never decides light registration (that is the renderer-side policy, see
ADR-0007).
_Avoid_: deciding registration in the classifier

**Emission IR**:
The owned lowering of the MDL emission expression DAG (constants, parameters,
calls, textures) the classifier folds over. Retains no MDL-SDK expression
pointers.

**Emission Descriptor**:
The immutable per-material output of the Emission Classifier: per-slot
`{verdict, edfKinds, magnitude, intensity mode}` plus the argument/resource
dependencies the emission reads.

**Faithful Set**:
The consumer-exported set of EDF kinds a renderer can evaluate faithfully on
its synthetic next-event hit (`kFaithfulSet`). A described slot registers as a
Geometry Light only when its EDF kinds are a subset of it.
