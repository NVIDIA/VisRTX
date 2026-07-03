# Frontend

Implements the ANARI object model for the VisRTX device: objects are created,
parameterized, and committed on the API side, and their state becomes resident
on the GPU.

## Language

### Object Lifecycle

**Commit**:
The point where an object's pending parameters are parsed, validated, and
packed into its GPU representation.

**Registry**:
The per-object-type collection that holds the GPU representations of all
committed objects of that type.

**Slot**:
One entry in a Registry. Freed slots are reused by later objects.

**Registry Index**:
The stable handle identifying an object's Slot. It survives recommit, so
objects referencing each other never need updating when a referenced object
changes.
_Avoid_: pointer, handle

**Upload**:
Copying an object's packed state into its Slot.

**Deferred Upload**:
Array data queued when the application unmaps it and transferred in batch when
the next frame renders, rather than immediately.

### Objects

**Material**:
The ANARI material object (matte, physically based, MDL). Names the API-side
object only.
_Avoid_: shader, MDL material (see the MDL context), material shader (see the
Render Pipeline context)

**Sampler**:
The ANARI sampler object mapping surface attributes to values, typically via
an image.
_Avoid_: texture

**Instance**:
The ANARI instance object placing a group in the world with a transform.

**Frame**:
The ANARI frame object: the render target owning result buffers (color, depth,
IDs) and the camera/renderer/world it renders with.
_Avoid_: framebuffer, accumulation frame (see the Render Pipeline context)

**Renderer**:
The ANARI renderer object selecting a rendering technique (fast, quality,
interactive, debug, test) and its parameters.
_Avoid_: pipeline

**Spatial Field**:
The volumetric data source a volume samples (structured, unstructured, NanoVDB,
neural).
_Avoid_: volume (a volume renders a spatial field; it is not one)
