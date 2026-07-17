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

### MaterialX

**Distribution Document**:
A `.mtlx` document shipped with a MaterialX installation: the standard-library
nodedefs and their implementation modules. Never authored, embedded, or shipped
by the device or an application — always resolved from an installation.
_Avoid_: stdlib file, builtin document

**Instantiation Document**:
A `.mtlx` document that binds a nodedef into a usable material (a
`surfacematerial` referencing a surfaceshader node). Scene content: authored by
the application, possibly generated. Distinct from the Distribution Documents
whose nodedefs it binds.
_Avoid_: builtin, preset document

**Search Chain**:
The ordered runtime resolution of the MaterialX installation: explicit device
parameter, then environment convention, then MaterialX self-discovery, then a
development-build last resort. First hit wins; the earlier the source, the more
explicit the user's intent.
