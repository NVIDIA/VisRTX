# TSD I/O

TSD I/O moves data between TSD and external representations and persists
native TSD state. Its language applies across TSD and is independent of the
application-specific vocabulary used by SciVis Studio.

## Language

### Native Persistence

**Archive**:
A native serialized representation of some TSD state. An Archive is independent
of its carrier: it may be embedded in another representation, transmitted
directly, or saved as a standalone `.tsd` file. Archive formats may carry
schema metadata when their validation or evolution requires it.

**Scene Archive**:
An Archive containing the current state of one entire TSD scene but no
animation-manager state; an explicit array-data policy may represent array
storage as proxy placeholders without excluding scene objects or layers.
Deserialization replaces the target scene's complete state.
_Avoid_: Scene dump

**Layer Subtree Archive**:
An Archive containing one layer subtree and the scene-object dependencies
required to reconstruct it. It does not contain animations; a higher-level
artifact may compose the subtree with Animation Archives. Deserialization
adds the subtree beneath a caller-selected location.

**Object Archive**:
An Archive containing one primary TSD scene object and the object dependencies
required to reconstruct it. Deserialization adds those objects to a scene.

**Camera Archive**:
An Archive containing the entire camera-object pool of a scene. Deserialization
replaces the target scene's camera pool rather than adding one selected object.

**Renderer Archive**:
An Archive containing the entire renderer-object pool of a scene.
Deserialization replaces the target scene's renderer pool rather than adding
one selected object.

**Animation Manager Archive**:
An Archive containing an animation manager's playback state and an embedded
Animation Archive for every animation it owns. Deserialization requires a
compatible scene to already exist and replaces the target manager's complete
state.

**Animation Archive**:
An Archive containing the state of one TSD animation. Its scene-object and
layer-node bindings refer into a compatible scene supplied during
deserialization; the Archive does not copy those scene dependencies.
Deserializing one Animation Archive adds that animation to a manager.

**Save**:
Persist a native TSD Archive or Application Dump to a file. The operation may
serialize an intermediate representation, but its boundary is a file.

**Load**:
Reconstruct native TSD state from an Archive or Application Dump stored in a
file. The operation may deserialize an intermediate representation, but its
boundary originates at a file.

**Serialize**:
Convert runtime TSD state into its native in-memory representation without
implying file I/O.

**Deserialize**:
Reconstruct runtime TSD state from its native in-memory representation without
implying file I/O.

**Write**:
Encode a native in-memory representation into a byte buffer without implying
file I/O.

**Read**:
Decode a byte buffer into a native in-memory representation without implying
file I/O.

**Import**:
Convert a non-native representation into TSD state.
_Avoid_: Load, deserialize

**Export**:
Convert TSD state into a non-native representation.
_Avoid_: Save, serialize

### Foreign-Format Import

**Stage**:
The composed scene an import reads a foreign representation from, identified by
the file that was opened.
_Avoid_: USD scene, USD file

**Stage Session**:
The retained open Stage and resolution chain shared by every animation binding
from one Import, identified by the Stage's file. Its lifetime is the bindings',
not the Import's: an Import with no animation lets go of its Session on return.
See
[ADR 0021](../../../docs/adr/0021-share-one-usd-stage-session-across-import-and-animation.md).
_Avoid_: stage cache, open stage

**Time Code**:
The Stage's own clock, distinct from TSD's normalized animation time. An Import
records the mapping between them; bindings re-resolve at a Time Code rather
than at a stored sample index.
_Avoid_: frame, timestep, sample index

**USD Layer**:
A single composition-arc source contributing opinions to a Stage. Always
qualified: unqualified **Layer** means a TSD Layer and never this.
_Avoid_: Layer (unqualified), sublayer

**Prototype**:
Content authored once and shared by every placement of it. An imported
Prototype is one set of TSD scene objects referenced from many layer nodes,
never a per-placement copy.

**USD Instance**:
A prim that places a Prototype at its own transform. Always qualified:
unqualified **Instance** means the ANARI object a render index emits.
_Avoid_: Instance (unqualified)

**Purpose**:
The visibility category of a foreign prim — `default`, `render`, `proxy`, or
`guide` — that determines whether an import includes it.

**Render Context**:
The flavor of shading network selected from a bound foreign material, such as
UsdPreviewSurface, MaterialX, or MDL.
_Avoid_: shader target, material backend

**Resolved Geometry**:
One foreign gprim read out into plain buffers — topology, points, and every
primvar already expanded and gathered — with no TSD scene objects involved.
Resolving is the half of geometry conversion that varies with Time Code;
building the Surfaces, Materials and Arrays around it is the half that does
not. An animation binding re-runs the first and never the second. See
[ADR 0022](../../../docs/adr/0022-refill-captured-arrays-rather-than-re-running-conversion.md).
_Avoid_: geometry cache, mesh data

**Part**:
One Surface's worth of a Resolved Geometry, named for the prim or subset it
came from. A mesh with per-face material subsets resolves to several Parts
sharing its vertex data; every other gprim resolves to one.
_Avoid_: submesh, chunk

**Claimed Prim**:
A prim an importer handles outside its generic conversion path, and which is
therefore withheld from that path.

**Placeholder Node**:
A named, empty, disabled layer node standing where a prim that produced no
renderable TSD content would have been.

**Import Report**:
A structured record of one Import: which prims became TSD scene objects and,
for each that did not, the reason.

**Import Options**:
The typed settings governing one Import — which Purposes to include, which
Render Contexts to prefer, what to emit materials as, how far to refine
subdivision surfaces, and which prim to import from. Converts to and from a
DataTree so it can be persisted with a project and driven from scripting.
_Avoid_: import config, import params

### Images

**Image**:
A decoded picture resident in a scene, held as an Array of texels. Identified
by its content, not by the sampler built from it: two materials binding the
same file at the same Color Space share one Image.

**Image Source**:
What identifies an Image's content — a resolved absolute path for a
file-backed image, or an importer-scoped stable id otherwise
(`gltf:<file>:<image>`, `assimp://embedded/<n>`, `pbrt:<file>::normal`) —
together with the Color Space it was decoded for and the Row Order it is
stored in.
_Avoid_: texture key, cache key

**Image Cache**:
The owner of every sampled Image for one scene. It holds the scene it caches
for, so a cached Array can never reach a different one, and it must not
outlive that scene. An image bound to a light's radiance rather than to a
Sampler may be built without it — see ADR 0014 — so "every" is a claim about
what a Sampler can read, not about every decode in the tree.
_Avoid_: texture cache

**Row Order**:
Which row of a picture a texel array stores first. Decoders declare the Row
Order their library produced and an Image Source asks for the one its consumer
wants; the Image Cache normalizes between them, and no importer flips texels
itself. A sampled image is stored top-down — row 0 is the top row, so texture
coordinate (0, 0) addresses the upper-left corner, which is where ANARI
addresses it. See
[ADR 0014](../../../docs/adr/0014-store-images-in-anari-orientation.md).
_Avoid_: flipped, vertical orientation, y-up

**Color Space**:
How a file's values relate to the linear values a renderer wants, either sRGB
or linear. Formats carrying an encoding of their own — EXR, and the block
format of a DDS — ignore what a caller asks for.
