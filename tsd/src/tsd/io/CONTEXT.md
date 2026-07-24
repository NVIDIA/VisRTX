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

**Claimed Prim**:
A prim an importer handles outside its generic conversion path, and which is
therefore withheld from that path.

**Placeholder Node**:
A named, empty, disabled layer node standing where a prim that produced no
renderable TSD content would have been.

**Import Report**:
A structured record of one Import: which prims became TSD scene objects and,
for each that did not, the reason.
