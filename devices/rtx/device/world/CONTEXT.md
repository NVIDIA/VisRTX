# World

Maintains the acceleration structures that make committed scene content
traceable: bottom-level structures per geometry, and separate top-level
structures for surfaces and volumes. Also assembles the per-instance light
list renderers sample, including lights synthesized from emissive surfaces.

## Language

**BLAS**:
Bottom-level acceleration structure built over one geometry's primitives.

**TLAS**:
Top-level acceleration structure over Instances. Two exist: the Surface TLAS
and the Volume TLAS — they are traversed independently.

**Instance**:
A transformed placement of a group's content in a TLAS.
_Avoid_: OptixInstance (its encoding, not the concept)

**Traversable**:
The handle a kernel traces rays against; the output of a TLAS or BLAS build.

**Rebuild**:
Reconstructing acceleration structures after committed scene content changes.
Triggered lazily, before the next launch that needs them.

**Interval**:
A ray segment inside a volume, bounded by its entry and exit points. Volume
traversal enumerates Intervals for integration; surface traversal yields hit
points. This difference is why the two TLASes are separate.

**Emissive Surface**:
A surface whose material's emission is not provably zero. Determined at
commit time from the material alone, never from rendering.

**Geometry Light**:
A light-registry entry synthesized from an Emissive Surface rather than
authored as an ANARI light object. Instanced and sampled exactly like lights
authored via the API. "Synthesize" names the act of generating one.
_Avoid_: area light (ambiguous with the Rect/Ring/Sphere analytic lights),
emitter, synthesized light (the origin is the verb, not the noun), mesh
light (covers sphere/cone geometries too)
