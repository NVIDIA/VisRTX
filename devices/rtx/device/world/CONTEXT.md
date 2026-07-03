# World

Maintains the acceleration structures that make committed scene content
traceable: bottom-level structures per geometry, and separate top-level
structures for surfaces and volumes.

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
