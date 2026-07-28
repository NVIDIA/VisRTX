# Bind mesh attributes per Surface, not once per mesh

A USD Mesh with `GeomSubset` children converts to one TSD Surface per subset.
`UsdGeometry.cpp` used to build the attribute set once, on the parent mesh, and
let each subset borrow it through a hard-coded list of `vertex.normal`,
`vertex.attribute0`, and `vertex.color`. That works only for vertex-interpolated
data, which is indexed by the same vertex indices the subset already carries.

Face-varying data is not. It is indexed by `3 * triangle + corner` against the
mesh's full triangulation, so a subset -- which draws a chosen subset of those
triangles -- cannot point at the parent array; the corners it wants are not
contiguous and not at the offsets its own primitives imply. The same is true of
uniform data, which the importer expands to one value per triangle. Sharing was
therefore not merely incomplete, it was unavailable: assets authoring
`interpolation = "faceVarying"` texture coordinates -- which is what USD assets
overwhelmingly do -- rendered their subsets with no UVs at all.

Attributes are now expanded onto the triangulation once per mesh
(`triangulatePrimvars`) and then *gathered* per Surface for the triangles that
Surface draws (`buildTriangleGeometry`). Vertex-interpolated primvars still
create a single Array shared by every Surface, because for them the gather is
the identity; only per-triangle and per-corner data is copied per subset. The
cost of the copy is the price of correctness, and it is bounded by the size of
the mesh however many subsets divide it.

Two consequences follow from moving the binding to the Surface.

A subset can resolve its own material's UV primvar name. `ResolvedMaterial`
carries the primvar its texture reader asked for, and that answer now reaches
the geometry the material is bound to, instead of the mesh-level binding
deciding `attribute0` for subsets it knows nothing about. A subset without its
own answer falls back to the mesh's, then to the conventional `st`.

Faces that no subset claims become their own Surface under the mesh's material.
Previously the parent geometry was built, populated, and then never surfaced
when subsets existed -- unassigned faces were invisible and the objects were
dead weight in the scene. USD's own model is that a face outside every
`materialBind` subset keeps the mesh's binding, so that is what it gets.

One older gap is now easier to see and is deliberately left alone. `GeomSubset`
indices name coarse faces, but when a mesh is refined (ADR 0017 leaves
`refinementLevel` at 2) the triangulation this code selects from describes
*refined* faces, so a subset over coarse face 1 of a `catmullClark` quad pair
picks up refined face 1 -- 2 triangles rather than the 32 that face became.
Correcting it means carrying OpenSubdiv's refined-to-coarse face mapping out of
`refineMesh`, which is its own change; it is called out here so the leftover
Surface's suddenly-visible triangle count is not read as a regression from this
one.

The appeal to `materialBind` above is approximate in one respect, also unchanged
by this work: the importer treats every `geomSubset` child as a material subset,
because `HdGeomSubsetSchema` exposes only `type` and `indices` -- the scene
index does not carry `familyName`. A face claimed by a subset from some other
family is therefore counted as claimed and stays out of the leftover Surface.
Assets in the wild author `materialBind` subsets under a Mesh; if one appears
that does not, the family will have to be recovered from the Stage rather than
from the scene index.
