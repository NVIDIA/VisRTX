# Import USD through a Hydra scene index

TSD imports USD stages by consuming a UsdImaging scene index chain rather than
traversing UsdGeom and UsdShade schemas directly. Composition, purpose and
visibility resolution, native and point instancing, material binding and render
context selection, primvar interpolation, implicit-shape conversion, NURBS
approximation, and skinning are all resolved by OpenUSD's own filtering scene
indices; the importer contributes one converter per Hydra prim type and nothing
more. The stage stays open alongside the scene index, so TSD-specific data that
Hydra does not model — `customData` carriers, `anari:` and `tsd:io:` attributes,
render settings — is still read directly from prims by path. This costs links
against `hd`, `usdImaging`, `hio`, and `hdsi`, and requires working in
data-source idioms rather than schema APIs, but it removes the class of silent
omissions that a hand-rolled traversal accumulates one unhandled prim type at a
time. A Hydra render delegate was rejected for the same job: it is a push-based
sync architecture for repeated frame updates, not a one-shot conversion into a
scene the user then edits by hand.
