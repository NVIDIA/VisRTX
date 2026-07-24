# Bake prototype-internal transforms when importing instanced USD content

`TransformsToAnariVisitor` never pushes a transform-array node's matrices onto
the transform stack, so a transform node nested beneath one composes against
that array node's ancestors instead: its subtree renders once, un-instanced,
rather than once per instance. Transform-array nodes are therefore leaf-only
instancing, while USD prototypes are typically `Xform` subtrees holding several
gprims at their own local transforms. The importer resolves this by importing
each prototype exactly once into shared TSD objects and baking each gprim's
prototype-root-relative transform into its vertex data, leaving a flat set of
Surfaces that a transform-array node may legally instance; a point instancer
becomes one transform-array node and a native instance becomes one mat4 node
reusing the same objects. Baking is cheap precisely because a prototype is
imported once regardless of instance count. Prototypes whose internal transforms
are themselves animated cannot be baked and fall back to expanded per-instance
transform nodes. Do not "correct" the baked vertex data without first changing
how the render index composes transform-array nodes.
