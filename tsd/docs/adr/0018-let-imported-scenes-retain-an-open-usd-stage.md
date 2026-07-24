# Let imported scenes retain an open UsdStage

Time-varying geometry is imported as one eager first frame plus a `FileBinding`
that re-pulls point and index arrays from the still-open stage and scene index
at time `t`, rather than baking every sampled frame into TSD Arrays. Baking a
hundred-frame, million-vertex mesh costs on the order of a gigabyte per mesh,
while the lazy binding keeps memory flat regardless of frame count and follows
the pattern `EnSightFileBinding` and `SpatialFieldFileBinding` already
establish. The consequence is that import is not a fully detached operation for
animated content: an imported scene's lifetime holds a `UsdStage` and its scene
index, serialization records file and prim paths and reconstructs by re-opening
rather than by copying data, scrubbing performs real work per frame, and stage
access must be accounted for when a scene is used across threads. Static content
carries no such dependency.
