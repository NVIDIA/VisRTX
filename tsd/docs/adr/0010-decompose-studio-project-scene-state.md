# Decompose SciVis Studio project scene state into owned archives

SciVis Studio projects no longer write a residual Scene Archive into the
project manifest. Datasets, light rigs, and camera rigs retain their
independently owned Archives, and the remaining scene-owned camera and renderer
pools move to required
`scene/cameras.tsd` and `scene/renderers.tsd` Archives. This makes project
ownership explicit and avoids treating a pruned scene as a full Scene Archive.
Legacy manifests with an embedded `context` remain readable; the next explicit
save writes and validates the decomposed layout before replacing the manifest.
