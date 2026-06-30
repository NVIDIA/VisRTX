# Delegate file-animation I/O to TSD

A file-animation dataset persists the one authoritative ordered source-file
list and enough TSD state to recreate a core file-animation object together
with its associated layer subtree. The core `Animation` and `FileBinding` are
derived runtime objects, not a second persisted authority. SciVis Studio creates
and drives that association but does not load, unload, copy, or otherwise
manage individual frames; the runtime binding owns per-frame I/O. The dataset
`.tsd` contains its source list and current scene representation, not an
embedded copy of every source frame. SciVis Studio does not preflight the list
or mark the dataset unavailable when an individual frame fails to load; that
failure remains local to the core animation when the frame is requested.
