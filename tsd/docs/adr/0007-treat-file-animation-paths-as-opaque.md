# Treat file-animation paths as opaque TSD state

SciVis Studio preserves a file-animation dataset's authoritative ordered source
paths exactly as stored. It does not resolve them relative to the dataset or
project, rebase them during Save As, copy their targets, or otherwise define
their semantics. The derived core TSD animation receives those paths and owns
path interpretation and per-frame I/O. A file-animation dataset remains usable
only where its recorded paths remain valid. Each persisted list entry is only a
path string; per-file size, modification time, absolute/relative duplicates,
and cached availability are omitted, while importer type and import options are
stored once at the dataset level.
