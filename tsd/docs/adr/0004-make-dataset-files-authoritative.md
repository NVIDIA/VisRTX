# Make dataset files authoritative

`project.tsd` stores only each dataset's project-local ID and human-readable
name; all dataset-specific metadata, provenance, frame references, and scene
content live in `datasets/<name>.tsd`. This mirrors the rig boundary, makes a
dataset independently discoverable and portable, avoids competing serialized
representations, and treats runtime availability as derived state rather than
persisted truth. Dataset files carry an explicit SciVis Studio dataset metadata
schema; discovery rejects other TSD asset types rather than inferring datasets
from arbitrary scene payloads. Static provenance stores source-path text and
dataset-level importer settings only, without cached file size/time metadata or
separate absolute and project-relative forms. A normal project save serializes
only dirty datasets. It stages and validates every dirty dataset before
replacing any managed asset, so a staging failure commits none of them;
`project.tsd` is written only after the asset replacements succeed. Save As
serializes every dataset into the new project directory so the destination is a
complete representation of current in-memory state. Save As fails with a
consolidated repair list if any dataset is unavailable; it never omits a
dataset or copies stale serialized bytes as a substitute. Explicit Dataset
Archive Load/Save complements filesystem copy plus discovery: Load assigns a
new project-local ID, while Save omits that identity from the portable Archive.
Opening a project eagerly loads every dataset in its inventory and reconstructs
its subtree and owned animations; this persistence split does not introduce
dataset-level lazy loading. Core file-animation bindings continue to load their
individual frames on demand.
