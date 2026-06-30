# Use human-readable dataset filenames

Managed dataset assets are stored as `datasets/<name>.tsd`, matching camera and
light rigs, while `project.tsd` retains a stable project-local dataset ID for
shot bindings. Dataset names therefore use a portable filename-safe character
set and are unique within a project under case-insensitive comparison; renaming
a dataset also renames its managed file. For a discovered, not-yet-managed file,
the embedded dataset name is proposed during review; confirmation resolves any
collision and normalizes the manifest name, embedded name, and filename stem to
the same value.
