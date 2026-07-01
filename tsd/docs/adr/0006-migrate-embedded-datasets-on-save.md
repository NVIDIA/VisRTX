# Migrate embedded datasets on explicit save

SciVis Studio opens legacy projects with embedded datasets without modifying
them, preserving dataset IDs and shot bindings while marking the datasets for
extraction. The next explicit save writes and verifies all required Dataset
Archives before replacing `project.tsd` with the new schema, so merely
opening a project never upgrades it and a failed extraction leaves the legacy
project intact. File-animation source frames are not preflighted during
migration: extraction uses the dataset's persisted source-file list as the
authority, serializes its layer subtree, and recreates the runtime TSD file
animation from that list. Any duplicate list held by a legacy runtime binding
is not authoritative. Later per-frame I/O remains the animation's
responsibility; migration blocks only when the dataset state cannot be
serialized or reconstructed.
