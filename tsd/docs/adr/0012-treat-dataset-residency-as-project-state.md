# Treat dataset residency as project state, not shot intent

Each dataset's residency (Loaded or Unloaded) is persisted per dataset in the
project manifest so that a memory-constrained machine can open a project
without hydrating datasets it cannot afford, while the same project renders
fully elsewhere. Residency lives in the manifest — not in the dataset asset,
which stays residency-agnostic and portable, and not in a per-machine sidecar,
which would add a second persistence channel for one bit of state. Session
residency is the single source of truth: the `--openUnloaded` override changes
only each dataset's initial residency at open (marking the project dirty when
it diverges from the manifest), and any subsequent save persists actual
residency. This revises ADR 0004's rule that opening a project eagerly loads
every dataset in its inventory: opening hydrates only resident datasets. It
still introduces no implicit lazy loading — an Unloaded dataset enters the
scene only through an explicit Dataset Load, never on demand. Unload requires
a clean dataset and never writes to disk, so an Unloaded dataset is read-only
as an asset. Residency is a workstation memory concern and never alters shot
intent: final shot rendering materializes every bound, enabled dataset
regardless of stored residency and fails hard when one cannot be made fully
resident, while the interactive viewport truthfully shows only what is
resident. Rejected alternatives: session-only residency (project open itself
becomes the out-of-memory event), and WYSIWYG final renders (farm output
silently missing datasets that were unloaded on a workstation).
