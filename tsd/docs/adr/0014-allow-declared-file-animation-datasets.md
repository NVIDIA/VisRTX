# Allow declared file-animation datasets

A File Animation Dataset can be created by declaration: from its source list
alone, without reading any source files. A declared dataset's asset is the
ordinary two-file pair — a dataset file recording importer type and settings,
and its sibling Source List File — but contains no scene representation.
Declaration exists so datasets can be authored for files that live only on
another machine (data too large to transfer); on the authoring machine the
dataset is expectedly Unavailable, which is a machine-relative assessment,
not a fault in the asset. Only file-animation datasets can be declared: a
static dataset without its content is a contradiction.

Declaration is an explicit creation mode, never a fallback. Default creation
imports eagerly and fails loudly when the data is absent, so a mistyped path
is an error rather than a silently "successful" dataset full of garbage
entries. Declared creation performs no filesystem access on the source paths
— not even an existence check (ADR 0003/0007: entries are opaque and never
preflighted) — so declaring behaves identically on every machine, including
one where the paths happen to resolve.

The first successful Dataset Load of a declared dataset materializes it: the
runtime representation is built by importing from the source list, and the
dataset is marked dirty so the next save records the scene representation,
upgrading the asset in place following the ADR 0006 migration pattern. Load
itself never writes to disk, and a failed materialization changes nothing.
Asset validation accepts the representation-less flavor for file-animation
datasets only.

Rejected alternatives: automatic fallback to declaration when paths do not
resolve (success becomes indistinguishable from failure in scripts); baking
the scene representation into the asset at load time (loading would write to
disk); and preflighting declared paths to warn about missing files (a "small"
path semantic that ADR 0007 deliberately refuses).
