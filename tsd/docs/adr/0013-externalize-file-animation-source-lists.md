# Externalize file-animation source lists into sibling Source List Files

A File Animation Dataset's authoritative source list is persisted only in a
Source List File: a UTF-8 text file with one path per line, saved as
`datasets/<name>.sources` next to `datasets/<name>.tsd`. The dataset file
stores importer settings but no frame paths; the sibling relationship is a
naming convention, not a stored path. This makes the list storable and
editable by humans and external tools without Studio, at the cost of a
two-file dataset asset. The pair is the asset everywhere: Dataset Archive
Save, Save As, and the ADR 0004 save transaction write both files together,
Dataset Archive Load fails cleanly without the sibling, and Dataset Discovery
still scans only `.tsd` files. This revises ADR 0003/0004's rule that the
dataset `.tsd` contains its frame references and ADR 0005's rule that the
dataset directory holds no supporting files.

Line order is frame order; blank lines are ignored and lines are trimmed;
there are no comments, globs, quoting, or includes — reserved syntax would
restrict what paths can contain, and glob expansion would make frame count
and order depend on directory state rather than on what the file says.
Relative entries are resolved once, at read, against the Source List File's
directory — a deliberate narrow revision of ADR 0007's opaque-path rule,
because a hand-written relative path that resolves against process CWD is a
trap. After that single anchoring, entries remain opaque: Studio still never
rebases entries, copies targets, or persists resolved duplicates, and writes
the raw entries back verbatim.

The file is read exactly at Dataset Load; the in-memory authority is the raw
entries as authored. Studio rewrites the file only when the source list was
edited in-tool (last writer wins; no watching, merging, or mtime checks), and
saving a clean dataset never touches it, so external edits survive unrelated
saves and take effect on the next load. A missing, unreadable, or empty
Source List File makes the dataset Unavailable; individual entries are still
never preflighted (ADR 0003), so one bad path costs one frame, not the
dataset. Legacy dataset files with embedded `sourceFiles` open unmodified and
are marked for migration; the next explicit save writes the Source List File
and rewrites the `.tsd` without paths, following the ADR 0006 pattern.
Entries migrate verbatim: a legacy relative entry changes anchor from CWD to
the file's directory, accepted because its old meaning was never defined.

Rejected alternatives: keeping an embedded copy alongside the external file
(competing serialized representations), a stored path to the list file
(bookkeeping whose only sanctioned value is "next to me"), and embedding the
list when saving portable archives (a second serialization path hidden from
the humans the file serves). The name "manifest" was rejected because it
already denotes `project.tsd`.
