---
status: superseded by ADR-0007
---

# Anchor file-animation paths to the Dataset Archive

Relative paths from a file-animation dataset to its external frame files are
resolved from the directory containing the Dataset Archive, rather than
from the project root. This keeps the dataset definition and its frame files
relocatable as a bundle and allows the dataset to move between independent
projects without rewriting its paths; absolute paths are fallback relink hints
only.
