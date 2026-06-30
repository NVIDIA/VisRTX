# Keep the dataset directory flat

Managed dataset assets occupy only direct children of the project directory as
`datasets/<name>.tsd`, and Dataset Discovery scans only that flat namespace.
The directory contains no per-dataset subdirectories or supporting files;
file-animation frame data lives elsewhere. This keeps filesystem presence,
discovery candidates, and the managed dataset namespace unambiguous. Normal
project save leaves unlisted dataset files untouched because they may be pending
discovery; it deletes only a path associated with an explicit dataset removal
or the superseded path from a successful rename.
