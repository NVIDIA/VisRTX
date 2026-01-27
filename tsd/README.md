TSD ("Testing Scene Description")
=================================

This project started as a medium to learn 3D scene graph library design in C++
as well as be an ongoing study on how a scene graph and ANARI can be paired. It
is by no means a production piece of software, but is made available on the off
chance it can be made useful to others.

While it is a "scene description" style library that has fairly generic
concepts, it has some very clear goals to keep the project scoped. First, the
'T' in TSD stands for "testing" -- TSD is just made for testing ANARI devices
in very specific ways based on development needs and interests. By no means does
TSD have ambition to be anything else, as representing _only_ the live
components of a scene shared by one or more ANARI devices is a small fraction of
everything that goes into production visualization or VFX pipelines. Second,
while things are not expected to change drastically, TSD does not (yet) track a
versioning system and thus API stability guarantees do not exist -- the
educational nature of TSD means API improvements are valued over stability.

With that in mind, this repository has the following components:

- [Core scene description library](src/)
- [Single-file examples showing off specfic concepts](apps/simple/)
- [Interactive ImGui-based viewer](apps/interactive/viewer/)
- [Experimental MPI distributed viewer](apps/interactive/mpiViewer/)
- [Basic unit tests](tests/)

Generally `libtsd` follows 1:1 with ANARI's object hierarchy, defined in the
ANARI specification
[here](https://registry.khronos.org/ANARI/specs/1.0/ANARI-1.0.html).

TSD is tested on Linux, but should work on other platforms (macOS + Windows).

## Notes on using tsdViewer

The main [`tsdViewer`](apps/interactive/viewer/) application is largely
self-explanitory, but has a few usage tips worth noting.

The list of available devices in the UI is controlled by a comma-separated list
stored in the `TSD_ANARI_LIBRARIES` environment variable. If this is empty, the
app will still populate the UI with some defaults, but users are encouraged to
always define `TSD_ANARI_LIBRARIES` in their environment (e.g. in a `.bashrc`
or similar).

Files can be loaded on the command line using a pattern of:

```bash
% ./tsdViewer -[loader type] [file1] [file2...]
```

When a loader type is selected, all filenames after it will use that same loader
until the next loader type is encountered. Common loader types are `assimp`,
`dlaf`, `hdri` (environment light), `obj`, and `ply`. See [tsd_io](src/tsd/io/)
for more, as well as [tsd::app::Core::parseCommandLine()](src/tsd/app/Core.cpp)
for the definitive list of available options. Note that some importers may only
work when explicitly enabled in CMake -- this is for when the importer incurs
an additional build dependency. Use `ccmake` or `cmake-gui` to discover what
additional options may need to be enabled.

## Components

- [Lua Scripting](src/tsd/scripting/) — Lua bindings and scripting API
