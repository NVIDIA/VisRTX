# TSD I/O Archive Refactor

Status: proposed

## Objective

Reorganize native TSD persistence around explicit Archives, reserve
import/export for non-native conversion, and move application-level snapshots
into Application Dumps owned by `tsd_app`. Preserve existing Archive payloads
and legacy read compatibility while making a clean source-API break.

The canonical domain language lives in:

- [`src/tsd/io/CONTEXT.md`](../src/tsd/io/CONTEXT.md)
- [`src/tsd/app/CONTEXT.md`](../src/tsd/app/CONTEXT.md)
- [`apps/interactive/scivisStudio/CONTEXT.md`](../apps/interactive/scivisStudio/CONTEXT.md)

## Verb boundaries

| Verb | Boundary |
|---|---|
| `serialize` / `deserialize` | Runtime state to or from a `DataNode` representation |
| `write` / `read` | `DataTree` to or from an in-memory byte buffer |
| `save` / `load` | Archive or Application Dump to or from a file |
| `import` / `export` | Conversion between TSD and a non-native format |

Public Archive functions use `verb_ThingArchive`, put the runtime subject or
target before the carrier, and report failure rather than relying on log-only
`void` operations. The old native `save_Scene`, `load_Scene`, `export_Object`,
`import_Subtree`, and related names are removed without deprecated wrappers.

Low-level component functions use names such as `serialize_Parameter` and
`deserialize_Parameter`; the `Archive` suffix is reserved for complete Archive
representations.

## Archive catalog

| Archive | Content | Deserialization |
|---|---|---|
| Scene | One complete `Scene`, without animation-manager state | Replaces the target Scene |
| Object | One primary object plus its dependency closure | Adds objects; initially Surface and Volume roots only |
| Camera | The complete camera pool | Replaces the target camera pool |
| Renderer | The complete renderer pool | Replaces the target renderer pool |
| Layer Subtree | One subtree plus its object dependency closure, without animations | Adds beneath a required destination parent |
| Animation | One animation whose bindings refer into a compatible Scene | Adds to an Animation Manager |
| Animation Manager | Timeline configuration plus all owned Animation Archives | Replaces the target manager after its compatible Scene exists |

Whole-owner Archives replace state; element Archives add state. Additive
deserialization rolls back resources it created on failure. Replacement
deserialization validates before clearing existing state. Application Dump
deserialization additionally stages its composed Scene and Animation Manager
Archives against a temporary compatible pair before replacing live TSD state.
A strong atomic swap of an entire Context is outside this refactor.

Animation Manager Archives persist time, increment, total frames, FPS, loop
mode, and owned animations. They do not persist active playback, the playback
accumulator, or transient application state, and they restore stopped.

Scene Archive serialization is observational and accepts a `const Scene&`.
It builds Archive-local dense mappings rather than defragmenting or otherwise
mutating the live Scene. Its only lossy option is an explicit array-data policy
with `IncludeData` and `ProxyOnly`; subtree exclusion is not a Scene Archive
option.

## Source organization

```text
src/tsd/io/
├── archives.hpp
├── archives/
│   ├── SceneArchive.*
│   ├── ObjectArchive.*
│   ├── CameraArchive.*
│   ├── RendererArchive.*
│   ├── LayerSubtreeArchive.*
│   ├── AnimationArchive.*
│   ├── AnimationManagerArchive.*
│   └── detail/
│       ├── ArchiveClosure.*
│       ├── ArchivePlan.*
│       └── AnimationRemap.*
├── serialization.hpp
├── serialization/
│   ├── Parameter.*
│   ├── Object.*
│   └── Layer.*
├── exporters.hpp
├── exporters/
│   ├── SceneToUSD.*
│   ├── StructuredVolumeToNanoVDB.*
│   └── NanoVdbSidecar.*
└── importers/
```

`archives.hpp` is a convenience umbrella over focused public Archive headers.
`serialization.hpp` exposes only reusable component serializers. USD and
NanoVDB export move to `exporters/`; their existing export vocabulary remains
correct.

Archive identity is fixed by its module. The Layer Subtree Archive API always
uses the established layer-subtree identity rather than accepting arbitrary
file-type and schema descriptors. Application-owned Dataset and Light Rig
Archives reuse lower-level subtree-content serialization under their own
envelopes.

`CameraPose` serialization moves from `tsd_io` to the new
`tsd_app/ApplicationDump.*` concern, removing the upward dependency from I/O to
rendering. It remains a compatibility concern and should not drive a larger
abstraction because Camera Poses are expected to disappear.

## Metadata and compatibility

This refactor does not broadly redesign existing Archive schemas. Existing
payload shapes remain readable, metadata stays case-by-case rather than
becoming mandatory, and deliberate write-layout changes are limited to the new
pool Archives, Application Dump composition, Animation Manager loop state, and
the versioned SciVis Studio migration.

- Existing Scene, Object, and Layer Subtree schema strings remain accepted.
- New Camera and Renderer Archives use minimal version-1 schemas
  `tsd.scene.cameras` and `tsd.scene.renderers`.
- The old `tsd.scene.cameras-and-renderers` payload remains readable but is no
  longer written.
- New Scene Archives never write animations. `tsd_app` compatibility code
  restores animations found in legacy scene-plus-animation payloads after the
  Scene has been restored.
- New Layer Subtree Archives never contain animations. Application-owned
  Archives such as SciVis Studio Datasets retain their existing animation
  behavior through higher-level composition.
- Existing payloads without metadata continue through their current legacy
  validation paths. New schema work beyond the two pool Archives is deferred.

## Application Dumps

`tsd_app` owns the stable Context portion of an Application Dump:

- Required Scene and Animation Manager Archives.
- ANARI device-manager settings.
- Offline-render settings.
- Logging settings.
- Camera Poses while they still exist.

Transient command-line state, selections, live device handles, and completion
flags are excluded. The base serializer writes only its named sections and
preserves or ignores unknown siblings so concrete applications can add UI and
domain state without upward dependencies.

New dumps use:

```text
archives/
  scene
  animationManager
```

Both embedded Archives are mandatory, including an explicitly empty Animation
Manager Archive. Deserialization reconstructs the Scene and then the dependent
Animation Manager in staged state. Only successful reconstruction is committed
through the replacement deserializers, preserving the identity and delegate
registrations of the live Scene. Stable Context settings are applied after that
commit, so archive-driven failures leave both TSD state and stable settings
unchanged. Each concrete application continues to own its outer schema and
version; a universal Application Dump schema is deferred.

Generic Lua bindings, tutorials, viewer labels, logs, and CLI routing adopt the
new Archive vocabulary. The `-tsd` flag remains, but native TSD loading leaves
`ImporterType` and routes directly through `tsd_app` Archive loading.

## SciVis Studio migration

SciVis Studio adopts the same vocabulary: native assets are saved and loaded
as Dataset, Camera Rig, and Light Rig Archives; import/reimport is reserved for
foreign source conversion.

The next project schema stops writing the residual Scene Archive under
`context`. Datasets, Light Rigs, and Camera Rigs retain their independently
owned Archives. The complete camera and renderer pools are written to required,
fixed paths:

```text
scene/cameras.tsd
scene/renderers.tsd
```

Both files are written and validated transactionally, even for empty pools. A
missing or invalid pool Archive makes a new-format project invalid. Legacy
projects with an embedded `context` remain readable and migrate to the
decomposed layout on their next explicit save. See
[`ADR 0010`](adr/0010-decompose-studio-project-scene-state.md).

## Implementation sequence

1. Rename `DataTree` buffer operations to `write`/`read`; move foreign
   exporters and add `exporters.hpp`.
2. Extract and rename low-level component serializers without changing their
   representations.
3. Add Archive modules and explicit APIs, including Camera and Renderer
   Archives, compatibility readers, non-mutating Scene serialization, and
   Archive-focused tests.
4. Migrate generic TSD callers, Lua, tutorials, UI labels, logs, and CLI
   routing; remove the old public APIs and `ImporterType::TSD`.
5. Add base Application Dump composition in `tsd_app` and migrate generic
   application state loading with legacy `context` support.
6. Migrate SciVis Studio to its next project schema and required pool Archive
   files while preserving Dataset, Camera Rig, and Light Rig features.

## Verification

- Split the monolithic serialization tests into component-serialization and
  per-Archive suites.
- Exercise every serialize/deserialize and save/load pair, including failure
  returns and additive rollback.
- Test Scene Archive serialization against sparse object pools and verify the
  live Scene is unchanged.
- Test both Scene array-data policies and the network `DataTree` write/read
  path.
- Keep explicit legacy tests for missing metadata, combined Camera/Renderer
  payloads, scene payloads containing animations, old Application Dump
  `context` nodes, and pre-migration SciVis Studio projects.
- Verify new SciVis Studio saves contain no residual Scene Archive, require
  both pool Archive files, and preserve Dataset, rig, shot, camera, renderer,
  and animation behavior after reopening.
- Run the standalone TSD build and focused CTest suites before the full TSD
  test suite.
