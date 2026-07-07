# SciVis Studio

SciVis Studio organizes scientific-visualization assets into projects and
combines them into renderable shots.

## Language

**Project**:
An independent workspace that owns its copies of datasets, camera rigs, and
light rigs. Changes to project-owned assets never propagate implicitly to
another project.

**Dataset**:
An authoritative, self-contained visualization asset with no inherent
association with a camera rig or light rig. A dataset is portable as a
standalone asset, subject to a File Animation Dataset's external path
requirements.
_Avoid_: Dataset cache, imported source

**Dataset ID**:
A stable identity assigned by the owning project. Shots refer to datasets by
this identity, which is independent of the dataset's human-readable name.

**Dataset Provenance**:
Descriptive information about a dataset's original source and how it was
imported. Provenance does not make that source a dependency or affect the
dataset's availability. It contains source-path text and dataset-level importer
settings, not cached filesystem metadata or duplicate path forms; a File
Animation Dataset's operational file list is not provenance.
_Avoid_: Dataset source dependency

**Dataset Reimport**:
An explicit, transactional replacement of a Static Dataset's content from its
provenance source. Reimport preserves the dataset's identity, name, and shot
associations; failure leaves the previous content unchanged.
_Avoid_: Automatic refresh

**Dataset Archive**:
A portable native TSD representation of a Dataset without its project-local
Dataset ID. For a File Animation Dataset, the Archive is the dataset file and
its sibling Source List File together.

**Dataset Archive Load**:
The incorporation of a Dataset Archive as an independent project-owned Dataset
with a fresh Dataset ID.
_Avoid_: Dataset import

**Dataset Archive Save**:
The persistence of a project Dataset as a Dataset Archive.
_Avoid_: Dataset export

**Dirty Dataset**:
A dataset whose authoritative asset differs from its last successfully saved
version. Saving unrelated project state does not rewrite a clean dataset.

**Dataset Removal**:
An explicit operation that removes a dataset from the inventory and deletes
the project-owned asset. Keeping the asset file is a separate, explicit choice.

**Static Dataset**:
A dataset whose complete runtime representation is stored in its dataset asset
and requires no external source I/O. It may contain ordinary core TSD
animations, but it does not contain a file animation.

**Dataset Inventory**:
The datasets that explicitly belong to a project. Merely placing a dataset file
in the project directory does not add it to the inventory.

**Dataset Availability**:
The runtime assessment of whether a dataset can currently be loaded. It is
derived from whether the dataset asset can recreate its layer subtree and any
associated file animation; it is not authoritative stored state. Availability
is machine-relative: the same asset can be Available on one machine and
Unavailable on another where its source paths do not resolve.

**Unavailable Dataset**:
A dataset that remains in the inventory and in shot associations but cannot
currently recreate its runtime representation. Restoring or replacing its
asset restores the existing project intent. Unavailability is not always a
fault: a Declared Dataset is expectedly Unavailable on machines where its
source paths do not resolve.
_Avoid_: Deleted dataset, unloaded dataset

**Dataset Residency**:
The user-controlled choice of which runtime representation of a dataset is
currently in the scene: Loaded or Unloaded. Residency is independent of
Dataset Availability: availability assesses whether the asset could recreate
the runtime representation; residency records whether the project has been
asked to. Residency is a workstation memory concern and never alters shot
intent: a final shot render loads every bound dataset regardless of
residency. Future representations (such as a SciVis proxy) would be
additional residency states. Residency records intent, not process state: a
tool that opens a project without building runtime representations neither
changes nor persists a different residency; residency changes only through an
explicit Dataset Load or Dataset Unload.

**Loaded Dataset**:
A dataset whose full runtime representation is currently in the scene.

**Unloaded Dataset**:
A dataset that remains in the inventory and in shot associations but whose
runtime representation has been deliberately removed to reclaim memory. Unlike
an Unavailable Dataset, unloading expresses user intent rather than failure;
an Unloaded Dataset with a valid asset remains Available. An Unloaded
Dataset's dataset file is read-only: operations that would modify or serialize
its scene representation require loading it first, while project bookkeeping
such as shot bindings and Dataset Removal remains available. Its Source List
File is not covered by this rule: it remains externally editable, with edits
taking effect at the next Dataset Load.
_Avoid_: Unavailable dataset, removed dataset

**Dataset Load**:
The recreation of an existing inventory dataset's runtime representation from
its dataset asset, preserving its Dataset ID and shot associations. Distinct
from Dataset Archive Load by direct object: loading an archive incorporates a
new dataset; loading a dataset makes an existing one resident. A failed load
changes nothing: the dataset remains Unloaded and is known to be Unavailable
until its asset is restored.

**Dataset Unload**:
The deliberate removal of a clean dataset's runtime representation while
keeping the dataset in the inventory and its shot associations intact. Unload
never writes to disk and never discards unsaved changes: a Dirty Dataset must
be saved before it can be unloaded.
_Avoid_: Dataset removal

**Dataset Discovery**:
An explicit project operation that scans for valid dataset files not yet in the
dataset inventory and presents them for review, preselected for incorporation.
Discovery does not change the inventory until the user confirms the selection.
_Avoid_: Automatic dataset import

**Dataset Candidate**:
A Dataset Archive found by Dataset Discovery that is not yet in the dataset
inventory. A generic TSD scene or another TSD Archive type is not a
dataset candidate.

**Camera Rig Archive**:
A portable native TSD representation of a camera rig.

**Light Rig Archive**:
A portable native TSD representation of a light rig.

**File Animation Dataset**:
A dataset that owns an authoritative ordered source-file list and uses it to
recreate one core TSD file animation together with its associated layer
subtree. It does not interpret paths or manage per-frame loading and unloading.
Its portability depends on its recorded paths remaining valid.
_Avoid_: Time-series dataset

**File Animation Source List**:
The authoritative ordered path entries owned by a File Animation Dataset and
persisted only in its Source List File. Relative entries are anchored once, at
read, to the Source List File's directory and are opaque thereafter; importer
settings belong to the dataset rather than individual entries.
_Avoid_: Manifest, embedded source list

**Source List File**:
The human-editable sibling text file that persists a File Animation Dataset's
Source List, named after its dataset file. It is part of the dataset asset:
Studio manages it with the dataset, and a missing or empty Source List File
makes the dataset Unavailable. External tools may rewrite it regardless of the
dataset's residency or availability; such edits take effect at the next
Dataset Load. An explicit source-list edit of a dataset that still embeds its
list migrates it to the Source List File form as part of that edit.
_Avoid_: Manifest, frame list

**Declared Dataset**:
A File Animation Dataset created from a Source List alone, without reading any
source files. Its asset records the importer settings and the Source List File
but no scene representation, so it is Unavailable wherever its paths do not
resolve — an expected condition, not a fault. Only a File Animation Dataset
can be declared; a Static Dataset without its content is a contradiction.
_Avoid_: Blind dataset, stub dataset, placeholder dataset

**Dataset Materialization**:
The first successful Dataset Load of a Declared Dataset, which builds its
runtime representation by importing from the Source List. Materialization
marks the dataset dirty; the next save records the scene representation,
making it an ordinary File Animation Dataset. A failed materialization changes
nothing.

**File Animation**:
A derived core TSD runtime animation that loads and unloads individual source
files as its time changes. It is recreated from a File Animation Dataset's
source list and owns per-frame I/O, but not the persisted definition.

**Live Dataset**:
A future dataset driven by a running simulation. Live datasets are outside the
standalone dataset model until concrete use cases establish their lifecycle and
persistence semantics.

**Shot**:
A project-owned composition of datasets, a camera rig, and a light rig. A shot,
rather than any of its constituent assets, owns the associations among them and
the playback timeline that drives their animations.
