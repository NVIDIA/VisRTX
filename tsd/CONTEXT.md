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

**Dataset Import**:
The incorporation of a portable dataset asset as an independent project-owned
dataset with a fresh Dataset ID.

**Dataset Export**:
The serialization of a project dataset as a portable asset without its
project-local Dataset ID.

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
associated file animation; it is not authoritative stored state.

**Unavailable Dataset**:
A dataset that remains in the inventory and in shot associations but cannot
currently recreate its runtime representation. Restoring or replacing its
asset restores the existing project intent.
_Avoid_: Deleted dataset

**Dataset Discovery**:
An explicit project operation that scans for valid dataset files not yet in the
dataset inventory and presents them for review, preselected for incorporation.
Discovery does not change the inventory until the user confirms the selection.
_Avoid_: Automatic dataset import

**Dataset Candidate**:
A standalone dataset asset found by Dataset Discovery that is not yet in the
dataset inventory. A generic TSD scene or another TSD asset type is not a
dataset candidate.

**File Animation Dataset**:
A dataset that owns an authoritative ordered source-file list and uses it to
recreate one core TSD file animation together with its associated layer
subtree. It does not interpret paths or manage per-frame loading and unloading.
Its portability depends on its recorded paths remaining valid.
_Avoid_: Time-series dataset

**File Animation Source List**:
The authoritative, persisted, ordered opaque path strings owned by a File
Animation Dataset. The runtime file binding implements this list but does not
own it; importer settings belong to the dataset rather than individual entries.

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
