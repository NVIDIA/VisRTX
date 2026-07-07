// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Project.h"
#include "ProjectPersistence.h"

#include "tsd/app/Context.h"
#include "tsd/io/importers.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace tsd::scivis_studio {

struct DatasetDirtyDelegate;

struct FileAnimationDatasetOptions
{
  bool setActiveShotFrameCount{true};
};

struct DatasetCandidate
{
  std::filesystem::path file;
  std::string proposedName;
};

struct ProjectContext
{
  ProjectContext() = default;
  explicit ProjectContext(tsd::app::Context *ctx);
  ~ProjectContext();

  void setAppContext(tsd::app::Context *ctx);
  tsd::app::Context *appContext() const;

  Project &project();
  const Project &project() const;

  void createUnsavedProject();
  bool addShot(const std::string &name = "");
  Dataset *addStaticDataset(const std::string &name,
      const std::filesystem::path &sourcePath,
      tsd::io::ImporterType importerType);
  Dataset *addFileAnimationDataset(const std::string &name,
      const std::vector<std::filesystem::path> &sourcePaths,
      tsd::io::ImporterType importerType,
      const FileAnimationDatasetOptions &options = {});
  // Create a Declared Dataset from its Source List alone: no source file is
  // read — not even an existence check — so declaring behaves identically on
  // every machine (ADR 0014). The dataset records Unloaded residency; the
  // first successful Dataset Load materializes it.
  Dataset *addDeclaredFileAnimationDataset(const std::string &name,
      const std::vector<std::string> &sourceList,
      tsd::io::ImporterType importerType,
      const FileAnimationDatasetOptions &options = {});
  bool renameDataset(const DatasetID &id,
      const std::string &newName,
      std::string *error = nullptr);
  bool removeDataset(const DatasetID &id,
      bool keepAssetFile = false,
      std::string *error = nullptr);
  bool reimportStaticDataset(const DatasetID &id, std::string *error = nullptr);
  // Dataset Load/Unload flip an inventory dataset's residency. Unload requires
  // a clean dataset and never touches disk; Load recreates the runtime from
  // the managed asset and changes nothing when it fails. Both are no-ops when
  // the dataset already has the requested residency.
  bool loadDataset(const DatasetID &id, std::string *error = nullptr);
  bool unloadDataset(const DatasetID &id, std::string *error = nullptr);
  // Cheap on-demand availability hint for an Unloaded dataset: a missing
  // asset file is definitively Unavailable. The check never upgrades status —
  // the authoritative assessment is the load attempt itself.
  void refreshUnloadedDatasetAvailability(Dataset &dataset) const;
  bool saveDatasetArchive(const DatasetID &id,
      const std::filesystem::path &file,
      std::string *error = nullptr);
  Dataset *loadDatasetArchive(
      const std::filesystem::path &file, std::string *error = nullptr);
  std::vector<DatasetCandidate> discoverDatasetCandidates() const;
  Dataset *incorporateDatasetCandidate(const DatasetCandidate &candidate,
      const std::string &name,
      std::string *error = nullptr);
  void applyActiveShot();
  void syncAnimationManagerToActiveShot();

  bool saveProject(const std::filesystem::path &directory,
      tsd::core::DataNode *windows = nullptr,
      const std::string &layout = "",
      tsd::core::DataNode *settings = nullptr,
      std::string *error = nullptr);
  bool openProject(const std::filesystem::path &directory,
      tsd::core::DataNode *windowsOut = nullptr,
      std::string *layoutOut = nullptr,
      tsd::core::DataNode *settingsOut = nullptr,
      std::string *error = nullptr,
      const ProjectOpenOptions &options = {});

  tsd::scene::LayerNodeRef resolve(const SceneNodeRef &ref) const;
  tsd::scene::Object *resolve(const SceneObjectRef &ref) const;
  SceneNodeRef refFor(
      const std::string &layerName, tsd::scene::LayerNodeRef ref) const;
  tsd::scene::LayerNodeRef resolveDatasetRoot(Dataset &dataset);
  tsd::scene::LayerNodeRef resolveLightRigRoot(LightRig &rig);
  tsd::scene::Object *resolveShotCamera(Shot &shot);
  LightRig *createLightRig(const std::string &name = "");
  LightRig *cloneLightRig(const LightRigID &id);
  bool removeLightRig(const LightRigID &id);
  // In-memory rename; rejects (returns false + error) an invalid format or a
  // name already used case-insensitively by another rig in the same collection.
  bool renameLightRig(const LightRigID &id,
      const std::string &newName,
      std::string *error = nullptr);
  bool renameCameraRig(const CameraRigID &id,
      const std::string &newName,
      std::string *error = nullptr);
  tsd::scene::LayerNodeRef addLightToRig(
      LightRig &rig, const std::string &subtype);
  bool removeLightFromRig(LightRig &rig, tsd::scene::LayerNodeRef lightNode);
  int shotUseCount(const LightRigID &id) const;
  CameraRig *createCameraRig(const std::string &name = "");
  bool removeCameraRig(const CameraRigID &id);
  int cameraRigUseCount(const CameraRigID &id) const;
  CameraRig *activeShotCameraRig();

  // Standalone rig Archive IO. Save writes the named rig to a .tsd file; Load
  // adds a new library entry (with a fresh id and a de-duplicated name) and
  // never alters shot bindings. Load returns the new rig, or nullptr on error.
  bool saveCameraRigArchive(const CameraRigID &id,
      const std::filesystem::path &file,
      std::string *error = nullptr);
  CameraRig *loadCameraRigArchive(
      const std::filesystem::path &file, std::string *error = nullptr);
  bool saveLightRigArchive(const LightRigID &id,
      const std::filesystem::path &file,
      std::string *error = nullptr);
  LightRig *loadLightRigArchive(
      const std::filesystem::path &file, std::string *error = nullptr);

 private:
  friend struct DatasetDirtyDelegate;
  // Shot semantics shared by eager and declared file-animation creates: bind
  // the new dataset to every shot, enabled only in the active one, and drive
  // the active shot's frame count from the source-list length.
  void applyFileAnimationShotSemantics(const Dataset &record,
      size_t frameCount,
      const FileAnimationDatasetOptions &options);
  void installDatasetDirtyDelegate();
  void markDatasetDirtyForObject(const tsd::scene::Object *object);
  Dataset *loadDatasetArchiveImpl(const std::filesystem::path &file,
      const std::string &name,
      bool alreadyManaged,
      std::string *error);
  tsd::scene::LayerNodeRef ensureChild(
      tsd::scene::LayerNodeRef parent, const char *name);
  tsd::scene::LayerNodeRef ensureStudioRoot();
  tsd::scene::LayerNodeRef ensureDatasetsRoot();
  tsd::scene::LayerNodeRef ensureShotsRoot();
  tsd::scene::LayerNodeRef ensureLightRigsRoot();
  void resetScene();
  void ensureRendererDefaults(Shot &shot);
  LightRig *ensureDefaultLightRig();
  CameraRig *ensureDefaultCameraRig();
  void installAnimationManagerCallback();
  void updateActiveShotFromAnimationTime();

  tsd::app::Context *m_ctx{nullptr};
  Project m_project;
  std::vector<std::filesystem::path> m_pendingAssetRemovals;
  bool m_syncingAnimationManager{false};
  // Residency operations rebuild or tear down whole dataset subtrees; the
  // per-object dirty tracking is meaningless (and O(n^2)) while they run.
  bool m_mutatingDatasetRuntime{false};
  tsd::scene::BaseUpdateDelegate *m_datasetDirtyDelegate{nullptr};
};

const char *toString(tsd::io::ImporterType importerType);
tsd::io::ImporterType importerTypeFromString(const std::string &s);

} // namespace tsd::scivis_studio
