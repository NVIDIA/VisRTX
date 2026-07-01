// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ProjectPersistence.h"

#include "CameraRig.h"
#include "DatasetIO.h"
#include "LightRigIO.h"
#include "ProjectSerialization.h"

#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/archives/CameraArchive.hpp"
#include "tsd/io/archives/RendererArchive.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/Camera.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace tsd::scivis_studio {

namespace {

struct StagedArchive
{
  std::shared_ptr<tsd::core::DataTree> tree;
  std::string error;
};

struct StagedCameraRig
{
  bool loaded{false};
  CameraRig rig;
  std::string error;
};

bool fail(std::string message, std::string *error)
{
  if (error)
    *error = std::move(message);
  return false;
}

StagedArchive stageArchive(const std::filesystem::path &file)
{
  StagedArchive staged;
  staged.tree = std::make_shared<tsd::core::DataTree>();
  if (!staged.tree->load(file.string().c_str())) {
    staged.tree.reset();
    staged.error = "Archive is missing or unreadable";
  }
  return staged;
}

tsd::scene::LayerNodeRef findDirectChild(
    tsd::scene::LayerNodeRef parent, const std::string &name)
{
  if (!parent)
    return {};
  auto child = parent->next();
  while (child && child != parent) {
    if ((*child)->name() == name)
      return child;
    child = child->sibling();
  }
  return {};
}

tsd::scene::LayerNodeRef ensureChild(
    tsd::scene::Scene &scene, tsd::scene::LayerNodeRef parent, const char *name)
{
  if (auto found = findDirectChild(parent, name))
    return found;
  return scene.insertChildNode(parent, name);
}

tsd::scene::LayerNodeRef ensureStudioRoot(tsd::scene::Scene &scene)
{
  auto *layer = scene.addLayer("studio");
  return layer ? layer->root() : tsd::scene::LayerNodeRef{};
}

tsd::scene::LayerNodeRef ensureCollectionRoot(
    tsd::scene::Scene &scene, const char *name)
{
  return ensureChild(scene, ensureStudioRoot(scene), name);
}

SceneNodeRef nodeRef(const char *layerName, tsd::scene::LayerNodeRef node)
{
  return {layerName, node ? node.index() : TSD_INVALID_INDEX};
}

tsd::scene::LayerNodeRef resolveNode(
    tsd::scene::Scene &scene, const SceneNodeRef &ref)
{
  if (ref.layerName.empty() || ref.nodeIndex == TSD_INVALID_INDEX)
    return {};
  auto *layer = scene.layer(ref.layerName.c_str());
  return layer ? layer->at(ref.nodeIndex) : tsd::scene::LayerNodeRef{};
}

tsd::scene::LayerNodeRef resolveAssetRoot(tsd::scene::Scene &scene,
    const char *collection,
    const std::string &id,
    SceneNodeRef &fallback)
{
  if (auto *layer = scene.layer("studio")) {
    auto collectionRoot = findDirectChild(layer->root(), collection);
    if (auto assetRoot = findDirectChild(collectionRoot, id)) {
      fallback = nodeRef("studio", assetRoot);
      return assetRoot;
    }
  }
  return resolveNode(scene, fallback);
}

void resetScene(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager)
{
  animationManager.removeAllAnimations();
  scene.removeAllObjects();
  scene.defaultMaterial();
  scene.defaultCamera();
}

void clearCameraRigBindings(Project &project, const CameraRigID &cameraRigId)
{
  for (auto &shot : project.shots) {
    if (shot.cameraRigId == cameraRigId)
      shot.cameraRigId.clear();
  }
}

void clearLightRigBindings(Project &project, const LightRigID &lightRigId)
{
  for (auto &shot : project.shots) {
    if (shot.lightRigId == lightRigId)
      shot.lightRigId.clear();
  }
}

void migrateLegacyShotLights(Project &project, tsd::scene::Scene &scene)
{
  if (!project.lightRigs.empty())
    return;
  auto *layer = scene.layer("studio");
  if (!layer)
    return;

  auto lightRigsRoot = ensureCollectionRoot(scene, "lightRigs");
  auto shotsRoot = findDirectChild(layer->root(), "shots");
  for (auto &shot : project.shots) {
    auto shotRoot = findDirectChild(shotsRoot, shot.id);
    auto legacyLights = findDirectChild(shotRoot, "lights");
    if (!legacyLights)
      continue;

    LightRig rig;
    rig.id = light_rig::nextLightRigId(project);
    rig.name =
        shot.name.empty() ? (shot.id + " Lights") : (shot.name + " Lights");
    if (auto existing = findDirectChild(lightRigsRoot, rig.id))
      scene.removeNode(existing, true);
    legacyLights->container()->move_subtree(legacyLights, lightRigsRoot);
    (*legacyLights)->name() = rig.id;
    rig.rootNode = nodeRef("studio", legacyLights);
    shot.lightRigId = rig.id;
    project.lightRigs.push_back(std::move(rig));
  }
  scene.signalLayerStructureChanged(layer);
}

void hydrateCameraRigs(
    const detail::ProjectOpenState &state, Project &project, bool logWarnings);
void hydrateLightRigs(const detail::ProjectOpenState &state,
    Project &project,
    tsd::scene::Scene &scene,
    bool logWarnings);
void hydrateDatasets(const detail::ProjectOpenState &state,
    Project &project,
    tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    bool logWarnings);

void refreshRuntimeRefs(Project &project, tsd::scene::Scene &scene)
{
  for (auto &dataset : project.datasets)
    resolveAssetRoot(scene, "datasets", dataset.id, dataset.rootNode);
  for (auto &rig : project.lightRigs)
    resolveAssetRoot(scene, "lightRigs", rig.id, rig.rootNode);

  for (auto &shot : project.shots) {
    const auto cameraName = shot.id + "_camera";
    const auto &cameras = scene.objectDB().camera;
    tsd::core::foreach_item_const(
        cameras, [&](const tsd::scene::Camera *camera) {
          if (camera && camera->name() == cameraName)
            shot.camera = {ANARI_CAMERA, camera->index()};
        });
  }
}

bool reconstructProject(const detail::ProjectOpenState &state,
    Project &project,
    tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    bool logWarnings,
    std::string *error);

} // namespace

namespace detail {

struct ProjectOpenState
{
  std::filesystem::path directory;
  std::shared_ptr<tsd::core::DataTree> manifest;
  Project manifestProject;
  int schemaVersion{0};
  StagedArchive cameras;
  StagedArchive renderers;
  std::vector<StagedCameraRig> cameraRigs;
  std::vector<StagedArchive> lightRigs;
  std::vector<StagedArchive> datasets;
};

} // namespace detail

namespace {

void hydrateCameraRigs(
    const detail::ProjectOpenState &state, Project &project, bool logWarnings)
{
  std::vector<CameraRig> kept;
  kept.reserve(project.cameraRigs.size());
  for (size_t i = 0; i < project.cameraRigs.size(); ++i) {
    auto &rig = project.cameraRigs[i];
    const auto &staged = state.cameraRigs[i];
    if (!staged.loaded) {
      if (logWarnings) {
        tsd::core::logWarning(
            "[SciVisStudio] Skipping Camera Rig Archive '%s': %s",
            rig.name.c_str(),
            staged.error.c_str());
      }
      clearCameraRigBindings(project, rig.id);
      continue;
    }
    rig.current = staged.rig.current;
    rig.keyframes = staged.rig.keyframes;
    rig.persistedName = rig.name;
    kept.push_back(std::move(rig));
  }
  project.cameraRigs = std::move(kept);
}

void hydrateLightRigs(const detail::ProjectOpenState &state,
    Project &project,
    tsd::scene::Scene &scene,
    bool logWarnings)
{
  std::vector<LightRig> kept;
  kept.reserve(project.lightRigs.size());
  for (size_t i = 0; i < project.lightRigs.size(); ++i) {
    auto &rig = project.lightRigs[i];
    const auto &staged = state.lightRigs[i];
    tsd::scene::LayerNodeRef root;
    if (staged.tree) {
      auto destination = ensureCollectionRoot(scene, "lightRigs");
      std::string displayName;
      root = deserializeLightRigArchive(
          scene, staged.tree->root(), destination, &displayName);
    }
    if (!root) {
      if (logWarnings) {
        tsd::core::logWarning(
            "[SciVisStudio] Skipping Light Rig Archive '%s': %s",
            rig.name.c_str(),
            staged.error.empty() ? "failed to load Archive"
                                 : staged.error.c_str());
      }
      clearLightRigBindings(project, rig.id);
      continue;
    }
    (*root)->name() = rig.id;
    rig.rootNode = nodeRef("studio", root);
    rig.persistedName = rig.name;
    kept.push_back(std::move(rig));
  }
  project.lightRigs = std::move(kept);
}

void hydrateDatasets(const detail::ProjectOpenState &state,
    Project &project,
    tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    bool logWarnings)
{
  auto destination = ensureCollectionRoot(scene, "datasets");
  for (size_t i = 0; i < project.datasets.size(); ++i) {
    auto &inventoryEntry = project.datasets[i];
    const auto &staged = state.datasets[i];
    Dataset loaded;
    tsd::scene::LayerNodeRef loadedRoot;
    std::string datasetError = staged.error;
    const bool loadedAsset = staged.tree
        && deserializeDatasetArchive(scene,
            animationManager,
            staged.tree->root(),
            destination,
            loaded,
            loadedRoot,
            &datasetError);
    if (!loadedAsset) {
      inventoryEntry.status = DatasetStatus::Unavailable;
      inventoryEntry.dirty = false;
      inventoryEntry.persistedName = inventoryEntry.name;
      if (logWarnings) {
        tsd::core::logWarning("[SciVisStudio] Dataset '%s' is unavailable: %s",
            inventoryEntry.name.c_str(),
            datasetError.c_str());
      }
      continue;
    }

    if (loaded.name != inventoryEntry.name) {
      removeDatasetRuntime(scene, animationManager, loadedRoot);
      inventoryEntry.status = DatasetStatus::Unavailable;
      inventoryEntry.dirty = false;
      inventoryEntry.persistedName = inventoryEntry.name;
      if (logWarnings) {
        tsd::core::logWarning(
            "[SciVisStudio] Dataset '%s' is unavailable: asset name is '%s'",
            inventoryEntry.name.c_str(),
            loaded.name.c_str());
      }
      continue;
    }

    loaded.id = inventoryEntry.id;
    loaded.name = inventoryEntry.name;
    loaded.persistedName = inventoryEntry.name;
    loaded.dirty = false;
    (*loadedRoot)->name() = loaded.id;
    loaded.rootNode = nodeRef("studio", loadedRoot);
    inventoryEntry = std::move(loaded);
  }
}

bool reconstructProject(const detail::ProjectOpenState &state,
    Project &project,
    tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    bool logWarnings,
    std::string *error)
{
  resetScene(scene, animationManager);
  project = state.manifestProject;

  if (state.schemaVersion >= DECOMPOSED_SCENE_SCHEMA_VERSION) {
    auto shotsRoot = ensureCollectionRoot(scene, "shots");
    ensureCollectionRoot(scene, "datasets");
    ensureCollectionRoot(scene, "lightRigs");
    for (const auto &shot : project.shots)
      ensureChild(scene, shotsRoot, shot.id.c_str());

    if (!state.cameras.tree || !state.renderers.tree
        || !tsd::io::deserialize_CameraArchive(
            scene, state.cameras.tree->root())
        || !tsd::io::deserialize_RendererArchive(
            scene, state.renderers.tree->root())) {
      return fail("failed to load required scene pool Archives", error);
    }
  } else if (auto *context = state.manifest->root().child("context")) {
    if (!tsd::io::detail::tryDeserializeLegacyScenePayload(
            scene, *context, nullptr, &animationManager)) {
      return fail("failed to load legacy project context", error);
    }
  }

  project.projectDirectory = state.directory;
  project.markClean();
  if (state.schemaVersion < 2)
    migrateLegacyShotLights(project, scene);
  if (state.schemaVersion >= 4) {
    hydrateCameraRigs(state, project, logWarnings);
    hydrateLightRigs(state, project, scene, logWarnings);
  }
  if (state.schemaVersion >= 5) {
    hydrateDatasets(state, project, scene, animationManager, logWarnings);
  } else {
    for (auto &dataset : project.datasets) {
      dataset.status =
          resolveAssetRoot(scene, "datasets", dataset.id, dataset.rootNode)
          ? DatasetStatus::Available
          : DatasetStatus::Unavailable;
    }
  }
  refreshRuntimeRefs(project, scene);
  return true;
}

bool stageRequiredPoolArchive(const std::filesystem::path &file,
    StagedArchive &staged,
    tsd::io::ArchiveValidationResult (*validate)(tsd::core::DataNode &),
    std::string *error)
{
  staged = stageArchive(file);
  if (!staged.tree)
    return fail("required Archive is missing: " + file.string(), error);
  const auto validation = validate(staged.tree->root());
  if (!validation.accepted()) {
    return fail("invalid required Archive '" + file.string()
            + "': " + validation.message,
        error);
  }
  return true;
}

} // namespace

bool stageProjectOpen(const std::filesystem::path &directory,
    ProjectOpenStage &stage,
    std::string *error)
{
  stage.m_state.reset();
  stage.project = {};
  stage.ui.root().reset();

  const auto validation = validateProjectRoot(directory);
  if (!validation.ok)
    return fail(validation.error, error);

  auto state = std::make_shared<detail::ProjectOpenState>();
  state->directory = directory;
  state->manifest = std::make_shared<tsd::core::DataTree>();
  if (!state->manifest->load(validation.manifestPath.string().c_str()))
    return fail("failed to load project.tsd", error);

  auto &root = state->manifest->root();
  auto *model = root.child("scivisStudio");
  if (!model)
    return fail("project.tsd is missing scivisStudio section", error);
  nodeToProject(*model, state->manifestProject);

  const auto metadata = tsd::core::readDataTreeMetadata(root);
  state->schemaVersion = metadata.found()
      ? metadata.metadata->schemaVersion
      : root["schemaVersion"].getValueOr<int>(1);

  if (state->schemaVersion >= DECOMPOSED_SCENE_SCHEMA_VERSION) {
    if (!stageRequiredPoolArchive(directory / "scene/cameras.tsd",
            state->cameras,
            tsd::io::validate_CameraArchive,
            error)
        || !stageRequiredPoolArchive(directory / "scene/renderers.tsd",
            state->renderers,
            tsd::io::validate_RendererArchive,
            error)) {
      return false;
    }
  }

  if (state->schemaVersion >= 4) {
    state->cameraRigs.reserve(state->manifestProject.cameraRigs.size());
    for (const auto &rig : state->manifestProject.cameraRigs) {
      StagedCameraRig staged;
      staged.loaded = camera_rig::loadCameraRigArchiveFile(
          directory / "cameras" / (rig.name + ".tsd"),
          staged.rig,
          &staged.error);
      state->cameraRigs.push_back(std::move(staged));
    }
    state->lightRigs.reserve(state->manifestProject.lightRigs.size());
    for (const auto &rig : state->manifestProject.lightRigs) {
      state->lightRigs.push_back(
          stageArchive(directory / "lights" / (rig.name + ".tsd")));
    }
  }
  if (state->schemaVersion >= 5) {
    state->datasets.reserve(state->manifestProject.datasets.size());
    for (const auto &dataset : state->manifestProject.datasets) {
      state->datasets.push_back(
          stageArchive(directory / "datasets" / (dataset.name + ".tsd")));
    }
  }

  if (auto *windows = root.child("windows"))
    stage.ui.root()["windows"] = *windows;
  if (auto *layout = root.child("layout"))
    stage.ui.root()["layout"] = layout->getValueAs<std::string>();
  if (auto *settings = root.child("settings"))
    stage.ui.root()["settings"] = *settings;

  tsd::scene::Scene stagedScene;
  tsd::animation::AnimationManager stagedAnimations(&stagedScene);
  if (!reconstructProject(
          *state, stage.project, stagedScene, stagedAnimations, false, error)) {
    return false;
  }

  stage.m_state = std::move(state);
  return true;
}

bool applyProjectOpen(ProjectOpenStage &stage,
    tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    std::string *error)
{
  if (!stage.m_state)
    return fail("project open has not been staged", error);
  return reconstructProject(
      *stage.m_state, stage.project, scene, animationManager, true, error);
}

} // namespace tsd::scivis_studio
