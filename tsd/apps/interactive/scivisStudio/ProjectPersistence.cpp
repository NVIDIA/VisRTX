// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ProjectPersistence.h"

#include "CameraRig.h"
#include "DatasetIO.h"
#include "LightRigIO.h"
#include "ProjectSerialization.h"

#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/io/archives/CameraArchive.hpp"
#include "tsd/io/archives/RendererArchive.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"
#include "tsd/scene/Scene.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <utility>

namespace tsd::scivis_studio {

namespace {

bool fail(std::string message, std::string *error)
{
  if (error)
    *error = std::move(message);
  return false;
}

bool assetNamesEqual(const std::string &a, const std::string &b)
{
  if (a.size() != b.size())
    return false;
  return std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
    return std::tolower(static_cast<unsigned char>(x))
        == std::tolower(static_cast<unsigned char>(y));
  });
}

template <typename AssetT>
void normalizeAssetNames(std::vector<AssetT> &assets)
{
  std::vector<std::string> assigned;
  for (auto &asset : assets) {
    const std::string base = sanitizeRigName(asset.name);
    std::string candidate = base;
    for (int n = 2; std::any_of(assigned.begin(),
             assigned.end(),
             [&](const std::string &a) {
               return assetNamesEqual(a, candidate);
             });
         ++n)
      candidate = base + " (" + std::to_string(n) + ")";
    asset.name = candidate;
    assigned.push_back(candidate);
  }
}

bool validateAssetMetadata(const std::filesystem::path &file,
    const char *expectedFileType,
    const char *expectedSchema,
    std::string *error)
{
  tsd::core::DataTree tree;
  if (!tree.load(file.string().c_str()))
    return fail("failed to reopen staged file", error);

  auto metadata = tsd::core::readDataTreeMetadata(tree.root());
  if (!metadata.found()) {
    return fail(
        metadata.malformed() ? metadata.message : "missing metadata", error);
  }
  const auto &m = *metadata.metadata;
  if (m.fileType != expectedFileType || m.schema != expectedSchema)
    return fail("unexpected asset metadata", error);
  return true;
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

tsd::scene::LayerNodeRef resolveNode(
    const tsd::scene::Scene &scene, const SceneNodeRef &ref)
{
  if (ref.layerName.empty() || ref.nodeIndex == TSD_INVALID_INDEX)
    return {};
  auto *layer = scene.layer(ref.layerName.c_str());
  return layer ? layer->at(ref.nodeIndex) : tsd::scene::LayerNodeRef{};
}

tsd::scene::LayerNodeRef resolveProjectAssetRoot(const tsd::scene::Scene &scene,
    const char *collection,
    const std::string &id,
    const SceneNodeRef &fallback)
{
  if (auto *layer = scene.layer("studio")) {
    auto collectionRoot = findDirectChild(layer->root(), collection);
    if (auto assetRoot = findDirectChild(collectionRoot, id))
      return assetRoot;
  }
  return resolveNode(scene, fallback);
}

using PoolArchiveSaver = bool (*)(const tsd::scene::Scene &, const char *);
using PoolArchiveValidator = tsd::io::ArchiveValidationResult (*)(
    tsd::core::DataNode &);

ProjectAssetWrite makePoolArchiveWrite(const tsd::scene::Scene &scene,
    bool replaceExisting,
    const char *description,
    const std::filesystem::path &target,
    PoolArchiveSaver saveArchive,
    PoolArchiveValidator validateArchive,
    std::string_view expectedSchema)
{
  ProjectAssetWrite write;
  write.description = description;
  write.target = target;
  if (replaceExisting)
    write.ownedTarget = target;
  write.writer = [&scene, saveArchive, description](
                     const std::filesystem::path &file,
                     std::string *writeError) {
    if (saveArchive(scene, file.string().c_str()))
      return true;
    return fail(std::string("failed to save ") + description, writeError);
  };
  write.validator = [validateArchive, expectedSchema](
                        const std::filesystem::path &file,
                        std::string *validationError) {
    tsd::core::DataTree archive;
    if (!archive.load(file.string().c_str()))
      return fail("failed to load staged Archive", validationError);

    const auto metadata = tsd::core::readDataTreeMetadata(archive.root());
    const auto validation = validateArchive(archive.root());
    if (metadata.found() && metadata.metadata->schema == expectedSchema
        && validation.accepted()) {
      return true;
    }
    return fail(validation.message.empty() ? "Archive has unexpected metadata"
                                           : validation.message,
        validationError);
  };
  return write;
}

bool isSavingCurrentProject(
    const Project &project, const std::filesystem::path &directory)
{
  if (project.projectDirectory.empty())
    return false;
  if (directory.lexically_normal()
      == project.projectDirectory.lexically_normal()) {
    return true;
  }
  std::error_code ec;
  return std::filesystem::equivalent(directory, project.projectDirectory, ec)
      && !ec;
}

void addRemoval(ProjectSavePlan &plan, const std::filesystem::path &path)
{
  const bool alreadyAdded = std::any_of(
      plan.removals.begin(), plan.removals.end(), [&](const auto &removal) {
        return assetNamesEqual(removal.generic_string(), path.generic_string());
      });
  if (!alreadyAdded)
    plan.removals.push_back(path);
}

} // namespace

ProjectSaveRequest::ProjectSaveRequest(const Project &project,
    const tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    std::filesystem::path directory)
    : project(project),
      scene(scene),
      animationManager(animationManager),
      directory(std::move(directory))
{}

bool buildProjectSavePlan(const ProjectSaveRequest &request,
    ProjectSaveResult &result,
    std::string *error)
{
  const bool savingCurrent =
      isSavingCurrentProject(request.project, request.directory);
  const auto manifest = request.directory / PROJECT_MANIFEST_FILENAME;
  if (std::filesystem::exists(manifest) && !savingCurrent)
    return fail("target directory already contains project.tsd", error);

  if (!savingCurrent) {
    std::vector<std::string> unavailable;
    for (const auto &dataset : request.project.datasets) {
      if (dataset.status != DatasetStatus::Available)
        unavailable.push_back(dataset.name);
    }
    if (!unavailable.empty()) {
      std::string message = "Save As requires every dataset to be available:";
      for (const auto &name : unavailable)
        message += "\n- " + name;
      return fail(std::move(message), error);
    }
  }

  result = {};
  result.project = request.project;
  result.project.name =
      (request.project.name.empty() || request.project.name == "Untitled")
      ? (request.directory.filename().string().empty()
                ? std::string("Untitled")
                : request.directory.filename().string())
      : request.project.name;
  result.project.projectDirectory = request.directory;
  normalizeAssetNames(result.project.datasets);
  normalizeAssetNames(result.project.cameraRigs);
  normalizeAssetNames(result.project.lightRigs);
  result.project.markClean();

  auto &plan = result.plan;
  plan.directory = request.directory;
  plan.directories = {"renders", "datasets", "cameras", "lights", "scene"};
  plan.assets.push_back(makePoolArchiveWrite(request.scene,
      savingCurrent,
      "Camera pool Archive",
      std::filesystem::path("scene") / "cameras.tsd",
      tsd::io::save_CameraArchive,
      tsd::io::validate_CameraArchive,
      tsd::io::schema::SCENE_CAMERAS));
  plan.assets.push_back(makePoolArchiveWrite(request.scene,
      savingCurrent,
      "Renderer pool Archive",
      std::filesystem::path("scene") / "renderers.tsd",
      tsd::io::save_RendererArchive,
      tsd::io::validate_RendererArchive,
      tsd::io::schema::SCENE_RENDERERS));

  for (size_t i = 0; i < result.project.datasets.size(); ++i) {
    auto &savedDataset = result.project.datasets[i];
    const auto &liveDataset = request.project.datasets[i];
    const bool needsWrite = !savingCurrent || liveDataset.dirty
        || liveDataset.pendingExtraction
        || liveDataset.persistedName != savedDataset.name;
    if (!needsWrite)
      continue;
    if (savedDataset.status != DatasetStatus::Available) {
      return fail("dataset '" + savedDataset.name
              + "' is unavailable and cannot be saved",
          error);
    }
    auto datasetRoot = resolveProjectAssetRoot(
        request.scene, "datasets", liveDataset.id, liveDataset.rootNode);
    if (!datasetRoot) {
      return fail(
          "dataset '" + savedDataset.name + "' has no scene subtree", error);
    }

    savedDataset.persistedName = savedDataset.name;
    savedDataset.pendingExtraction = false;
    savedDataset.dirty = false;
    ProjectAssetWrite write;
    write.description = "dataset '" + savedDataset.name + "'";
    write.target =
        std::filesystem::path("datasets") / (savedDataset.name + ".tsd");
    if (savingCurrent && !liveDataset.persistedName.empty()) {
      write.ownedTarget = std::filesystem::path("datasets")
          / (liveDataset.persistedName + ".tsd");
    }
    const auto dataset = savedDataset;
    auto *animationManager = &request.animationManager;
    write.writer = [dataset, datasetRoot, animationManager](
                       const std::filesystem::path &file,
                       std::string *writeError) {
      return saveDatasetArchiveFile(
          dataset, datasetRoot, *animationManager, file, writeError);
    };
    const auto expectedName = savedDataset.name;
    write.validator = [expectedName](const std::filesystem::path &file,
                          std::string *validationError) {
      auto validation = validateDatasetAsset(file);
      if (!validation.ok)
        return fail(validation.error, validationError);
      if (validation.dataset.name != expectedName)
        return fail("dataset name does not match target", validationError);
      return true;
    };
    plan.assets.push_back(std::move(write));
  }

  for (size_t i = 0; i < result.project.cameraRigs.size(); ++i) {
    auto &savedRig = result.project.cameraRigs[i];
    const auto &liveRig = request.project.cameraRigs[i];
    savedRig.persistedName = savedRig.name;

    ProjectAssetWrite write;
    write.description = "camera rig '" + savedRig.name + "'";
    write.target = std::filesystem::path("cameras") / (savedRig.name + ".tsd");
    if (savingCurrent && !liveRig.persistedName.empty()) {
      write.ownedTarget =
          std::filesystem::path("cameras") / (liveRig.persistedName + ".tsd");
    }
    const auto rig = savedRig;
    write.writer = [rig](const std::filesystem::path &file,
                       std::string *writeError) {
      return camera_rig::saveCameraRigArchiveFile(rig, file, writeError);
    };
    const auto expectedName = savedRig.name;
    write.validator = [expectedName](const std::filesystem::path &file,
                          std::string *validationError) {
      CameraRig staged;
      if (!camera_rig::loadCameraRigArchiveFile(file, staged, validationError))
        return false;
      if (staged.name != expectedName)
        return fail("camera rig name does not match target", validationError);
      return true;
    };
    plan.assets.push_back(std::move(write));
  }

  for (size_t i = 0; i < result.project.lightRigs.size(); ++i) {
    auto &savedRig = result.project.lightRigs[i];
    const auto &liveRig = request.project.lightRigs[i];
    auto rigRoot = resolveProjectAssetRoot(
        request.scene, "lightRigs", liveRig.id, liveRig.rootNode);
    if (!rigRoot)
      return fail("light rig '" + savedRig.name + "' has no scene node", error);
    savedRig.persistedName = savedRig.name;

    ProjectAssetWrite write;
    write.description = "light rig '" + savedRig.name + "'";
    write.target = std::filesystem::path("lights") / (savedRig.name + ".tsd");
    if (savingCurrent && !liveRig.persistedName.empty()) {
      write.ownedTarget =
          std::filesystem::path("lights") / (liveRig.persistedName + ".tsd");
    }
    const auto expectedName = savedRig.name;
    write.writer = [rigRoot, expectedName](const std::filesystem::path &file,
                       std::string *writeError) {
      if (saveLightRigArchiveFile(rigRoot, file, expectedName))
        return true;
      return fail(
          "failed to serialize light rig (see log for details)", writeError);
    };
    write.validator = [](const std::filesystem::path &file,
                          std::string *validationError) {
      tsd::core::DataTree archive;
      if (!archive.load(file.string().c_str()))
        return fail("failed to reopen staged file", validationError);
      const auto validation = validateLightRigArchive(archive.root());
      if (!validation.accepted())
        return fail(validation.message, validationError);
      return true;
    };
    plan.assets.push_back(std::move(write));
  }

  auto manifestTree = std::make_shared<tsd::core::DataTree>();
  auto &root = manifestTree->root();
  tsd::core::writeDataTreeMetadata(root,
      {tsd::core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          PROJECT_FILE_TYPE,
          PROJECT_SCHEMA,
          SCHEMA_VERSION});
  projectToNode(result.project, root["scivisStudio"]);
  if (request.windows)
    root["windows"] = *request.windows;
  if (!request.layout.empty())
    root["layout"] = request.layout;
  if (request.settings)
    root["settings"] = *request.settings;

  plan.manifest.description = "project manifest";
  plan.manifest.target = PROJECT_MANIFEST_FILENAME;
  if (savingCurrent)
    plan.manifest.ownedTarget = PROJECT_MANIFEST_FILENAME;
  plan.manifest.writer = [manifestTree](const std::filesystem::path &file,
                             std::string *writeError) {
    if (manifestTree->save(file.string().c_str()))
      return true;
    return fail("failed to serialize project manifest", writeError);
  };
  plan.manifest.validator = [](const std::filesystem::path &file,
                                std::string *validationError) {
    return validateAssetMetadata(
        file, PROJECT_FILE_TYPE, PROJECT_SCHEMA, validationError);
  };

  if (savingCurrent) {
    for (const auto &removal : request.pendingAssetRemovals)
      addRemoval(plan, removal);
    for (size_t i = 0; i < result.project.datasets.size(); ++i) {
      const auto &oldName = request.project.datasets[i].persistedName;
      if (!oldName.empty()
          && !assetNamesEqual(oldName, result.project.datasets[i].name)) {
        addRemoval(
            plan, std::filesystem::path("datasets") / (oldName + ".tsd"));
      }
    }
    for (size_t i = 0; i < result.project.cameraRigs.size(); ++i) {
      const auto &oldName = request.project.cameraRigs[i].persistedName;
      if (!oldName.empty()
          && !assetNamesEqual(oldName, result.project.cameraRigs[i].name)) {
        addRemoval(plan, std::filesystem::path("cameras") / (oldName + ".tsd"));
      }
    }
    for (size_t i = 0; i < result.project.lightRigs.size(); ++i) {
      const auto &oldName = request.project.lightRigs[i].persistedName;
      if (!oldName.empty()
          && !assetNamesEqual(oldName, result.project.lightRigs[i].name)) {
        addRemoval(plan, std::filesystem::path("lights") / (oldName + ".tsd"));
      }
    }
  }

  return true;
}

} // namespace tsd::scivis_studio
