// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "DatasetIO.h"

#include "ProjectSerialization.h"

#include "tsd/animation/Animation.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/animation/SpatialFieldFileBinding.hpp"
#include "tsd/io/archives/SubtreeArchiveContent.hpp"
#include "tsd/io/archives/detail/ArchivePlan.hpp"
#include "tsd/scene/objects/Volume.hpp"

#include <algorithm>
#include <vector>

namespace tsd::scivis_studio {

namespace {

const tsd::io::SubtreeArchiveContentDesc DATASET_DESC{
    DATASET_FILE_TYPE, DATASET_SCHEMA, tsd::io::ArchiveObjectPolicy::All};

bool fail(std::string message, std::string *error)
{
  if (error)
    *error = std::move(message);
  return false;
}

void datasetMetadataToNode(const Dataset &dataset, tsd::core::DataNode &node)
{
  node["name"] = dataset.name;
  node["sourceKind"] = dataset::toString(dataset.sourceKind);
  node["importerType"] = dataset.importerType;
  auto &settings = node["importerSettings"];
  for (const auto &setting : dataset.source.importerSettings) {
    auto &entry = settings.append();
    entry["name"] = setting.first;
    entry["value"] = setting.second;
  }

  if (dataset.sourceKind == DatasetSourceKind::Static) {
    auto &provenance = node["provenance"];
    provenance["sourcePath"] = dataset.source.sourcePath;
  } else if (dataset.sourceKind == DatasetSourceKind::FileAnimation) {
    auto &files = node["sourceFiles"];
    for (const auto &source : dataset.sourceFiles)
      files.append() = source.path;
  }
}

bool nodeToDatasetMetadata(
    tsd::core::DataNode &node, Dataset &dataset, std::string *error)
{
  if (node.child("id"))
    return fail("Dataset Archives must not contain a project-local ID", error);

  dataset = {};
  dataset.name = node["name"].getValueOr<std::string>("");
  if (!validateRigName(dataset.name, error))
    return false;

  const auto sourceKind = node["sourceKind"].getValueOr<std::string>("");
  if (sourceKind != "Static" && sourceKind != "FileAnimation")
    return fail("dataset sourceKind must be Static or FileAnimation", error);
  dataset.sourceKind = dataset::sourceKindFromString(sourceKind);

  dataset.importerType = node["importerType"].getValueOr<std::string>("");
  if (dataset.importerType.empty())
    return fail("dataset importerType is required", error);
  if (auto *settings = node.child("importerSettings")) {
    settings->foreach_child([&](tsd::core::DataNode &entry) {
      dataset.source.importerSettings.set(
          entry["name"].getValueOr<std::string>(""),
          entry["value"].getValueOr<std::string>(""));
    });
  }
  if (std::any_of(dataset.source.importerSettings.begin(),
          dataset.source.importerSettings.end(),
          [](const auto &setting) { return setting.first.empty(); }))
    return fail("dataset importer setting names cannot be empty", error);

  if (dataset.sourceKind == DatasetSourceKind::Static) {
    if (auto *provenance = node.child("provenance")) {
      dataset.source.sourcePath =
          (*provenance)["sourcePath"].getValueOr<std::string>("");
    }
    if (auto *files = node.child("sourceFiles"); files && files->numChildren())
      return fail("static datasets cannot contain a source-file list", error);
  } else {
    auto *files = node.child("sourceFiles");
    if (!files || files->numChildren() == 0)
      return fail("file-animation datasets require sourceFiles", error);
    files->foreach_child([&](tsd::core::DataNode &entry) {
      dataset.sourceFiles.push_back({entry.getValueOr<std::string>("")});
    });
    if (std::any_of(dataset.sourceFiles.begin(),
            dataset.sourceFiles.end(),
            [](const DatasetSourceFile &source) {
              return source.path.empty();
            }))
      return fail("file-animation source paths cannot be empty", error);
    if (dataset.importerType != "VOLUME_ANIMATION")
      return fail("unsupported file-animation importerType", error);
  }

  dataset.status = DatasetStatus::Unavailable;
  dataset.dirty = false;
  dataset.persistedName = dataset.name;
  return true;
}

tsd::scene::Volume *findDatasetVolume(tsd::scene::LayerNodeRef root)
{
  if (!root)
    return nullptr;
  tsd::scene::Volume *volume = nullptr;
  auto *layer = (*root).value().layer();
  layer->traverse(root, [&](tsd::scene::LayerNode &node, int) {
    if (!volume && node->isObject() && node->type() == ANARI_VOLUME)
      volume = dynamic_cast<tsd::scene::Volume *>(node->getObject());
    return volume == nullptr;
  });
  return volume;
}

bool recreateFileAnimation(const Dataset &dataset,
    tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    tsd::scene::LayerNodeRef root,
    std::string *error)
{
  auto *volume = findDatasetVolume(root);
  if (!volume)
    return fail("file-animation dataset has no volume target", error);

  auto *field =
      volume->parameterValueAsObject<tsd::scene::SpatialField>("value");
  if (!field)
    return fail("file-animation volume has no initial spatial field", error);

  std::vector<std::string> files;
  files.reserve(dataset.sourceFiles.size());
  for (const auto &source : dataset.sourceFiles)
    files.push_back(source.path);

  auto &animation =
      animationManager.addAnimation(dataset.name + " file animation");
  animation.emplaceFileBinding<tsd::io::SpatialFieldFileBinding>(
      &scene, volume, field->self(), std::move(files));
  return true;
}

} // namespace

DatasetAssetValidationResult validateDatasetAsset(
    const std::filesystem::path &file)
{
  DatasetAssetValidationResult result;
  tsd::core::DataTree tree;
  if (!tree.load(file.string().c_str())) {
    result.error = "failed to load Dataset Archive";
    return result;
  }

  auto subtreeResult =
      tsd::io::validate_SubtreeArchiveContent(tree.root(), DATASET_DESC);
  if (!subtreeResult.accepted()) {
    result.error = subtreeResult.message;
    return result;
  }

  auto *metadata = tree.root().child("dataset");
  if (!metadata) {
    result.error = "Dataset Archive requires dataset metadata";
    return result;
  }
  if (!nodeToDatasetMetadata(*metadata, result.dataset, &result.error))
    return result;

  if (result.dataset.sourceKind == DatasetSourceKind::FileAnimation) {
    auto *volumes = tree.root()["objectDB"].child("volume");
    if (!volumes || volumes->numChildren() != 1) {
      result.error =
          "file-animation datasets require exactly one volume target";
      return result;
    }
  }

  if (auto *animations = tree.root().child("animations")) {
    bool hasPersistedFileBinding = false;
    animations->foreach_child([&](tsd::core::DataNode &animation) {
      if (auto *bindings = animation.child("fileBindings"))
        hasPersistedFileBinding |= bindings->numChildren() != 0;
    });
    if (hasPersistedFileBinding) {
      result.error =
          "Dataset Archives cannot persist derived runtime file bindings";
      return result;
    }
  }

  const auto displayName =
      tree.root()["displayName"].getValueOr<std::string>("");
  const auto subtreeName =
      tree.root()["subtree"]["name"].getValueOr<std::string>("");
  if (displayName != result.dataset.name
      || subtreeName != result.dataset.name) {
    result.error = "dataset name must match displayName and subtree root name";
    return result;
  }

  result.ok = true;
  return result;
}

bool saveDatasetArchiveFile(const Dataset &dataset,
    tsd::scene::LayerNodeRef root,
    tsd::animation::AnimationManager &animationManager,
    const std::filesystem::path &file,
    std::string *error)
{
  if (dataset.sourceKind != DatasetSourceKind::Static
      && dataset.sourceKind != DatasetSourceKind::FileAnimation)
    return fail("unsupported dataset source kind", error);
  if (!validateRigName(dataset.name, error))
    return false;

  auto *scene = (*root).value().layer()->scene();
  tsd::io::ArchivePlanOptions planOptions;
  planOptions.animationManager = &animationManager;
  planOptions.fileBindings = tsd::io::FileBindingArchivePolicy::Omit;
  const auto planResult =
      tsd::io::plan_SubtreeArchive(*scene, root, planOptions);
  if (!planResult.accepted())
    return fail(planResult.message, error);

  size_t ownedFileAnimations = 0;
  for (const auto index : planResult.plan.ownedAnimations) {
    if (!animationManager.animations()[index].fileBindings().empty())
      ++ownedFileAnimations;
  }
  if (dataset.sourceKind == DatasetSourceKind::Static
      && ownedFileAnimations != 0)
    return fail("static datasets cannot own file animations", error);
  if (dataset.sourceKind == DatasetSourceKind::FileAnimation
      && ownedFileAnimations != 1)
    return fail(
        "file-animation datasets must own one runtime file animation", error);

  tsd::io::SubtreeArchiveContentOptions options;
  options.animationManager = &animationManager;
  options.fileBindings = tsd::io::FileBindingArchivePolicy::Omit;
  tsd::core::DataTree tree;
  if (!tsd::io::serialize_SubtreeArchiveContent(
          root, tree.root(), DATASET_DESC, dataset.name, options))
    return fail("failed to serialize dataset subtree", error);

  datasetMetadataToNode(dataset, tree.root()["dataset"]);
  tree.root()["subtree"]["name"] = dataset.name;
  if (!tree.save(file.string().c_str()))
    return fail("failed to write dataset metadata", error);

  auto validation = validateDatasetAsset(file);
  if (!validation.ok)
    return fail("dataset validation failed: " + validation.error, error);
  return true;
}

bool loadDatasetArchiveFile(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    const std::filesystem::path &file,
    tsd::scene::LayerNodeRef destinationParent,
    Dataset &datasetOut,
    tsd::scene::LayerNodeRef &rootOut,
    std::string *error)
{
  auto validation = validateDatasetAsset(file);
  if (!validation.ok)
    return fail(validation.error, error);

  tsd::core::DataTree tree;
  if (!tree.load(file.string().c_str()))
    return fail("failed to load Dataset Archive", error);

  tsd::io::SubtreeArchiveContentOptions options;
  options.animationManager = &animationManager;
  auto loadedContent = tsd::io::deserialize_SubtreeArchiveContent(
      scene, tree.root(), destinationParent, DATASET_DESC, nullptr, options);
  if (!loadedContent.valid() || !loadedContent.root)
    return fail("failed to reconstruct dataset subtree", error);

  if (validation.dataset.sourceKind == DatasetSourceKind::FileAnimation
      && !recreateFileAnimation(validation.dataset,
          scene,
          animationManager,
          loadedContent.root,
          error)) {
    tsd::io::rollback_SubtreeArchiveContent(
        scene, animationManager, loadedContent);
    return false;
  }

  datasetOut = std::move(validation.dataset);
  datasetOut.status = DatasetStatus::Available;
  rootOut = loadedContent.root;
  return true;
}

void removeDatasetRuntime(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    tsd::scene::LayerNodeRef root)
{
  if (!root)
    return;

  tsd::io::ArchivePlanOptions options;
  options.animationManager = &animationManager;
  const auto result = tsd::io::plan_SubtreeArchive(scene, root, options);
  if (!result.accepted()) {
    tsd::core::logError("[removeDatasetRuntime] %s", result.message.c_str());
    return;
  }

  for (auto it = result.plan.ownedAnimations.rbegin();
       it != result.plan.ownedAnimations.rend();
       ++it)
    animationManager.removeAnimation(*it);
  scene.removeNode(root, true);
}

bool datasetRuntimeContainsObject(tsd::scene::Scene &scene,
    tsd::scene::LayerNodeRef root,
    const tsd::scene::Object *object)
{
  if (!root || !object)
    return false;
  const auto result = tsd::io::plan_SubtreeArchive(scene, root);
  return result.accepted() && result.plan.containsObject(object);
}

} // namespace tsd::scivis_studio
