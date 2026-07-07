// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "DatasetIO.h"

#include "ProjectAssetTransaction.h"
#include "ProjectSerialization.h"

#include "tsd/animation/Animation.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/animation/SpatialFieldFileBinding.hpp"
#include "tsd/io/archives/SubtreeArchiveContent.hpp"
#include "tsd/io/archives/detail/ArchivePlan.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/scene/objects/Volume.hpp"

#include <algorithm>
#include <fstream>
#include <memory>
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
  }
  // A File Animation Dataset's source list is persisted only in its sibling
  // Source List File; dataset files no longer embed it (ADR 0013).
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
    // A new-format dataset file carries no paths (the sibling Source List
    // File is the only persisted list); legacy embedded sourceFiles still
    // load unmodified and mark the dataset for migration (ADR 0013).
    if (auto *files = node.child("sourceFiles");
        files && files->numChildren() != 0) {
      files->foreach_child([&](tsd::core::DataNode &entry) {
        dataset.sourceFiles.push_back({entry.getValueOr<std::string>("")});
      });
      if (std::any_of(dataset.sourceFiles.begin(),
              dataset.sourceFiles.end(),
              [](const DatasetSourceFile &source) {
                return source.path.empty();
              }))
        return fail("file-animation source paths cannot be empty", error);
      dataset.pendingSourceListMigration = true;
    }
    if (dataset.importerType != "VOLUME_ANIMATION")
      return fail("unsupported file-animation importerType", error);
  }

  dataset.status = DatasetStatus::Unavailable;
  dataset.dirty = false;
  dataset.persistedName = dataset.name;
  return true;
}

// True when the archive serializes any scene content at all — an object or a
// subtree node below the root. A Declared Dataset's asset has neither
// (ADR 0014).
bool datasetArchiveHasSceneRepresentation(tsd::core::DataNode &archive)
{
  bool hasContent = false;
  if (auto *objectDB = archive.child("objectDB")) {
    objectDB->foreach_child([&](tsd::core::DataNode &typeNode) {
      hasContent |= typeNode.numChildren() != 0;
    });
  }
  if (auto *subtree = archive.child("subtree")) {
    if (auto *children = subtree->child("children"))
      hasContent |= children->numChildren() != 0;
  }
  return hasContent;
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
  for (const auto &source : dataset.sourceFiles) {
    files.push_back(
        source.resolvedPath.empty() ? source.path : source.resolvedPath);
  }

  auto &animation =
      animationManager.addAnimation(dataset.name + " file animation");
  animation.emplaceFileBinding<tsd::io::SpatialFieldFileBinding>(
      &scene, volume, field->self(), std::move(files));
  return true;
}

// Dataset Materialization: the first successful Dataset Load of a Declared
// Dataset builds its runtime representation by importing from the Source
// List (ADR 0014). The dataset comes back dirty so the next save records the
// scene representation, upgrading the asset in place. A failed
// materialization changes nothing.
bool materializeDeclaredDataset(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    Dataset &&dataset,
    tsd::scene::LayerNodeRef destinationParent,
    Dataset &datasetOut,
    tsd::scene::LayerNodeRef &rootOut,
    std::string *error)
{
  std::vector<std::string> files;
  files.reserve(dataset.sourceFiles.size());
  for (const auto &source : dataset.sourceFiles) {
    files.push_back(
        source.resolvedPath.empty() ? source.path : source.resolvedPath);
  }

  auto root = scene.insertChildNode(destinationParent, dataset.name.c_str());
  auto discardRuntime = [&]() {
    removeDatasetRuntime(scene, animationManager, root);
  };
  try {
    // Validation restricts file-animation datasets to the VOLUME_ANIMATION
    // importer, so materialization is a volume-animation import.
    tsd::io::import_animations(scene,
        animationManager,
        {{tsd::io::ImporterType::VOLUME_ANIMATION, files}},
        root);
  } catch (const std::exception &e) {
    discardRuntime();
    return fail(
        std::string("declared dataset materialization failed: ") + e.what(),
        error);
  } catch (...) {
    discardRuntime();
    return fail("declared dataset materialization failed", error);
  }

  if (!findDatasetVolume(root)) {
    discardRuntime();
    return fail(
        "declared dataset materialization created no volume; check that the "
        "Source List File's entries resolve on this machine",
        error);
  }

  dataset.declared = false;
  dataset.dirty = true;
  datasetOut = std::move(dataset);
  datasetOut.status = DatasetStatus::Available;
  rootOut = root;
  return true;
}

} // namespace

std::filesystem::path sourceListFilePath(
    const std::filesystem::path &datasetFile)
{
  auto path = datasetFile;
  path.replace_extension(SOURCE_LIST_FILE_EXTENSION);
  return path;
}

bool readSourceListFile(const std::filesystem::path &file,
    std::vector<DatasetSourceFile> &sourceList,
    std::string *error)
{
  sourceList.clear();
  std::ifstream in(file, std::ios::binary);
  if (!in) {
    return fail(
        "Source List File is missing or unreadable: " + file.string(), error);
  }

  const auto anchor = file.parent_path();
  std::string line;
  while (std::getline(in, line)) {
    const auto first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
      continue;
    const auto last = line.find_last_not_of(" \t\r\n");
    DatasetSourceFile entry;
    entry.path = line.substr(first, last - first + 1);
    if (std::filesystem::path(entry.path).is_relative())
      entry.resolvedPath = (anchor / entry.path).string();
    sourceList.push_back(std::move(entry));
  }
  if (in.bad()) {
    sourceList.clear();
    return fail("Source List File is unreadable: " + file.string(), error);
  }
  if (sourceList.empty())
    return fail("Source List File is empty: " + file.string(), error);
  return true;
}

bool writeSourceListFile(const std::filesystem::path &file,
    const std::vector<DatasetSourceFile> &sourceList,
    std::string *error)
{
  if (sourceList.empty())
    return fail("File Animation Source List is empty", error);
  for (const auto &source : sourceList) {
    if (source.path.empty())
      return fail("file-animation source paths cannot be empty", error);
    if (source.path.find_first_of("\r\n") != std::string::npos) {
      return fail(
          "file-animation source paths cannot contain line breaks", error);
    }
  }

  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  if (!out) {
    return fail(
        "failed to open Source List File for writing: " + file.string(),
        error);
  }
  for (const auto &source : sourceList)
    out << source.path << '\n';
  out.flush();
  if (!out)
    return fail("failed to write Source List File: " + file.string(), error);
  return true;
}

bool datasetArchiveUsesSourceListFile(tsd::core::DataNode &archive)
{
  auto *metadata = archive.child("dataset");
  if (!metadata
      || (*metadata)["sourceKind"].getValueOr<std::string>("")
          != dataset::toString(DatasetSourceKind::FileAnimation))
    return false;
  auto *files = metadata->child("sourceFiles");
  return !files || files->numChildren() == 0;
}

bool readDatasetSourceListEntries(const std::filesystem::path &datasetFile,
    std::vector<std::string> &entries,
    std::string *error)
{
  entries.clear();
  auto validation = validateDatasetArchiveFile(datasetFile);
  if (!validation.ok)
    return fail(validation.error, error);
  if (validation.dataset.sourceKind != DatasetSourceKind::FileAnimation) {
    return fail("dataset '" + validation.dataset.name
            + "' is not a file-animation dataset",
        error);
  }

  if (validation.dataset.pendingSourceListMigration) {
    for (const auto &source : validation.dataset.sourceFiles)
      entries.push_back(source.path);
    return true;
  }

  std::vector<DatasetSourceFile> sourceList;
  if (!readSourceListFile(sourceListFilePath(datasetFile), sourceList, error))
    return false;
  for (const auto &source : sourceList)
    entries.push_back(source.path);
  return true;
}

bool writeDatasetSourceListEdit(const std::filesystem::path &datasetFile,
    const std::vector<std::string> &entries,
    std::string *error)
{
  auto tree = std::make_shared<tsd::core::DataTree>();
  if (!tree->load(datasetFile.string().c_str())) {
    return fail(
        "failed to load Dataset Archive: " + datasetFile.string(), error);
  }
  auto validation = validateDatasetArchive(tree->root());
  if (!validation.ok)
    return fail(validation.error, error);
  if (validation.dataset.sourceKind != DatasetSourceKind::FileAnimation) {
    return fail("dataset '" + validation.dataset.name
            + "' is not a file-animation dataset",
        error);
  }

  std::vector<DatasetSourceFile> sourceList;
  sourceList.reserve(entries.size());
  for (const auto &entry : entries)
    sourceList.push_back({entry});

  ProjectSavePlan plan;
  plan.directory = datasetFile.parent_path();
  const auto expectedName = validation.dataset.name;

  // The Source List File carries the edit and installs last.
  plan.manifest.description =
      "dataset '" + expectedName + "' Source List File";
  plan.manifest.target = sourceListFilePath(datasetFile).filename();
  plan.manifest.ownedTarget = plan.manifest.target;
  plan.manifest.writer = [sourceList](const std::filesystem::path &file,
                             std::string *writeError) {
    return writeSourceListFile(file, sourceList, writeError);
  };
  plan.manifest.validator = [](const std::filesystem::path &file,
                                std::string *validationError) {
    std::vector<DatasetSourceFile> staged;
    return readSourceListFile(file, staged, validationError);
  };

  if (validation.dataset.pendingSourceListMigration) {
    // Migrate the legacy asset in place: the same edit that externalizes the
    // list rewrites the dataset file without it, as the pair transaction.
    tree->root()["dataset"].remove("sourceFiles");
    ProjectAssetWrite datasetWrite;
    datasetWrite.description = "dataset '" + expectedName + "'";
    datasetWrite.target = datasetFile.filename();
    datasetWrite.ownedTarget = datasetWrite.target;
    datasetWrite.writer = [tree](const std::filesystem::path &file,
                              std::string *writeError) {
      if (tree->save(file.string().c_str()))
        return true;
      return fail("failed to rewrite Dataset Archive", writeError);
    };
    datasetWrite.validator = [expectedName](const std::filesystem::path &file,
                                 std::string *validationError) {
      auto staged = validateDatasetArchiveFile(file);
      if (!staged.ok)
        return fail(staged.error, validationError);
      if (staged.dataset.name != expectedName) {
        return fail("dataset name does not match target", validationError);
      }
      if (staged.dataset.pendingSourceListMigration) {
        return fail(
            "staged dataset file still embeds sourceFiles", validationError);
      }
      return true;
    };
    plan.assets.push_back(std::move(datasetWrite));
  }

  AssetTransaction transaction;
  return transaction.commit(plan, error);
}

DatasetAssetValidationResult validateDatasetAsset(
    const std::filesystem::path &file)
{
  auto result = validateDatasetArchiveFile(file);
  if (result.ok
      && result.dataset.sourceKind == DatasetSourceKind::FileAnimation
      && !result.dataset.pendingSourceListMigration) {
    result.ok = readSourceListFile(
        sourceListFilePath(file), result.dataset.sourceFiles, &result.error);
  }
  return result;
}

DatasetAssetValidationResult validateDatasetArchiveFile(
    const std::filesystem::path &file)
{
  tsd::core::DataTree tree;
  if (!tree.load(file.string().c_str())) {
    DatasetAssetValidationResult result;
    result.error = "failed to load Dataset Archive";
    return result;
  }

  return validateDatasetArchive(tree.root());
}

DatasetAssetValidationResult validateDatasetArchive(
    tsd::core::DataNode &archive)
{
  DatasetAssetValidationResult result;

  auto subtreeResult =
      tsd::io::validate_SubtreeArchiveContent(archive, DATASET_DESC);
  if (!subtreeResult.accepted()) {
    result.error = subtreeResult.message;
    return result;
  }

  auto *metadata = archive.child("dataset");
  if (!metadata) {
    result.error = "Dataset Archive requires dataset metadata";
    return result;
  }
  if (!nodeToDatasetMetadata(*metadata, result.dataset, &result.error))
    return result;

  const bool hasRepresentation = datasetArchiveHasSceneRepresentation(archive);
  if (result.dataset.sourceKind == DatasetSourceKind::Static
      && !hasRepresentation) {
    // A static dataset without its content is a contradiction; only
    // file-animation datasets have a declared, representation-less flavor
    // (ADR 0014).
    result.error = "static datasets require a scene representation";
    return result;
  }
  if (result.dataset.sourceKind == DatasetSourceKind::FileAnimation) {
    if (!hasRepresentation) {
      result.dataset.declared = true;
    } else {
      auto *volumes = archive["objectDB"].child("volume");
      if (!volumes || volumes->numChildren() != 1) {
        result.error =
            "file-animation datasets require exactly one volume target";
        return result;
      }
    }
  }

  if (auto *animations = archive.child("animations")) {
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

  const auto displayName = archive["displayName"].getValueOr<std::string>("");
  const auto subtreeName =
      archive["subtree"]["name"].getValueOr<std::string>("");
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

  // Only the dataset file is validated here: the sibling Source List File is
  // written, staged, and validated by the caller that owns the pair.
  auto validation = validateDatasetArchiveFile(file);
  if (!validation.ok)
    return fail("dataset validation failed: " + validation.error, error);
  return true;
}

bool saveDeclaredDatasetArchiveFile(
    const Dataset &dataset, const std::filesystem::path &file, std::string *error)
{
  if (dataset.sourceKind != DatasetSourceKind::FileAnimation) {
    return fail("only file-animation datasets can be declared", error);
  }
  if (!validateRigName(dataset.name, error))
    return false;

  // The declared asset is an ordinary Dataset Archive whose subtree is empty:
  // serialize a fresh root with no children and no objects.
  tsd::scene::Scene stagingScene;
  auto *layer = stagingScene.addLayer("staging");
  if (!layer)
    return fail("failed to stage declared dataset subtree", error);
  auto root = stagingScene.insertChildNode(layer->root(), dataset.name.c_str());

  tsd::core::DataTree tree;
  if (!tsd::io::serialize_SubtreeArchiveContent(
          root, tree.root(), DATASET_DESC, dataset.name)) {
    return fail("failed to serialize declared dataset subtree", error);
  }

  datasetMetadataToNode(dataset, tree.root()["dataset"]);
  tree.root()["subtree"]["name"] = dataset.name;
  if (!tree.save(file.string().c_str()))
    return fail("failed to write dataset metadata", error);

  auto validation = validateDatasetArchiveFile(file);
  if (!validation.ok)
    return fail("dataset validation failed: " + validation.error, error);
  if (!validation.dataset.declared)
    return fail("declared dataset asset has a scene representation", error);
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
  tsd::core::DataTree tree;
  if (!tree.load(file.string().c_str()))
    return fail("failed to load Dataset Archive", error);

  // Dataset Load is the one moment the Source List File is read; the load
  // fails cleanly when the sibling is missing, unreadable, or empty.
  std::vector<DatasetSourceFile> sourceList;
  const std::vector<DatasetSourceFile> *sourceListPtr = nullptr;
  if (datasetArchiveUsesSourceListFile(tree.root())) {
    if (!readSourceListFile(sourceListFilePath(file), sourceList, error))
      return false;
    sourceListPtr = &sourceList;
  }

  return deserializeDatasetArchive(scene,
      animationManager,
      tree.root(),
      destinationParent,
      sourceListPtr,
      datasetOut,
      rootOut,
      error);
}

bool deserializeDatasetArchive(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    tsd::core::DataNode &archive,
    tsd::scene::LayerNodeRef destinationParent,
    const std::vector<DatasetSourceFile> *sourceList,
    Dataset &datasetOut,
    tsd::scene::LayerNodeRef &rootOut,
    std::string *error)
{
  auto validation = validateDatasetArchive(archive);
  if (!validation.ok)
    return fail(validation.error, error);

  if (validation.dataset.sourceKind == DatasetSourceKind::FileAnimation
      && validation.dataset.sourceFiles.empty()) {
    if (!sourceList || sourceList->empty()) {
      return fail(
          "file-animation dataset requires its Source List File", error);
    }
    validation.dataset.sourceFiles = *sourceList;
  }

  if (validation.dataset.declared) {
    return materializeDeclaredDataset(scene,
        animationManager,
        std::move(validation.dataset),
        destinationParent,
        datasetOut,
        rootOut,
        error);
  }

  tsd::io::SubtreeArchiveContentOptions options;
  options.animationManager = &animationManager;
  auto loadedContent = tsd::io::deserialize_SubtreeArchiveContent(
      scene, archive, destinationParent, DATASET_DESC, nullptr, options);
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

bool removeDatasetRuntime(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    tsd::scene::LayerNodeRef root)
{
  if (!root)
    return true;

  tsd::io::ArchivePlanOptions options;
  options.animationManager = &animationManager;
  const auto result = tsd::io::plan_SubtreeArchive(scene, root, options);
  if (!result.accepted()) {
    tsd::core::logError("[removeDatasetRuntime] %s", result.message.c_str());
    return false;
  }

  for (auto it = result.plan.ownedAnimations.rbegin();
       it != result.plan.ownedAnimations.rend();
       ++it)
    animationManager.removeAnimation(*it);

  scene.removeNode(root, false);

  // Erase the subtree's whole object closure, not just the objects its leaf
  // nodes referenced: a dataset owns everything reachable from its subtree
  // (that is what its Archive serializes), and reclaiming that memory is the
  // point of teardown. Objects still used elsewhere — e.g. the scene default
  // material shared into an imported mesh — must survive, so only erase
  // objects whose use count has dropped to zero, iterating to a fixpoint as
  // dependents release their references.
  std::vector<tsd::scene::Object *> pending;
  pending.reserve(result.plan.objects.size());
  for (const auto &object : result.plan.objects) {
    if (object.source)
      pending.push_back(object.source);
  }
  for (bool erased = true; erased && !pending.empty();) {
    erased = false;
    for (auto it = pending.begin(); it != pending.end();) {
      if ((*it)->totalUseCount() == 0) {
        scene.removeObject(*it);
        it = pending.erase(it);
        erased = true;
      } else
        ++it;
    }
  }
  return true;
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
