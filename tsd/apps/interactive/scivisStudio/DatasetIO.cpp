// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "DatasetIO.h"

#include "ProjectSerialization.h"

#include "tsd/animation/Animation.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/io/animation/SpatialFieldFileBinding.hpp"
#include "tsd/io/serialization.hpp"
#include "tsd/scene/objects/Volume.hpp"

#include <algorithm>
#include <vector>

namespace tsd::scivis_studio {

namespace {

const tsd::io::SubtreeIODesc DATASET_DESC{
    DATASET_FILE_TYPE, DATASET_SCHEMA, false};

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
    return fail("dataset assets must not contain a project-local ID", error);

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

struct ObjectKey
{
  anari::DataType type{ANARI_UNKNOWN};
  size_t index{TSD_INVALID_INDEX};
};

ObjectKey objectKey(const tsd::scene::Object &object)
{
  return {anari::isArray(object.type()) ? ANARI_ARRAY : object.type(),
      object.index()};
}

bool containsObject(
    const std::vector<ObjectKey> &objects, anari::DataType type, size_t index)
{
  const auto canonical = anari::isArray(type) ? ANARI_ARRAY : type;
  return std::any_of(objects.begin(), objects.end(), [&](const ObjectKey &key) {
    return key.type == canonical && key.index == index;
  });
}

std::vector<ObjectKey> collectDatasetObjects(
    tsd::scene::Scene &scene, tsd::scene::LayerNodeRef root)
{
  std::vector<ObjectKey> objects;
  std::vector<tsd::scene::Object *> pending;
  auto add = [&](tsd::scene::Object *object) {
    if (!object || containsObject(objects, object->type(), object->index()))
      return;
    objects.push_back(objectKey(*object));
    pending.push_back(object);
  };

  auto *layer = (*root).value().layer();
  layer->traverse(root, [&](tsd::scene::LayerNode &node, int) {
    if (node->isObject())
      add(node->getObject());
    for (const auto &parameter : node->getInstanceParameters()) {
      if (parameter.second.holdsObject())
        add(scene.getObject(parameter.second));
    }
    return true;
  });

  for (size_t i = 0; i < pending.size(); ++i) {
    auto *object = pending[i];
    for (size_t p = 0; p < object->numParameters(); ++p) {
      const auto &parameter = object->parameterAt(p);
      if (parameter.value().holdsObject())
        add(scene.getObject(parameter.value()));
      if (parameter.hasMin() && parameter.min().holdsObject())
        add(scene.getObject(parameter.min()));
      if (parameter.hasMax() && parameter.max().holdsObject())
        add(scene.getObject(parameter.max()));
    }
    for (size_t m = 0; m < object->numMetadata(); ++m) {
      const auto *name = object->getMetadataName(m);
      anari::DataType arrayType = ANARI_UNKNOWN;
      const void *array = nullptr;
      size_t arraySize = 0;
      object->getMetadataArray(name, &arrayType, &array, &arraySize);
      if (arrayType == ANARI_UNKNOWN) {
        const auto value = object->getMetadataValue(name);
        if (value.holdsObject())
          add(scene.getObject(value));
      }
    }
  }
  return objects;
}

bool animationTargetsDataset(const tsd::animation::Animation &animation,
    tsd::scene::LayerNodeRef root,
    const std::vector<ObjectKey> &objects)
{
  for (const auto &binding : animation.objectParameterBindings()) {
    auto *target = binding.target();
    if (target && containsObject(objects, target->type(), target->index()))
      return true;
  }

  auto *layer = (*root).value().layer();
  for (const auto &binding : animation.transformBindings()) {
    auto target = binding.target();
    if (target && (target == root || layer->isAncestorOf(root, target)))
      return true;
  }

  for (const auto &binding : animation.fileBindings()) {
    tsd::core::DataTree tree;
    binding->toDataNode(tree.root());
    if (binding->kind() == "spatialField") {
      const auto index =
          tree.root()["targetIndex"].getValueOr<size_t>(TSD_INVALID_INDEX);
      if (containsObject(objects, ANARI_VOLUME, index))
        return true;
    } else if (binding->kind() == "ensight") {
      bool found = false;
      tree.root()["parts"].foreach_child([&](tsd::core::DataNode &part) {
        found |= containsObject(objects,
            ANARI_GEOMETRY,
            part["targetIndex"].getValueOr<size_t>(TSD_INVALID_INDEX));
      });
      if (found)
        return true;
    }
  }
  return false;
}

} // namespace

DatasetAssetValidationResult validateDatasetAsset(
    const std::filesystem::path &file)
{
  DatasetAssetValidationResult result;
  tsd::core::DataTree tree;
  if (!tree.load(file.string().c_str())) {
    result.error = "failed to load dataset asset";
    return result;
  }

  auto subtreeResult =
      tsd::io::validate_SubtreePayload(tree.root(), DATASET_DESC);
  if (!subtreeResult.accepted()) {
    result.error = subtreeResult.message;
    return result;
  }

  auto *metadata = tree.root().child("dataset");
  if (!metadata) {
    result.error = "dataset asset requires dataset metadata";
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
          "dataset assets cannot persist derived runtime file bindings";
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

bool exportDatasetAsset(const Dataset &dataset,
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
  const auto datasetObjects = collectDatasetObjects(*scene, root);
  size_t ownedFileAnimations = 0;
  for (const auto &animation : animationManager.animations()) {
    if (!animation.fileBindings().empty()
        && animationTargetsDataset(animation, root, datasetObjects))
      ++ownedFileAnimations;
  }
  if (dataset.sourceKind == DatasetSourceKind::Static
      && ownedFileAnimations != 0)
    return fail("static datasets cannot own file animations", error);
  if (dataset.sourceKind == DatasetSourceKind::FileAnimation
      && ownedFileAnimations != 1)
    return fail(
        "file-animation datasets must own one runtime file animation", error);

  tsd::io::SubtreeIOOptions options;
  options.animationManager = &animationManager;
  options.includeFileBindingAnimations = false;
  if (!tsd::io::export_Subtree(
          file.string().c_str(), root, DATASET_DESC, dataset.name, options))
    return fail("failed to serialize dataset subtree", error);

  tsd::core::DataTree tree;
  if (!tree.load(file.string().c_str()))
    return fail("failed to reopen serialized dataset", error);
  datasetMetadataToNode(dataset, tree.root()["dataset"]);
  tree.root()["subtree"]["name"] = dataset.name;
  if (!tree.save(file.string().c_str()))
    return fail("failed to write dataset metadata", error);

  auto validation = validateDatasetAsset(file);
  if (!validation.ok)
    return fail("dataset validation failed: " + validation.error, error);
  return true;
}

bool importDatasetAsset(tsd::scene::Scene &scene,
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

  const auto originalAnimationCount = animationManager.animations().size();
  tsd::io::SubtreeIOOptions options;
  options.animationManager = &animationManager;
  auto root = tsd::io::import_Subtree(scene,
      file.string().c_str(),
      destinationParent,
      DATASET_DESC,
      nullptr,
      options);
  if (!root)
    return fail("failed to reconstruct dataset subtree", error);

  if (validation.dataset.sourceKind == DatasetSourceKind::FileAnimation
      && !recreateFileAnimation(
          validation.dataset, scene, animationManager, root, error)) {
    while (animationManager.animations().size() > originalAnimationCount)
      animationManager.removeAnimation(originalAnimationCount);
    scene.removeNode(root, true);
    return false;
  }

  datasetOut = std::move(validation.dataset);
  datasetOut.status = DatasetStatus::Available;
  rootOut = root;
  return true;
}

void removeDatasetRuntime(tsd::scene::Scene &scene,
    tsd::animation::AnimationManager &animationManager,
    tsd::scene::LayerNodeRef root)
{
  if (!root)
    return;
  const auto objects = collectDatasetObjects(scene, root);
  for (size_t i = animationManager.animations().size(); i > 0; --i) {
    if (animationTargetsDataset(
            animationManager.animations()[i - 1], root, objects))
      animationManager.removeAnimation(i - 1);
  }
  scene.removeNode(root, true);
}

bool datasetRuntimeContainsObject(tsd::scene::Scene &scene,
    tsd::scene::LayerNodeRef root,
    const tsd::scene::Object *object)
{
  if (!root || !object)
    return false;
  const auto objects = collectDatasetObjects(scene, root);
  return containsObject(objects, object->type(), object->index());
}

} // namespace tsd::scivis_studio
