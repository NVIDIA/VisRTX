// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/serialization.hpp"
#include "tsd/io/serialization/serialization_animation_archive.hpp"
#include "tsd/io/serialization/serialization_closure.hpp"
// std
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace tsd::io {

using namespace tsd::io::detail;

namespace {

// Schemas the subtree loader recognizes well enough to report an incompatible
// (rather than unknown) schema error.
const std::vector<std::string_view> KNOWN_SUBTREE_SCHEMAS = {
    schema::LAYER_SUBTREE,
    schema::OBJECT_SURFACE,
    schema::OBJECT_VOLUME,
    schema::SCENE_FULL,
    schema::SCENE_CAMERAS_AND_RENDERERS};

ClosurePolicy policyForDesc(const SubtreeIODesc &desc)
{
  return desc.objectPolicy == ArchiveObjectPolicy::LightsOnly
      ? lightRigPolicy()
      : layerSubtreePolicy();
}

// Import helpers /////////////////////////////////////////////////////////////

// Collect the distinct file-local object keys referenced anywhere in the
// serialized subtree (node values + instance parameters). These seed payload
// reachability validation.
std::vector<ObjectKey> collectSubtreeRefKeys(core::DataNode &subtree)
{
  std::vector<ObjectKey> keys;
  subtree.traverse([&](core::DataNode &n, int) {
    if (n.holdsObjectIdx()) {
      anari::DataType type = ANARI_UNKNOWN;
      size_t index = tsd::core::INVALID_INDEX;
      n.getValueAsObjectIdx(&type, &index);
      auto key = makeKey(type, index);
      if (std::none_of(keys.begin(), keys.end(), [&](auto &k) {
            return sameKey(k, key);
          }))
        keys.push_back(key);
    }
    return true;
  });
  return keys;
}

} // namespace

bool export_Subtree(const char *filename,
    LayerNodeRef root,
    const SubtreeIODesc &desc,
    std::string_view displayName,
    const SubtreeIOOptions &options)
{
  if (!filename) {
    tsd::core::logError("[export_Subtree] filename is null");
    return false;
  }

  if (!root) {
    tsd::core::logError("[export_Subtree] root node is invalid");
    return false;
  }

  auto *layer = (*root).value().layer();
  auto *scene = layer ? layer->scene() : nullptr;
  if (!layer || !scene) {
    tsd::core::logError("[export_Subtree] root node has no owning Scene");
    return false;
  }

  ArchivePlanOptions planOptions;
  planOptions.objectPolicy = desc.objectPolicy;
  planOptions.animationManager = options.animationManager;
  planOptions.fileBindings = options.fileBindings;
  auto planResult = plan_SubtreeArchive(*scene, root, planOptions);
  if (!planResult.accepted()) {
    tsd::core::logError("[export_Subtree] %s", planResult.message.c_str());
    return false;
  }
  const auto entries = closureEntriesForPlan(planResult.plan);
  std::string errorMessage;

  core::DataTree tree;
  auto &dbRoot = tree.root();
  dbRoot.reset();

  core::writeDataTreeMetadata(dbRoot,
      {core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          std::string(desc.fileType),
          std::string(desc.schema),
          1});

  if (!displayName.empty())
    dbRoot["displayName"] = std::string(displayName);

  if (!writeObjectDB(dbRoot["objectDB"], entries, errorMessage)) {
    tsd::core::logError("[export_Subtree] %s", errorMessage.c_str());
    return false;
  }

  auto &subtreeNode = dbRoot["subtree"];
  layerSubtreeToNode(*layer, root, subtreeNode);
  if (!rewriteRefsToLocal(subtreeNode, entries, errorMessage)) {
    tsd::core::logError("[export_Subtree] %s", errorMessage.c_str());
    return false;
  }

  if (options.animationManager
      && !writeSubtreeAnimations(dbRoot["animations"],
          *options.animationManager,
          planResult.plan,
          errorMessage)) {
    tsd::core::logError("[export_Subtree] %s", errorMessage.c_str());
    return false;
  }

  if (!tree.save(filename)) {
    tsd::core::logError("[export_Subtree] failed to write file '%s'", filename);
    return false;
  }

  return true;
}

PayloadValidationResult validate_SubtreePayload(
    core::DataNode &root, const SubtreeIODesc &desc)
{
  auto result = validateEnvelope(
      root, desc.fileType, {desc.schema}, KNOWN_SUBTREE_SCHEMAS);
  if (!result.accepted())
    return result;

  auto *objectDB = root.child("objectDB");
  if (!objectDB) {
    result.status = PayloadValidationStatus::MissingRequiredNode;
    result.message = std::string(desc.fileType) + " payload requires objectDB";
    return result;
  }

  auto *subtree = root.child("subtree");
  if (!subtree) {
    result.status = PayloadValidationStatus::MissingRequiredNode;
    result.message = std::string(desc.fileType) + " payload requires subtree";
    return result;
  }

  const auto policy = policyForDesc(desc);

  bool ok = true;
  objectDB->foreach_child([&](core::DataNode &poolNode) {
    if (!ok)
      return;
    if (poolNode.numChildren() > 0 && !poolAllowed(policy, poolNode.name())) {
      result.status = PayloadValidationStatus::IncompatibleSchema;
      result.message = std::string(desc.fileType)
          + " payload contains unsupported object pool '" + poolNode.name()
          + "'";
      ok = false;
    }
  });
  if (!ok)
    return result;

  std::vector<FileObjectEntry> entries;
  if (!collectFileObjects(*objectDB, entries, result))
    return result;

  auto seedKeys = collectSubtreeRefKeys(*subtree);
  collectAnimationRefKeys(root, seedKeys);
  checkGraphConsistency(
      entries, seedKeys, policy, /*requireAllReachable=*/true, result);
  if (!result.accepted())
    return result;
  validateSubtreeAnimations(root, entries, *subtree, result);
  return result;
}

LayerNodeRef import_Subtree(Scene &scene,
    const char *filename,
    LayerNodeRef destinationParent,
    const SubtreeIODesc &desc,
    std::string *displayNameOut,
    const SubtreeIOOptions &options)
{
  return import_SubtreeWithOwnership(
      scene, filename, destinationParent, desc, displayNameOut, options)
      .root;
}

bool export_LayerSubtree(const char *filename, LayerNodeRef root)
{
  return export_Subtree(filename,
      root,
      {"layer-subtree", schema::LAYER_SUBTREE, ArchiveObjectPolicy::All});
}

PayloadValidationResult validate_LayerSubtreePayload(core::DataNode &root)
{
  return validate_SubtreePayload(
      root, {"layer-subtree", schema::LAYER_SUBTREE, ArchiveObjectPolicy::All});
}

LayerNodeRef import_LayerSubtree(
    Scene &scene, const char *filename, LayerNodeRef destinationParent)
{
  return import_Subtree(scene,
      filename,
      destinationParent,
      {"layer-subtree", schema::LAYER_SUBTREE, ArchiveObjectPolicy::All});
}

} // namespace tsd::io
