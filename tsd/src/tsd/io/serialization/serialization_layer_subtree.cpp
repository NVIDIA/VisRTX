// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/serialization.hpp"
#include "tsd/io/serialization/serialization_closure.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
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

// Export helpers /////////////////////////////////////////////////////////////

// Gather every Scene object directly referenced by the subtree's node values
// and instance parameters; these seed the export closure.
std::vector<Object *> collectSubtreeSeeds(
    const Layer &layer, LayerNodeRef root, Scene &scene)
{
  std::vector<Object *> seeds;
  layer.traverse_const(root, [&](const LayerNode &tsdNode, int) {
    if (tsdNode->isObject()) {
      if (auto *o = tsdNode->getObject())
        seeds.push_back(o);
    }
    for (const auto &p : tsdNode->getInstanceParameters()) {
      if (p.second.holdsObject()) {
        if (auto *o = scene.getObject(p.second))
          seeds.push_back(o);
      }
    }
    return true;
  });
  return seeds;
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

// Apply a node's serialized instance parameters to a live node, remapping any
// object-valued parameters through the import target table.
void applyInstanceParameters(core::DataNode &src,
    LayerNodeData &data,
    const std::vector<TargetObjectEntry> &targets)
{
  auto *ipNode = src.child("instanceParameters");
  if (!ipNode)
    return;

  ipNode->foreach_child([&](core::DataNode &p) {
    Any value = p.getValue();
    if (value.holdsObject()) {
      if (auto *t = findTargetEntry(targets, makeKey(value)))
        value = Any(value.type(), t->target.getAsObjectIndex());
    }
    data.setInstanceParameter(p.name(), value);
  });
}

// Recursively recreate a serialized subtree as a new child of parent, remapping
// object references to freshly created Scene objects. Records every created
// node so the whole graft can be erased if a later step fails.
LayerNodeRef spliceSubtree(Scene &scene,
    core::DataNode &src,
    LayerNodeRef parent,
    const std::vector<TargetObjectEntry> &targets,
    std::vector<LayerNodeRef> &createdNodes,
    std::string &errorMessage)
{
  const std::string name = src["name"].getValueAs<std::string>();

  LayerNodeRef node;
  if (auto *srtNode = src.child("transformSRT"); srtNode != nullptr) {
    node = scene.insertChildTransformNode(parent, math::IDENTITY_MAT4, name.c_str());
    if (node)
      (*node).value().setAsTransform(srtNode->getValueAs<math::mat3>());
  } else {
    const Any value = src["value"].getValue();
    if (value.holdsObject()) {
      auto *target = findTargetEntry(targets, makeKey(value));
      if (!target) {
        errorMessage = "subtree node references an object missing from objectDB";
        return {};
      }
      node = scene.insertChildObjectNode(
          parent, value.type(), target->target.getAsObjectIndex(), name.c_str());
    } else if (value.type() == ANARI_FLOAT32_MAT4) {
      node = scene.insertChildTransformNode(
          parent, value.getAs<math::mat4>(), name.c_str());
    } else {
      node = scene.insertChildNode(parent, name.c_str());
    }
  }

  if (!node) {
    if (errorMessage.empty())
      errorMessage = "failed to create subtree node";
    return {};
  }

  createdNodes.push_back(node);

  (*node).value().setEnabled(src["enabled"].getValueOr(true));
  applyInstanceParameters(src, (*node).value(), targets);

  if (auto *childrenNode = src.child("children"); childrenNode != nullptr) {
    bool ok = true;
    childrenNode->foreach_child([&](core::DataNode &childSrc) {
      if (!ok)
        return;
      if (!spliceSubtree(
              scene, childSrc, node, targets, createdNodes, errorMessage))
        ok = false;
    });
    if (!ok)
      return {};
  }

  return node;
}

} // namespace

bool export_LayerSubtree(const char *filename, LayerNodeRef root)
{
  if (!filename) {
    tsd::core::logError("[export_LayerSubtree] filename is null");
    return false;
  }

  if (!root) {
    tsd::core::logError("[export_LayerSubtree] root node is invalid");
    return false;
  }

  auto *layer = (*root).value().layer();
  auto *scene = layer ? layer->scene() : nullptr;
  if (!layer || !scene) {
    tsd::core::logError("[export_LayerSubtree] root node has no owning Scene");
    return false;
  }

  const auto policy = layerSubtreePolicy();
  const auto seeds = collectSubtreeSeeds(*layer, root, *scene);

  std::vector<ClosureEntry> entries;
  std::string errorMessage;
  if (!buildClosure(*scene, seeds, policy, ObjectKey{}, entries, errorMessage)) {
    tsd::core::logError("[export_LayerSubtree] %s", errorMessage.c_str());
    return false;
  }

  core::DataTree tree;
  auto &dbRoot = tree.root();
  dbRoot.reset();

  core::writeDataTreeMetadata(dbRoot,
      {core::DATA_TREE_METADATA_ENVELOPE_VERSION,
          "layer-subtree",
          std::string(schema::LAYER_SUBTREE),
          1});

  if (!writeObjectDB(dbRoot["objectDB"], entries, errorMessage)) {
    tsd::core::logError("[export_LayerSubtree] %s", errorMessage.c_str());
    return false;
  }

  auto &subtreeNode = dbRoot["subtree"];
  layerSubtreeToNode(*layer, root, subtreeNode);
  if (!rewriteRefsToLocal(subtreeNode, entries, errorMessage)) {
    tsd::core::logError("[export_LayerSubtree] %s", errorMessage.c_str());
    return false;
  }

  if (!tree.save(filename)) {
    tsd::core::logError(
        "[export_LayerSubtree] failed to write file '%s'", filename);
    return false;
  }

  return true;
}

PayloadValidationResult validate_LayerSubtreePayload(core::DataNode &root)
{
  auto result = validateEnvelope(
      root, "layer-subtree", {schema::LAYER_SUBTREE}, KNOWN_SUBTREE_SCHEMAS);
  if (!result.accepted())
    return result;

  auto *objectDB = root.child("objectDB");
  if (!objectDB) {
    result.status = PayloadValidationStatus::MissingRequiredNode;
    result.message = "layer-subtree payload requires objectDB";
    return result;
  }

  auto *subtree = root.child("subtree");
  if (!subtree) {
    result.status = PayloadValidationStatus::MissingRequiredNode;
    result.message = "layer-subtree payload requires subtree";
    return result;
  }

  const auto policy = layerSubtreePolicy();

  bool ok = true;
  objectDB->foreach_child([&](core::DataNode &poolNode) {
    if (!ok)
      return;
    if (!isKnownObjectPoolName(poolNode.name()) && poolNode.numChildren() > 0) {
      result.status = PayloadValidationStatus::IncompatibleSchema;
      result.message = "layer-subtree payload contains unsupported object pool '"
          + poolNode.name() + "'";
      ok = false;
    }
  });
  if (!ok)
    return result;

  std::vector<FileObjectEntry> entries;
  if (!collectFileObjects(*objectDB, entries, result))
    return result;

  const auto seedKeys = collectSubtreeRefKeys(*subtree);
  checkGraphConsistency(
      entries, seedKeys, policy, /*requireAllReachable=*/true, result);
  return result;
}

LayerNodeRef import_LayerSubtree(
    Scene &scene, const char *filename, LayerNodeRef destinationParent)
{
  if (!filename) {
    tsd::core::logError("[import_LayerSubtree] filename is null");
    return {};
  }

  core::DataTree tree;
  if (!tree.load(filename)) {
    tsd::core::logError(
        "[import_LayerSubtree] failed to load file '%s'", filename);
    return {};
  }

  auto &root = tree.root();
  auto result = validate_LayerSubtreePayload(root);
  if (!result.accepted()) {
    tsd::core::logError("[import_LayerSubtree] payload validation failed: %s",
        result.message.c_str());
    return {};
  }

  std::vector<FileObjectEntry> fileEntries;
  if (!collectFileObjects(root["objectDB"], fileEntries, result)) {
    tsd::core::logError("[import_LayerSubtree] %s", result.message.c_str());
    return {};
  }

  std::vector<TargetObjectEntry> targetEntries;
  std::vector<Any> createdRefs;
  std::string errorMessage;
  if (!instantiateObjectDB(
          scene, fileEntries, targetEntries, createdRefs, errorMessage)) {
    tsd::core::logError("[import_LayerSubtree] %s", errorMessage.c_str());
    return {};
  }

  // Objects-only import: no destination to graft the subtree under.
  if (!destinationParent)
    return {};

  auto &subtreeNode = root["subtree"];
  std::vector<LayerNodeRef> createdNodes;
  auto splicedRoot = spliceSubtree(scene,
      subtreeNode,
      destinationParent,
      targetEntries,
      createdNodes,
      errorMessage);

  if (!splicedRoot) {
    if (!createdNodes.empty())
      scene.removeNode(createdNodes.front(), false);
    rollbackCreatedObjects(scene, createdRefs);
    tsd::core::logError("[import_LayerSubtree] %s",
        errorMessage.empty() ? "failed to reconstruct subtree"
                             : errorMessage.c_str());
    return {};
  }

  if (auto *layer = (*destinationParent).value().layer())
    scene.signalLayerStructureChanged(layer);

  return splicedRoot;
}

} // namespace tsd::io
