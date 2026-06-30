// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/DataTreeMetadata.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/serialization.hpp"
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
  return desc.lightsOnly ? lightRigPolicy() : layerSubtreePolicy();
}

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

std::vector<LayerNodeRef> collectSubtreeNodes(Layer &layer, LayerNodeRef root)
{
  std::vector<LayerNodeRef> nodes;
  layer.traverse(root, [&](LayerNode &node, int) {
    nodes.push_back(layer.at(node.index()));
    return true;
  });
  return nodes;
}

const ClosureEntry *findSourceEntry(const std::vector<ClosureEntry> &entries,
    anari::DataType type,
    size_t index)
{
  const auto key = makeKey(type, index);
  auto it = std::find_if(entries.begin(),
      entries.end(),
      [&](const auto &entry) { return sameKey(entry.source, key); });
  return it == entries.end() ? nullptr : &*it;
}

bool readObjectIndexArray(core::DataNode &node,
    anari::DataType expectedType,
    const size_t *&indices,
    size_t &count)
{
  anari::DataType actualType = ANARI_UNKNOWN;
  const void *data = nullptr;
  node.getValueAsArray(&actualType, &data, &count);
  if (actualType != expectedType) {
    indices = nullptr;
    count = 0;
    return false;
  }
  indices = static_cast<const size_t *>(data);
  return true;
}

size_t subtreeNodeIndex(
    const std::vector<LayerNodeRef> &nodes, LayerNodeRef target)
{
  auto it = std::find(nodes.begin(), nodes.end(), target);
  return it == nodes.end()
      ? tsd::core::INVALID_INDEX
      : static_cast<size_t>(std::distance(nodes.begin(), it));
}

struct AnimationSelection
{
  std::vector<size_t> indices;
  std::vector<Object *> dependencySeeds;
};

bool classifyObjectTarget(const std::vector<ClosureEntry> &entries,
    anari::DataType type,
    size_t index,
    bool &inside,
    bool &outside,
    std::string &errorMessage)
{
  if (type == ANARI_UNKNOWN || index == tsd::core::INVALID_INDEX) {
    errorMessage = "animation has an invalid object target";
    return false;
  }
  if (findSourceEntry(entries, type, index))
    inside = true;
  else
    outside = true;
  return true;
}

bool classifyFileBinding(const animation::FileBinding &binding,
    const std::vector<ClosureEntry> &entries,
    bool &inside,
    bool &outside,
    std::string &errorMessage)
{
  core::DataTree scratch;
  auto &node = scratch.root();
  binding.toDataNode(node);

  if (binding.kind() == "spatialField") {
    return classifyObjectTarget(entries,
        ANARI_VOLUME,
        node["targetIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX),
        inside,
        outside,
        errorMessage);
  }

  if (binding.kind() == "ensight") {
    auto *parts = node.child("parts");
    if (!parts || parts->numChildren() == 0) {
      errorMessage = "EnSight animation has no geometry targets";
      return false;
    }
    bool ok = true;
    parts->foreach_child([&](core::DataNode &part) {
      if (ok)
        ok = classifyObjectTarget(entries,
            ANARI_GEOMETRY,
            part["targetIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX),
            inside,
            outside,
            errorMessage);
    });
    return ok;
  }

  errorMessage = "cannot determine ownership of file animation kind '"
      + binding.kind() + "'";
  return false;
}

bool selectOwnedAnimations(const animation::AnimationManager &manager,
    Scene &scene,
    const std::vector<ClosureEntry> &baseEntries,
    const std::vector<LayerNodeRef> &subtreeNodes,
    bool includeFileBindings,
    AnimationSelection &selection,
    std::string &errorMessage)
{
  const auto &animations = manager.animations();
  for (size_t animationIndex = 0; animationIndex < animations.size();
       ++animationIndex) {
    const auto &animation = animations[animationIndex];
    if (!includeFileBindings && !animation.fileBindings().empty())
      continue;

    bool inside = false;
    bool outside = false;
    for (const auto &binding : animation.objectParameterBindings()) {
      auto *target = binding.target();
      if (!classifyObjectTarget(baseEntries,
              target ? target->type() : ANARI_UNKNOWN,
              target ? target->index() : tsd::core::INVALID_INDEX,
              inside,
              outside,
              errorMessage))
        return false;
    }

    for (const auto &binding : animation.transformBindings()) {
      auto target = binding.target();
      if (!target) {
        errorMessage = "animation has an invalid transform target";
        return false;
      }
      if (subtreeNodeIndex(subtreeNodes, target) != tsd::core::INVALID_INDEX)
        inside = true;
      else
        outside = true;
    }

    for (const auto &binding : animation.fileBindings()) {
      if (!classifyFileBinding(
              *binding, baseEntries, inside, outside, errorMessage))
        return false;
    }

    if (inside && outside) {
      errorMessage = "animation '" + animation.name()
          + "' has targets both inside and outside the exported subtree";
      return false;
    }
    if (!inside)
      continue;

    selection.indices.push_back(animationIndex);
    for (const auto &binding : animation.objectParameterBindings()) {
      if (!anari::isObject(binding.type()))
        continue;
      const auto *indices = static_cast<const size_t *>(binding.data().data());
      for (size_t i = 0; indices && i < binding.data().size(); ++i) {
        if (indices[i] == tsd::core::INVALID_INDEX)
          continue;
        auto *dependency = scene.getObject(binding.type(), indices[i]);
        if (!dependency) {
          errorMessage = "animation '" + animation.name()
              + "' references a missing keyframe object";
          return false;
        }
        selection.dependencySeeds.push_back(dependency);
      }
    }
  }
  return true;
}

bool remapObjectArrayToLocal(core::DataNode &binding,
    const std::vector<ClosureEntry> &entries,
    std::string &errorMessage)
{
  const auto type = static_cast<anari::DataType>(
      binding["dataType"].getValueOr<int>(ANARI_UNKNOWN));
  auto *data = binding.child("data");
  if (!data || !anari::isObject(type))
    return true;

  const size_t *indices = nullptr;
  size_t count = 0;
  if (!readObjectIndexArray(*data, type, indices, count)) {
    errorMessage = "animation keyframe object array has an invalid type";
    return false;
  }
  std::vector<size_t> remapped(count, tsd::core::INVALID_INDEX);
  for (size_t i = 0; i < count; ++i) {
    if (indices[i] == tsd::core::INVALID_INDEX)
      continue;
    auto *entry = findSourceEntry(entries, type, indices[i]);
    if (!entry) {
      errorMessage =
          "animation keyframe object is outside the exported closure";
      return false;
    }
    remapped[i] = entry->localIndex;
  }
  data->setValueAsArray(type, remapped.data(), remapped.size());
  return true;
}

bool remapFileBindingToLocal(core::DataNode &binding,
    const std::vector<ClosureEntry> &entries,
    std::string &errorMessage)
{
  const auto kind = binding["kind"].getValueOr<std::string>("");
  auto remapTarget = [&](core::DataNode &node, anari::DataType type) {
    const auto index =
        node["targetIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX);
    auto *entry = findSourceEntry(entries, type, index);
    if (!entry) {
      errorMessage = "file animation target is outside the exported closure";
      return false;
    }
    node["targetIndex"] = entry->localIndex;
    return true;
  };

  if (kind == "spatialField")
    return remapTarget(binding, ANARI_VOLUME);
  if (kind == "ensight") {
    bool ok = true;
    binding["parts"].foreach_child([&](core::DataNode &part) {
      if (ok)
        ok = remapTarget(part, ANARI_GEOMETRY);
    });
    return ok;
  }
  errorMessage = "cannot remap file animation kind '" + kind + "'";
  return false;
}

bool writeAnimations(core::DataNode &animationsNode,
    const animation::AnimationManager &manager,
    const AnimationSelection &selection,
    const std::vector<ClosureEntry> &entries,
    const std::vector<LayerNodeRef> &subtreeNodes,
    std::string &errorMessage)
{
  for (auto animationIndex : selection.indices) {
    auto &node = animationsNode.append();
    animationToNode(manager.animations()[animationIndex], node);

    if (auto *bindings = node.child("objectBindings")) {
      bool ok = true;
      bindings->foreach_child([&](core::DataNode &binding) {
        if (!ok)
          return;
        const auto type = static_cast<anari::DataType>(
            binding["targetType"].getValueOr<int>(ANARI_UNKNOWN));
        const auto index =
            binding["targetIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX);
        auto *entry = findSourceEntry(entries, type, index);
        if (!entry) {
          errorMessage = "animation target is outside the exported closure";
          ok = false;
          return;
        }
        binding["targetIndex"] = entry->localIndex;
        ok = remapObjectArrayToLocal(binding, entries, errorMessage);
      });
      if (!ok)
        return false;
    }

    if (auto *bindings = node.child("transformBindings")) {
      bool ok = true;
      size_t i = 0;
      bindings->foreach_child([&](core::DataNode &binding) {
        if (!ok)
          return;
        const auto &liveBinding =
            manager.animations()[animationIndex].transformBindings()[i++];
        const auto local = subtreeNodeIndex(subtreeNodes, liveBinding.target());
        if (local == tsd::core::INVALID_INDEX) {
          errorMessage = "animation transform target is outside the subtree";
          ok = false;
          return;
        }
        binding["layerName"] = "__subtree__";
        binding["nodeIndex"] = local;
      });
      if (!ok)
        return false;
    }

    if (auto *bindings = node.child("fileBindings")) {
      bool ok = true;
      bindings->foreach_child([&](core::DataNode &binding) {
        if (ok)
          ok = remapFileBindingToLocal(binding, entries, errorMessage);
      });
      if (!ok)
        return false;
    }
  }
  return true;
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

void collectAnimationRefKeys(core::DataNode &root, std::vector<ObjectKey> &keys)
{
  auto add = [&](anari::DataType type, size_t index) {
    if (type == ANARI_UNKNOWN || index == tsd::core::INVALID_INDEX)
      return;
    const auto key = makeKey(type, index);
    if (std::none_of(keys.begin(), keys.end(), [&](const auto &existing) {
          return sameKey(existing, key);
        }))
      keys.push_back(key);
  };

  auto *animations = root.child("animations");
  if (!animations)
    return;
  animations->foreach_child([&](core::DataNode &animation) {
    if (auto *bindings = animation.child("objectBindings")) {
      bindings->foreach_child([&](core::DataNode &binding) {
        const auto targetType = static_cast<anari::DataType>(
            binding["targetType"].getValueOr<int>(ANARI_UNKNOWN));
        add(targetType,
            binding["targetIndex"].getValueOr<size_t>(
                tsd::core::INVALID_INDEX));
        const auto dataType = static_cast<anari::DataType>(
            binding["dataType"].getValueOr<int>(ANARI_UNKNOWN));
        auto *data = binding.child("data");
        if (!data || !anari::isObject(dataType))
          return;
        const size_t *indices = nullptr;
        size_t count = 0;
        if (!readObjectIndexArray(*data, dataType, indices, count))
          return;
        for (size_t i = 0; i < count; ++i)
          add(dataType, indices[i]);
      });
    }
    if (auto *bindings = animation.child("fileBindings")) {
      bindings->foreach_child([&](core::DataNode &binding) {
        const auto kind = binding["kind"].getValueOr<std::string>("");
        if (kind == "spatialField") {
          add(ANARI_VOLUME,
              binding["targetIndex"].getValueOr<size_t>(
                  tsd::core::INVALID_INDEX));
        } else if (kind == "ensight") {
          binding["parts"].foreach_child([&](core::DataNode &part) {
            add(ANARI_GEOMETRY,
                part["targetIndex"].getValueOr<size_t>(
                    tsd::core::INVALID_INDEX));
          });
        }
      });
    }
  });
}

size_t countSerializedSubtreeNodes(core::DataNode &node)
{
  size_t count = 1;
  if (auto *children = node.child("children")) {
    children->foreach_child([&](core::DataNode &child) {
      count += countSerializedSubtreeNodes(child);
    });
  }
  return count;
}

bool validateAnimationObjectKey(const std::vector<FileObjectEntry> &entries,
    anari::DataType type,
    size_t index,
    PayloadValidationResult &result)
{
  const auto key = makeKey(type, index);
  const bool found = std::any_of(entries.begin(),
      entries.end(),
      [&](auto &entry) { return sameKey(entry.file, key); });
  if (type == ANARI_UNKNOWN || index == tsd::core::INVALID_INDEX || !found) {
    result.status = PayloadValidationStatus::IncompatibleSchema;
    result.message = "subtree animation references an object outside objectDB";
    return false;
  }
  return true;
}

bool validateSubtreeAnimations(core::DataNode &root,
    std::vector<FileObjectEntry> &entries,
    core::DataNode &subtree,
    PayloadValidationResult &result)
{
  auto *animations = root.child("animations");
  if (!animations)
    return true;

  const auto nodeCount = countSerializedSubtreeNodes(subtree);
  bool ok = true;
  animations->foreach_child([&](core::DataNode &animation) {
    if (!ok)
      return;
    if (!animation.child("name")) {
      result.status = PayloadValidationStatus::MissingRequiredNode;
      result.message = "subtree animation requires a name";
      ok = false;
      return;
    }

    if (auto *bindings = animation.child("objectBindings")) {
      bindings->foreach_child([&](core::DataNode &binding) {
        if (!ok)
          return;
        const auto targetType = static_cast<anari::DataType>(
            binding["targetType"].getValueOr<int>(ANARI_UNKNOWN));
        ok = validateAnimationObjectKey(entries,
            targetType,
            binding["targetIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX),
            result);
        const auto dataType = static_cast<anari::DataType>(
            binding["dataType"].getValueOr<int>(ANARI_UNKNOWN));
        auto *data = binding.child("data");
        if (!ok || !data || !anari::isObject(dataType))
          return;
        const size_t *indices = nullptr;
        size_t count = 0;
        if (!readObjectIndexArray(*data, dataType, indices, count)) {
          result.status = PayloadValidationStatus::IncompatibleSchema;
          result.message =
              "animation keyframe object array has an invalid type";
          ok = false;
          return;
        }
        for (size_t i = 0; ok && i < count; ++i) {
          if (indices[i] != tsd::core::INVALID_INDEX)
            ok = validateAnimationObjectKey(
                entries, dataType, indices[i], result);
        }
      });
    }

    if (auto *bindings = animation.child("transformBindings")) {
      bindings->foreach_child([&](core::DataNode &binding) {
        if (!ok)
          return;
        const auto layerName = binding["layerName"].getValueOr<std::string>("");
        const auto index =
            binding["nodeIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX);
        if (layerName != "__subtree__" || index >= nodeCount) {
          result.status = PayloadValidationStatus::IncompatibleSchema;
          result.message =
              "subtree animation references a node outside the subtree";
          ok = false;
        }
      });
    }

    if (auto *bindings = animation.child("fileBindings")) {
      bindings->foreach_child([&](core::DataNode &binding) {
        if (!ok)
          return;
        const auto kind = binding["kind"].getValueOr<std::string>("");
        if (kind == "spatialField") {
          ok = validateAnimationObjectKey(entries,
              ANARI_VOLUME,
              binding["targetIndex"].getValueOr<size_t>(
                  tsd::core::INVALID_INDEX),
              result);
        } else if (kind == "ensight") {
          auto *parts = binding.child("parts");
          if (!parts || parts->numChildren() == 0) {
            result.status = PayloadValidationStatus::MissingRequiredNode;
            result.message = "EnSight animation requires geometry targets";
            ok = false;
            return;
          }
          parts->foreach_child([&](core::DataNode &part) {
            if (ok)
              ok = validateAnimationObjectKey(entries,
                  ANARI_GEOMETRY,
                  part["targetIndex"].getValueOr<size_t>(
                      tsd::core::INVALID_INDEX),
                  result);
          });
        } else {
          result.status = PayloadValidationStatus::IncompatibleSchema;
          result.message = "unknown subtree file animation kind '" + kind + "'";
          ok = false;
        }
      });
    }
  });
  return ok;
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
    node = scene.insertChildTransformNode(
        parent, math::IDENTITY_MAT4, name.c_str());
    if (node)
      (*node).value().setAsTransform(srtNode->getValueAs<math::mat3>());
  } else {
    const Any value = src["value"].getValue();
    if (value.holdsObject()) {
      auto *target = findTargetEntry(targets, makeKey(value));
      if (!target) {
        errorMessage =
            "subtree node references an object missing from objectDB";
        return {};
      }
      node = scene.insertChildObjectNode(parent,
          value.type(),
          target->target.getAsObjectIndex(),
          name.c_str());
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

bool remapObjectArrayToTarget(core::DataNode &binding,
    const std::vector<TargetObjectEntry> &targets,
    std::string &errorMessage)
{
  const auto type = static_cast<anari::DataType>(
      binding["dataType"].getValueOr<int>(ANARI_UNKNOWN));
  auto *data = binding.child("data");
  if (!data || !anari::isObject(type))
    return true;

  const size_t *indices = nullptr;
  size_t count = 0;
  if (!readObjectIndexArray(*data, type, indices, count)) {
    errorMessage = "animation keyframe object array has an invalid type";
    return false;
  }
  std::vector<size_t> remapped(count, tsd::core::INVALID_INDEX);
  for (size_t i = 0; i < count; ++i) {
    if (indices[i] == tsd::core::INVALID_INDEX)
      continue;
    auto *target = findTargetEntry(targets, makeKey(type, indices[i]));
    if (!target) {
      errorMessage = "animation keyframe object has no import mapping";
      return false;
    }
    remapped[i] = target->target.getAsObjectIndex();
  }
  data->setValueAsArray(type, remapped.data(), remapped.size());
  return true;
}

bool remapFileBindingToTarget(core::DataNode &binding,
    const std::vector<TargetObjectEntry> &targets,
    std::string &errorMessage)
{
  const auto kind = binding["kind"].getValueOr<std::string>("");
  auto remapTarget = [&](core::DataNode &node, anari::DataType type) {
    const auto local =
        node["targetIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX);
    auto *target = findTargetEntry(targets, makeKey(type, local));
    if (!target) {
      errorMessage = "file animation target has no import mapping";
      return false;
    }
    node["targetIndex"] = target->target.getAsObjectIndex();
    return true;
  };

  if (kind == "spatialField")
    return remapTarget(binding, ANARI_VOLUME);
  if (kind == "ensight") {
    bool ok = true;
    binding["parts"].foreach_child([&](core::DataNode &part) {
      if (ok)
        ok = remapTarget(part, ANARI_GEOMETRY);
    });
    return ok;
  }
  errorMessage = "cannot remap file animation kind '" + kind + "'";
  return false;
}

bool remapAnimationsToTarget(core::DataNode &animations,
    Scene &scene,
    const std::vector<TargetObjectEntry> &targets,
    const std::vector<LayerNodeRef> &createdNodes,
    std::string &errorMessage)
{
  bool ok = true;
  animations.foreach_child([&](core::DataNode &animation) {
    if (!ok)
      return;
    if (auto *bindings = animation.child("objectBindings")) {
      bindings->foreach_child([&](core::DataNode &binding) {
        if (!ok)
          return;
        const auto type = static_cast<anari::DataType>(
            binding["targetType"].getValueOr<int>(ANARI_UNKNOWN));
        const auto local =
            binding["targetIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX);
        auto *target = findTargetEntry(targets, makeKey(type, local));
        if (!target) {
          errorMessage = "animation target has no import mapping";
          ok = false;
          return;
        }
        binding["targetIndex"] = target->target.getAsObjectIndex();
        ok = remapObjectArrayToTarget(binding, targets, errorMessage);
      });
    }

    if (auto *bindings = animation.child("transformBindings")) {
      bindings->foreach_child([&](core::DataNode &binding) {
        if (!ok)
          return;
        const auto local =
            binding["nodeIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX);
        if (local >= createdNodes.size() || !createdNodes[local]) {
          errorMessage = "animation transform target is outside the subtree";
          ok = false;
          return;
        }
        auto target = createdNodes[local];
        binding["layerName"] =
            scene.getLayerName((*target).value().layer()).str();
        binding["nodeIndex"] = target.index();
      });
    }

    if (auto *bindings = animation.child("fileBindings")) {
      bindings->foreach_child([&](core::DataNode &binding) {
        if (ok)
          ok = remapFileBindingToTarget(binding, targets, errorMessage);
      });
    }
  });
  return ok;
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

  const auto policy = policyForDesc(desc);
  auto seeds = collectSubtreeSeeds(*layer, root, *scene);

  std::vector<ClosureEntry> baseEntries;
  std::string errorMessage;
  if (!buildClosure(
          *scene, seeds, policy, ObjectKey{}, baseEntries, errorMessage)) {
    tsd::core::logError("[export_Subtree] %s", errorMessage.c_str());
    return false;
  }

  auto subtreeNodes = collectSubtreeNodes(*layer, root);
  AnimationSelection animations;
  if (options.animationManager
      && !selectOwnedAnimations(*options.animationManager,
          *scene,
          baseEntries,
          subtreeNodes,
          options.includeFileBindingAnimations,
          animations,
          errorMessage)) {
    tsd::core::logError("[export_Subtree] %s", errorMessage.c_str());
    return false;
  }
  seeds.insert(seeds.end(),
      animations.dependencySeeds.begin(),
      animations.dependencySeeds.end());

  std::vector<ClosureEntry> entries;
  if (!buildClosure(
          *scene, seeds, policy, ObjectKey{}, entries, errorMessage)) {
    tsd::core::logError("[export_Subtree] %s", errorMessage.c_str());
    return false;
  }

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
      && !writeAnimations(dbRoot["animations"],
          *options.animationManager,
          animations,
          entries,
          subtreeNodes,
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
  if (!filename) {
    tsd::core::logError("[import_Subtree] filename is null");
    return {};
  }

  core::DataTree tree;
  if (!tree.load(filename)) {
    tsd::core::logError("[import_Subtree] failed to load file '%s'", filename);
    return {};
  }

  auto &root = tree.root();
  auto result = validate_SubtreePayload(root, desc);
  if (!result.accepted()) {
    tsd::core::logError("[import_Subtree] payload validation failed: %s",
        result.message.c_str());
    return {};
  }

  if (displayNameOut)
    *displayNameOut = root["displayName"].getValueOr<std::string>("");

  std::vector<FileObjectEntry> fileEntries;
  if (!collectFileObjects(root["objectDB"], fileEntries, result)) {
    tsd::core::logError("[import_Subtree] %s", result.message.c_str());
    return {};
  }

  std::vector<TargetObjectEntry> targetEntries;
  std::vector<Any> createdRefs;
  std::string errorMessage;
  if (!instantiateObjectDB(
          scene, fileEntries, targetEntries, createdRefs, errorMessage)) {
    tsd::core::logError("[import_Subtree] %s", errorMessage.c_str());
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
    tsd::core::logError("[import_Subtree] %s",
        errorMessage.empty() ? "failed to reconstruct subtree"
                             : errorMessage.c_str());
    return {};
  }

  if (options.animationManager) {
    if (auto *animations = root.child("animations")) {
      if (!remapAnimationsToTarget(
              *animations, scene, targetEntries, createdNodes, errorMessage)) {
        scene.removeNode(splicedRoot, false);
        rollbackCreatedObjects(scene, createdRefs);
        tsd::core::logError("[import_Subtree] %s", errorMessage.c_str());
        return {};
      }

      const auto originalCount = options.animationManager->animations().size();
      try {
        animations->foreach_child([&](core::DataNode &animationNode) {
          auto &animation = options.animationManager->addAnimation();
          nodeToAnimation(animationNode, animation, scene);
        });
      } catch (const std::exception &e) {
        while (options.animationManager->animations().size() > originalCount)
          options.animationManager->removeAnimation(originalCount);
        scene.removeNode(splicedRoot, false);
        rollbackCreatedObjects(scene, createdRefs);
        tsd::core::logError(
            "[import_Subtree] failed to reconstruct animations: %s", e.what());
        return {};
      }
    }
  }

  if (auto *layer = (*destinationParent).value().layer())
    scene.signalLayerStructureChanged(layer);

  return splicedRoot;
}

bool export_LayerSubtree(const char *filename, LayerNodeRef root)
{
  return export_Subtree(
      filename, root, {"layer-subtree", schema::LAYER_SUBTREE, false});
}

PayloadValidationResult validate_LayerSubtreePayload(core::DataNode &root)
{
  return validate_SubtreePayload(
      root, {"layer-subtree", schema::LAYER_SUBTREE, false});
}

LayerNodeRef import_LayerSubtree(
    Scene &scene, const char *filename, LayerNodeRef destinationParent)
{
  return import_Subtree(scene,
      filename,
      destinationParent,
      {"layer-subtree", schema::LAYER_SUBTREE, false});
}

} // namespace tsd::io
