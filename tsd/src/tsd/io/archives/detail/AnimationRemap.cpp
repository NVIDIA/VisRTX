// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/archives/detail/AnimationRemap.hpp"
#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"

#include <algorithm>

namespace tsd::io::detail {

namespace {

const ClosureEntry *findSourceEntry(const std::vector<ClosureEntry> &entries,
    anari::DataType type,
    size_t index)
{
  const auto key = makeKey(type, index);
  const auto it = std::find_if(entries.begin(),
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

size_t archiveNodeIndex(const ArchivePlan &plan, LayerNodeRef target)
{
  const auto it = std::find_if(plan.nodes.begin(),
      plan.nodes.end(),
      [&](const auto &node) { return node.source == target; });
  return it == plan.nodes.end() ? tsd::core::INVALID_INDEX : it->archiveIndex;
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
      [&](const auto &entry) { return sameKey(entry.file, key); });
  if (type == ANARI_UNKNOWN || index == tsd::core::INVALID_INDEX || !found) {
    result.status = PayloadValidationStatus::IncompatibleSchema;
    result.message = "subtree animation references an object outside objectDB";
    return false;
  }
  return true;
}

} // namespace

std::vector<ClosureEntry> closureEntriesForPlan(const ArchivePlan &plan)
{
  std::vector<ClosureEntry> entries;
  entries.reserve(plan.objects.size());
  for (const auto &object : plan.objects) {
    entries.push_back({makeKey(object.type, object.sourceIndex),
        object.type,
        object.archiveIndex,
        object.source});
  }
  return entries;
}

bool writeSubtreeAnimations(core::DataNode &animationsNode,
    const animation::AnimationManager &manager,
    const ArchivePlan &plan,
    std::string &errorMessage)
{
  const auto entries = closureEntriesForPlan(plan);
  for (const auto animationIndex : plan.archivedAnimations) {
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
        const auto local = archiveNodeIndex(plan, liveBinding.target());
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

bool remapSubtreeAnimationsToTarget(core::DataNode &animations,
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

bool remapSceneAnimations(core::DataNode &animations,
    const ObjectIndexRemapper &remapObject,
    const LayerNodeIndexRemapper &remapLayerNode,
    std::string &errorMessage)
{
  auto remapRequiredObject =
      [&](core::DataNode &node, const char *field, anari::DataType type) {
        const auto source =
            node[field].getValueOr<size_t>(tsd::core::INVALID_INDEX);
        const auto target = remapObject(type, source);
        if (source != tsd::core::INVALID_INDEX
            && target == tsd::core::INVALID_INDEX) {
          errorMessage = "animation references an excluded object";
          return false;
        }
        node[field] = target;
        return true;
      };

  auto *objects = animations.child("objects");
  if (!objects)
    return true;

  bool ok = true;
  objects->foreach_child([&](core::DataNode &animation) {
    if (!ok)
      return;
    if (auto *bindings = animation.child("objectBindings")) {
      bindings->foreach_child([&](core::DataNode &binding) {
        if (!ok)
          return;
        const auto targetType = static_cast<anari::DataType>(
            binding["targetType"].getValueOr<int>(ANARI_UNKNOWN));
        ok = remapRequiredObject(binding, "targetIndex", targetType);

        const auto dataType = static_cast<anari::DataType>(
            binding["dataType"].getValueOr<int>(ANARI_UNKNOWN));
        auto *data = binding.child("data");
        if (!ok || !data || !anari::isObject(dataType))
          return;
        const size_t *indices = nullptr;
        size_t count = 0;
        if (!readObjectIndexArray(*data, dataType, indices, count)) {
          errorMessage = "animation keyframe object array has an invalid type";
          ok = false;
          return;
        }
        std::vector<size_t> remapped(count, tsd::core::INVALID_INDEX);
        for (size_t i = 0; ok && i < count; ++i) {
          if (indices[i] == tsd::core::INVALID_INDEX)
            continue;
          remapped[i] = remapObject(dataType, indices[i]);
          if (remapped[i] == tsd::core::INVALID_INDEX) {
            errorMessage = "animation keyframe references an excluded object";
            ok = false;
          }
        }
        if (ok)
          data->setValueAsArray(dataType, remapped.data(), remapped.size());
      });
    }

    if (auto *bindings = animation.child("transformBindings")) {
      bindings->foreach_child([&](core::DataNode &binding) {
        if (!ok)
          return;
        const auto layerName = binding["layerName"].getValueOr<std::string>("");
        const auto source =
            binding["nodeIndex"].getValueOr<size_t>(tsd::core::INVALID_INDEX);
        const auto target = remapLayerNode(layerName, source);
        if (source != tsd::core::INVALID_INDEX
            && target == tsd::core::INVALID_INDEX) {
          errorMessage = "animation references an excluded layer node";
          ok = false;
          return;
        }
        binding["nodeIndex"] = target;
      });
    }

    if (auto *bindings = animation.child("fileBindings")) {
      bindings->foreach_child([&](core::DataNode &binding) {
        if (!ok)
          return;
        const auto kind = binding["kind"].getValueOr<std::string>("");
        if (kind == "spatialField") {
          ok = remapRequiredObject(binding, "targetIndex", ANARI_VOLUME);
        } else if (kind == "ensight") {
          auto *parts = binding.child("parts");
          if (!parts) {
            errorMessage = "EnSight animation has no geometry targets";
            ok = false;
            return;
          }
          parts->foreach_child([&](core::DataNode &part) {
            if (ok)
              ok = remapRequiredObject(part, "targetIndex", ANARI_GEOMETRY);
          });
        } else {
          errorMessage = "cannot remap file animation kind '" + kind + "'";
          ok = false;
        }
      });
    }
  });
  return ok;
}

} // namespace tsd::io::detail
