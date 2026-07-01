// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/io/serialization/serialization_closure.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"

#include <algorithm>

namespace tsd::io {

using namespace tsd::io::detail;

namespace {

ClosurePolicy closurePolicy(ArchiveObjectPolicy policy)
{
  return policy == ArchiveObjectPolicy::LightsOnly ? lightRigPolicy()
                                                   : layerSubtreePolicy();
}

bool containsNode(const std::vector<ArchiveNode> &nodes, LayerNodeRef target)
{
  return std::any_of(nodes.begin(), nodes.end(), [&](const auto &node) {
    return node.source == target;
  });
}

bool collectSubtree(Scene &scene,
    LayerNodeRef root,
    ArchivePlan &plan,
    std::vector<Object *> &seeds,
    std::string &error)
{
  if (!root) {
    error = "archive plan requires a valid subtree root";
    return false;
  }

  auto *layer = (*root).value().layer();
  if (!layer || layer->scene() != &scene) {
    error = "archive plan subtree root belongs to a different Scene";
    return false;
  }

  plan.root = root;
  size_t archiveIndex = 0;
  layer->traverse(root, [&](LayerNode &node, int) {
    auto nodeRef = layer->at(node.index());
    plan.nodes.push_back({nodeRef, archiveIndex++});

    if (node->isObject()) {
      if (auto *object = node->getObject())
        seeds.push_back(object);
    }
    for (const auto &parameter : node->getInstanceParameters()) {
      if (parameter.second.holdsObject()) {
        if (auto *object = scene.getObject(parameter.second))
          seeds.push_back(object);
      }
    }
    return true;
  });
  return true;
}

bool classifyObjectTarget(const std::vector<ClosureEntry> &entries,
    anari::DataType type,
    size_t index,
    bool &inside,
    bool &outside,
    ArchivePlanResult &result)
{
  if (type == ANARI_UNKNOWN || index == tsd::core::INVALID_INDEX) {
    result.status = ArchivePlanStatus::InvalidAnimationTarget;
    result.message = "animation has an invalid object target";
    return false;
  }

  const auto key = makeKey(type, index);
  if (std::any_of(entries.begin(), entries.end(), [&](const auto &entry) {
        return sameKey(entry.source, key);
      }))
    inside = true;
  else
    outside = true;
  return true;
}

bool classifyFileBinding(const animation::FileBinding &binding,
    const std::vector<ClosureEntry> &entries,
    bool &inside,
    bool &outside,
    ArchivePlanResult &result)
{
  core::DataTree scratch;
  binding.toDataNode(scratch.root());

  if (binding.kind() == "spatialField") {
    return classifyObjectTarget(entries,
        ANARI_VOLUME,
        scratch.root()["targetIndex"].getValueOr<size_t>(
            tsd::core::INVALID_INDEX),
        inside,
        outside,
        result);
  }

  if (binding.kind() == "ensight") {
    auto *parts = scratch.root().child("parts");
    if (!parts || parts->numChildren() == 0) {
      result.status = ArchivePlanStatus::InvalidAnimationTarget;
      result.message = "EnSight animation has no geometry targets";
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
            result);
    });
    return ok;
  }

  result.status = ArchivePlanStatus::UnsupportedFileBinding;
  result.message = "cannot determine ownership of file animation kind '"
      + binding.kind() + "'";
  return false;
}

bool collectAnimationDependencies(const animation::Animation &animation,
    Scene &scene,
    std::vector<Object *> &dependencies,
    ArchivePlanResult &result)
{
  for (const auto &binding : animation.objectParameterBindings()) {
    if (!anari::isObject(binding.type()))
      continue;

    const auto *indices = static_cast<const size_t *>(binding.data().data());
    for (size_t i = 0; indices && i < binding.data().size(); ++i) {
      if (indices[i] == tsd::core::INVALID_INDEX)
        continue;
      auto *dependency = scene.getObject(binding.type(), indices[i]);
      if (!dependency) {
        result.status = ArchivePlanStatus::InvalidAnimationTarget;
        result.message = "animation '" + animation.name()
            + "' references a missing keyframe object";
        return false;
      }
      dependencies.push_back(dependency);
    }
  }
  return true;
}

bool classifyAnimations(const animation::AnimationManager &manager,
    Scene &scene,
    const std::vector<ClosureEntry> &baseEntries,
    const ArchivePlanOptions &options,
    ArchivePlanResult &result,
    std::vector<Object *> &seeds)
{
  const auto &animations = manager.animations();
  for (size_t animationIndex = 0; animationIndex < animations.size();
       ++animationIndex) {
    const auto &animation = animations[animationIndex];
    bool inside = false;
    bool outside = false;

    for (const auto &binding : animation.objectParameterBindings()) {
      auto *target = binding.target();
      if (!classifyObjectTarget(baseEntries,
              target ? target->type() : ANARI_UNKNOWN,
              target ? target->index() : tsd::core::INVALID_INDEX,
              inside,
              outside,
              result))
        return false;
    }

    for (const auto &binding : animation.transformBindings()) {
      const auto target = binding.target();
      if (!target) {
        result.status = ArchivePlanStatus::InvalidAnimationTarget;
        result.message = "animation has an invalid transform target";
        return false;
      }
      if (containsNode(result.plan.nodes, target))
        inside = true;
      else
        outside = true;
    }

    for (const auto &binding : animation.fileBindings()) {
      if (!classifyFileBinding(*binding, baseEntries, inside, outside, result))
        return false;
    }

    std::vector<Object *> dependencies;
    if (!collectAnimationDependencies(animation, scene, dependencies, result))
      return false;
    for (auto *dependency : dependencies) {
      result.plan.animationDependencies.push_back({animationIndex, dependency});
    }

    if (inside && outside) {
      result.status = ArchivePlanStatus::MixedAnimationTargets;
      result.message = "animation '" + animation.name()
          + "' has targets both inside and outside the planned subtree";
      return false;
    }
    if (!inside)
      continue;

    result.plan.ownedAnimations.push_back(animationIndex);
    const bool archived =
        options.fileBindings == FileBindingArchivePolicy::Include
        || animation.fileBindings().empty();
    if (archived) {
      result.plan.archivedAnimations.push_back(animationIndex);
      seeds.insert(seeds.end(), dependencies.begin(), dependencies.end());
    }
  }
  return true;
}

bool closureContains(
    const std::vector<ClosureEntry> &entries, const ObjectKey &key)
{
  return std::any_of(entries.begin(), entries.end(), [&](const auto &entry) {
    return sameKey(entry.source, key);
  });
}

} // namespace

bool ArchivePlan::containsObject(const Object *object) const
{
  return object
      && std::any_of(objects.begin(), objects.end(), [&](const auto &entry) {
           return entry.source == object;
         });
}

bool ArchivePlanResult::accepted() const
{
  return status == ArchivePlanStatus::Valid;
}

ArchivePlanResult plan_SubtreeArchive(
    Scene &scene, LayerNodeRef root, const ArchivePlanOptions &options)
{
  ArchivePlanResult result;
  std::vector<Object *> seeds;
  if (!collectSubtree(scene, root, result.plan, seeds, result.message)) {
    result.status = ArchivePlanStatus::InvalidSubtree;
    return result;
  }

  const auto policy = closurePolicy(options.objectPolicy);
  if (options.animationManager && options.animationManager->scene() != &scene) {
    result.status = ArchivePlanStatus::InvalidAnimationTarget;
    result.message = "animation manager belongs to a different Scene";
    return result;
  }
  std::vector<ClosureEntry> baseEntries;
  if (!buildClosure(
          scene, seeds, policy, ObjectKey{}, baseEntries, result.message)) {
    result.status = ArchivePlanStatus::ObjectClosureFailure;
    return result;
  }

  if (options.animationManager
      && !classifyAnimations(*options.animationManager,
          scene,
          baseEntries,
          options,
          result,
          seeds))
    return result;

  std::vector<ClosureEntry> entries;
  if (!buildClosure(
          scene, seeds, policy, ObjectKey{}, entries, result.message)) {
    result.status = ArchivePlanStatus::ObjectClosureFailure;
    return result;
  }

  result.plan.objects.reserve(entries.size());
  for (const auto &entry : entries) {
    result.plan.objects.push_back({entry.object,
        entry.objectType,
        entry.source.index,
        entry.localIndex,
        !closureContains(baseEntries, entry.source)});
  }
  return result;
}

} // namespace tsd::io
