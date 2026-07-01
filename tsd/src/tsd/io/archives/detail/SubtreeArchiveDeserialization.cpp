// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/archives/detail/AnimationRemap.hpp"
#include "tsd/io/archives/detail/ArchiveClosure.hpp"
#include "tsd/io/serialization/serialization_internal.hpp"

namespace tsd::io {

using namespace tsd::io::detail;

namespace {

void applyInstanceParameters(core::DataNode &source,
    LayerNodeData &target,
    const std::vector<TargetObjectEntry> &objectTargets)
{
  auto *parameters = source.child("instanceParameters");
  if (!parameters)
    return;

  parameters->foreach_child([&](core::DataNode &parameter) {
    Any value = parameter.getValue();
    if (value.holdsObject()) {
      if (auto *entry = findTargetEntry(objectTargets, makeKey(value)))
        value = Any(value.type(), entry->target.getAsObjectIndex());
    }
    target.setInstanceParameter(parameter.name(), value);
  });
}

LayerNodeRef spliceSubtree(Scene &scene,
    core::DataNode &source,
    LayerNodeRef parent,
    const std::vector<TargetObjectEntry> &objectTargets,
    std::vector<LayerNodeRef> &createdNodes,
    std::string &errorMessage)
{
  const auto name = source["name"].getValueAs<std::string>();
  LayerNodeRef node;
  if (auto *srt = source.child("transformSRT")) {
    node = scene.insertChildTransformNode(
        parent, math::IDENTITY_MAT4, name.c_str());
    if (node)
      (*node).value().setAsTransform(srt->getValueAs<math::mat3>());
  } else {
    const Any value = source["value"].getValue();
    if (value.holdsObject()) {
      auto *target = findTargetEntry(objectTargets, makeKey(value));
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
    errorMessage = "failed to create subtree node";
    return {};
  }

  createdNodes.push_back(node);
  (*node).value().setEnabled(source["enabled"].getValueOr(true));
  applyInstanceParameters(source, (*node).value(), objectTargets);

  if (auto *children = source.child("children")) {
    bool ok = true;
    children->foreach_child([&](core::DataNode &child) {
      if (ok
          && !spliceSubtree(
              scene, child, node, objectTargets, createdNodes, errorMessage))
        ok = false;
    });
    if (!ok)
      return {};
  }
  return node;
}

void rollbackDeserializedState(Scene &scene,
    animation::AnimationManager *animationManager,
    SubtreeArchiveResult &result)
{
  if (animationManager) {
    for (auto it = result.createdAnimations.rbegin();
         it != result.createdAnimations.rend();
         ++it)
      animationManager->removeAnimation(*it);
  }
  if (result.root)
    scene.removeNode(result.root, false);
  rollbackCreatedObjects(scene, result.createdObjects);
  result = {};
}

struct DeserializationRollbackGuard
{
  DeserializationRollbackGuard(Scene &scene,
      animation::AnimationManager &animationManager,
      SubtreeArchiveResult &result);
  ~DeserializationRollbackGuard();

  TSD_NOT_COPYABLE(DeserializationRollbackGuard)

  void release();

 private:
  Scene &m_scene;
  animation::AnimationManager &m_animationManager;
  SubtreeArchiveResult &m_result;
  bool m_active{true};
};

DeserializationRollbackGuard::DeserializationRollbackGuard(Scene &scene,
    animation::AnimationManager &animationManager,
    SubtreeArchiveResult &result)
    : m_scene(scene), m_animationManager(animationManager), m_result(result)
{}

DeserializationRollbackGuard::~DeserializationRollbackGuard()
{
  if (m_active)
    rollbackDeserializedState(m_scene, &m_animationManager, m_result);
}

void DeserializationRollbackGuard::release()
{
  m_active = false;
}

} // namespace

bool SubtreeArchiveResult::valid() const
{
  return m_succeeded;
}

SubtreeArchiveResult deserialize_SubtreeArchiveContent(Scene &scene,
    core::DataNode &root,
    LayerNodeRef destinationParent,
    const SubtreeArchiveContentDesc &desc,
    std::string *displayNameOut,
    const SubtreeArchiveContentOptions &options)
{
  SubtreeArchiveResult deserialized;
  auto validation = validate_SubtreeArchiveContent(root, desc);
  if (!validation.accepted()) {
    tsd::core::logError(
        "[deserialize_SubtreeArchiveContent] Archive validation failed: %s",
        validation.message.c_str());
    return deserialized;
  }

  if (displayNameOut)
    *displayNameOut = root["displayName"].getValueOr<std::string>("");

  std::vector<FileObjectEntry> fileEntries;
  if (!collectFileObjects(root["objectDB"], fileEntries, validation)) {
    tsd::core::logError(
        "[deserialize_SubtreeArchiveContent] %s", validation.message.c_str());
    return deserialized;
  }

  std::vector<TargetObjectEntry> objectTargets;
  std::string errorMessage;
  if (!instantiateObjectDB(scene,
          fileEntries,
          objectTargets,
          deserialized.createdObjects,
          errorMessage)) {
    deserialized.createdObjects.clear();
    tsd::core::logError(
        "[deserialize_SubtreeArchiveContent] %s", errorMessage.c_str());
    return deserialized;
  }

  // Objects-only deserialization intentionally has no layer root.
  if (!destinationParent) {
    deserialized.m_succeeded = true;
    return deserialized;
  }

  std::vector<LayerNodeRef> createdNodes;
  deserialized.root = spliceSubtree(scene,
      root["subtree"],
      destinationParent,
      objectTargets,
      createdNodes,
      errorMessage);
  if (!deserialized.root) {
    if (!createdNodes.empty())
      deserialized.root = createdNodes.front();
    rollbackDeserializedState(scene, nullptr, deserialized);
    tsd::core::logError("[deserialize_SubtreeArchiveContent] %s",
        errorMessage.empty() ? "failed to reconstruct subtree"
                             : errorMessage.c_str());
    return deserialized;
  }

  if (options.animationManager) {
    if (auto *animations = root.child("animations")) {
      if (!remapSubtreeAnimationsToTarget(
              *animations, scene, objectTargets, createdNodes, errorMessage)) {
        rollbackDeserializedState(
            scene, options.animationManager, deserialized);
        tsd::core::logError(
            "[deserialize_SubtreeArchiveContent] %s", errorMessage.c_str());
        return deserialized;
      }

      DeserializationRollbackGuard guard(
          scene, *options.animationManager, deserialized);
      animations->foreach_child([&](core::DataNode &animationNode) {
        const auto index = options.animationManager->animations().size();
        auto &animation = options.animationManager->addAnimation();
        deserialized.createdAnimations.push_back(index);
        nodeToAnimation(animationNode, animation, scene);
      });
      guard.release();
    }
  }

  if (auto *layer = (*destinationParent).value().layer())
    scene.signalLayerStructureChanged(layer);
  deserialized.m_succeeded = true;
  return deserialized;
}

void rollback_SubtreeArchiveContent(Scene &scene,
    animation::AnimationManager &animationManager,
    SubtreeArchiveResult &result)
{
  rollbackDeserializedState(scene, &animationManager, result);
}

} // namespace tsd::io
