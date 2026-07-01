// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/Animation.hpp"
#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/serialization/serialization_animation_archive.hpp"
#include "tsd/io/serialization/serialization_closure.hpp"
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

void rollbackImportedState(Scene &scene,
    animation::AnimationManager *animationManager,
    SubtreeImportResult &result)
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

struct ImportRollbackGuard
{
  ImportRollbackGuard(Scene &scene,
      animation::AnimationManager &animationManager,
      SubtreeImportResult &result);
  ~ImportRollbackGuard();

  TSD_NOT_COPYABLE(ImportRollbackGuard)

  void release();

 private:
  Scene &m_scene;
  animation::AnimationManager &m_animationManager;
  SubtreeImportResult &m_result;
  bool m_active{true};
};

ImportRollbackGuard::ImportRollbackGuard(Scene &scene,
    animation::AnimationManager &animationManager,
    SubtreeImportResult &result)
    : m_scene(scene), m_animationManager(animationManager), m_result(result)
{}

ImportRollbackGuard::~ImportRollbackGuard()
{
  if (m_active)
    rollbackImportedState(m_scene, &m_animationManager, m_result);
}

void ImportRollbackGuard::release()
{
  m_active = false;
}

} // namespace

bool SubtreeImportResult::valid() const
{
  return m_succeeded;
}

SubtreeImportResult import_SubtreeWithOwnership(Scene &scene,
    const char *filename,
    LayerNodeRef destinationParent,
    const SubtreeIODesc &desc,
    std::string *displayNameOut,
    const SubtreeIOOptions &options)
{
  SubtreeImportResult imported;
  if (!filename) {
    tsd::core::logError("[import_Subtree] filename is null");
    return imported;
  }

  core::DataTree tree;
  if (!tree.load(filename)) {
    tsd::core::logError("[import_Subtree] failed to load file '%s'", filename);
    return imported;
  }

  auto &root = tree.root();
  auto validation = validate_SubtreePayload(root, desc);
  if (!validation.accepted()) {
    tsd::core::logError("[import_Subtree] payload validation failed: %s",
        validation.message.c_str());
    return imported;
  }

  if (displayNameOut)
    *displayNameOut = root["displayName"].getValueOr<std::string>("");

  std::vector<FileObjectEntry> fileEntries;
  if (!collectFileObjects(root["objectDB"], fileEntries, validation)) {
    tsd::core::logError("[import_Subtree] %s", validation.message.c_str());
    return imported;
  }

  std::vector<TargetObjectEntry> objectTargets;
  std::string errorMessage;
  if (!instantiateObjectDB(scene,
          fileEntries,
          objectTargets,
          imported.createdObjects,
          errorMessage)) {
    imported.createdObjects.clear();
    tsd::core::logError("[import_Subtree] %s", errorMessage.c_str());
    return imported;
  }

  // Objects-only imports intentionally have no layer root.
  if (!destinationParent) {
    imported.m_succeeded = true;
    return imported;
  }

  std::vector<LayerNodeRef> createdNodes;
  imported.root = spliceSubtree(scene,
      root["subtree"],
      destinationParent,
      objectTargets,
      createdNodes,
      errorMessage);
  if (!imported.root) {
    if (!createdNodes.empty())
      imported.root = createdNodes.front();
    rollbackImportedState(scene, nullptr, imported);
    tsd::core::logError("[import_Subtree] %s",
        errorMessage.empty() ? "failed to reconstruct subtree"
                             : errorMessage.c_str());
    return imported;
  }

  if (options.animationManager) {
    if (auto *animations = root.child("animations")) {
      if (!remapSubtreeAnimationsToTarget(
              *animations, scene, objectTargets, createdNodes, errorMessage)) {
        rollbackImportedState(scene, options.animationManager, imported);
        tsd::core::logError("[import_Subtree] %s", errorMessage.c_str());
        return imported;
      }

      ImportRollbackGuard guard(scene, *options.animationManager, imported);
      animations->foreach_child([&](core::DataNode &animationNode) {
        const auto index = options.animationManager->animations().size();
        auto &animation = options.animationManager->addAnimation();
        imported.createdAnimations.push_back(index);
        nodeToAnimation(animationNode, animation, scene);
      });
      guard.release();
    }
  }

  if (auto *layer = (*destinationParent).value().layer())
    scene.signalLayerStructureChanged(layer);
  imported.m_succeeded = true;
  return imported;
}

void rollback_SubtreeImport(Scene &scene,
    animation::AnimationManager &animationManager,
    SubtreeImportResult &result)
{
  rollbackImportedState(scene, &animationManager, result);
}

} // namespace tsd::io
