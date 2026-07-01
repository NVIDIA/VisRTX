// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "LegacyApplicationContext.h"

#include "Context.h"
// tsd_core
#include "tsd/core/DataTree.hpp"
// tsd_io
#include "tsd/io/archives/AnimationManagerArchive.hpp"
#include "tsd/io/archives/SceneArchive.hpp"

namespace tsd::app::detail {

namespace {

void copyArchiveContents(
    core::DataNode &destination, const core::DataNode &source)
{
  destination.reset();
  for (size_t i = 0; i < source.numChildren(); ++i) {
    if (auto *child = source.child(i))
      destination.append(child->name()) = *child;
  }
}

} // namespace

void serializeLegacyApplicationContext(Context &context, core::DataNode &node)
{
  core::DataTree animationManagerArchive;
  if (io::serialize_SceneAndAnimationManagerArchives(context.tsd.scene,
          context.tsd.animationMgr,
          node,
          animationManagerArchive.root())) {
    copyArchiveContents(node["animations"], animationManagerArchive.root());
  }
}

bool deserializeLegacyApplicationContext(Context &context, core::DataNode &node)
{
  return deserializeLegacySceneState(
      context.tsd.scene, context.tsd.animationMgr, node);
}

bool deserializeLegacySceneState(scene::Scene &scene,
    animation::AnimationManager &animationManager,
    core::DataNode &node)
{
  auto *context = node.child("context");
  auto &payload = context ? *context : node;
  if (!io::deserialize_SceneArchive(scene, payload))
    return false;
  if (auto *animations = payload.child("animations")) {
    return io::deserialize_AnimationManagerArchive(
        animationManager, *animations);
  }
  return true;
}

} // namespace tsd::app::detail
