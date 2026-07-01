// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "LegacyApplicationContext.h"

#include "Context.h"
#include "tsd/io/serialization/serialization_internal.hpp"

namespace tsd::app::detail {

void serializeLegacyApplicationContext(Context &context, core::DataNode &node)
{
  io::detail::LegacySceneSerializationOptions options;
  options.animationManager = &context.tsd.animationMgr;
  io::detail::serializeLegacyScenePayload(context.tsd.scene, node, options);
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
  return io::detail::tryDeserializeLegacyScenePayload(
      scene, node, nullptr, &animationManager);
}

} // namespace tsd::app::detail
