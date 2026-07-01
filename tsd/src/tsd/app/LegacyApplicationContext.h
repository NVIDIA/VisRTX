// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tsd::animation {
struct AnimationManager;
} // namespace tsd::animation

namespace tsd::core {
struct DataNode;
} // namespace tsd::core

namespace tsd::scene {
struct Scene;
} // namespace tsd::scene

namespace tsd::app {

struct Context;

namespace detail {

void serializeLegacyApplicationContext(Context &context, core::DataNode &node);
bool deserializeLegacyApplicationContext(
    Context &context, core::DataNode &node);
bool deserializeLegacySceneState(scene::Scene &scene,
    animation::AnimationManager &animationManager,
    core::DataNode &node);

} // namespace detail

} // namespace tsd::app
