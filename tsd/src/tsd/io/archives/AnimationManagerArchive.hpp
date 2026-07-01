// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tsd::animation {
struct AnimationManager;
} // namespace tsd::animation

namespace tsd::core {
struct DataNode;
} // namespace tsd::core

namespace tsd::io {

bool serialize_AnimationManagerArchive(
    const animation::AnimationManager &manager, core::DataNode &archive);
bool deserialize_AnimationManagerArchive(
    animation::AnimationManager &manager, core::DataNode &archive);
bool save_AnimationManagerArchive(
    const animation::AnimationManager &manager, const char *filename);
bool load_AnimationManagerArchive(
    animation::AnimationManager &manager, const char *filename);

} // namespace tsd::io
