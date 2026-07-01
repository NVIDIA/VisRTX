// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <string>

namespace tsd::animation {
struct Animation;
struct AnimationManager;
} // namespace tsd::animation

namespace tsd::core {
struct DataNode;
} // namespace tsd::core

namespace tsd::io {

bool serialize_AnimationArchive(
    const animation::Animation &animation, core::DataNode &archive);
bool validate_AnimationArchive(const animation::AnimationManager &manager,
    core::DataNode &archive,
    std::string *message = nullptr);
animation::Animation *deserialize_AnimationArchive(
    animation::AnimationManager &manager, core::DataNode &archive);
bool save_AnimationArchive(
    const animation::Animation &animation, const char *filename);
animation::Animation *load_AnimationArchive(
    animation::AnimationManager &manager, const char *filename);

} // namespace tsd::io
