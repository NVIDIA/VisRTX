// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/io/serialization/serialization_closure.hpp"

#include <functional>

namespace tsd::animation {
struct AnimationManager;
}

namespace tsd::io::detail {

std::vector<ClosureEntry> closureEntriesForPlan(const ArchivePlan &plan);

bool writeSubtreeAnimations(core::DataNode &animationsNode,
    const animation::AnimationManager &manager,
    const ArchivePlan &plan,
    std::string &errorMessage);

void collectAnimationRefKeys(
    core::DataNode &root, std::vector<ObjectKey> &keys);

bool validateSubtreeAnimations(core::DataNode &root,
    std::vector<FileObjectEntry> &entries,
    core::DataNode &subtree,
    PayloadValidationResult &result);

bool remapSubtreeAnimationsToTarget(core::DataNode &animations,
    Scene &scene,
    const std::vector<TargetObjectEntry> &targets,
    const std::vector<LayerNodeRef> &createdNodes,
    std::string &errorMessage);

using ObjectIndexRemapper = std::function<size_t(anari::DataType, size_t)>;
using LayerNodeIndexRemapper =
    std::function<size_t(const std::string &, size_t)>;

bool remapSceneAnimations(core::DataNode &animations,
    const ObjectIndexRemapper &remapObject,
    const LayerNodeIndexRemapper &remapLayerNode,
    std::string &errorMessage);

} // namespace tsd::io::detail
