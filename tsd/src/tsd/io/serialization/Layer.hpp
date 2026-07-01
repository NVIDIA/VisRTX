// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/DataTree.hpp"
// tsd_scene
#include "tsd/scene/Layer.hpp"
namespace tsd::scene {
struct Scene;
} // namespace tsd::scene

namespace tsd::io {

void serialize_Layer(const scene::Layer &layer, core::DataNode &node);
void serialize_LayerSubtree(
    const scene::Layer &layer, scene::LayerNodeRef start, core::DataNode &node);
void deserialize_Layer(
    core::DataNode &node, scene::Layer &layer, scene::Scene &scene);

} // namespace tsd::io
