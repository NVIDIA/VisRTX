// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/DataTree.hpp"
// tsd_scene
#include "tsd/scene/Scene.hpp"

namespace tsd::io {

void serialize_Object(const scene::Object &object,
    core::DataNode &node,
    bool forceArraysAsProxies = false);
void deserialize_Object(core::DataNode &node, scene::Object &object);
void deserialize_Object(scene::Scene &scene, core::DataNode &node);

} // namespace tsd::io
