// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/DataTree.hpp"
// tsd_scene
#include "tsd/scene/Parameter.hpp"

namespace tsd::io {

void serialize_Parameter(
    const scene::Parameter &parameter, core::DataNode &node);
void deserialize_Parameter(core::DataNode &node, scene::Parameter &parameter);

} // namespace tsd::io
