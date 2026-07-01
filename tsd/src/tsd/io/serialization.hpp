// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_io
#include "tsd/io/serialization/Layer.hpp"
#include "tsd/io/serialization/Object.hpp"
#include "tsd/io/serialization/Parameter.hpp"
// tsd_rendering
#include "tsd/rendering/view/Manipulator.hpp"

namespace tsd::io {

void serialize_CameraPose(
    const rendering::CameraPose &pose, core::DataNode &node);
void deserialize_CameraPose(core::DataNode &node, rendering::CameraPose &pose);

} // namespace tsd::io
