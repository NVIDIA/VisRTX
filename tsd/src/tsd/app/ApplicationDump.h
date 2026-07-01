// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tsd::core {
struct DataNode;
} // namespace tsd::core

namespace tsd::rendering {
struct CameraPose;
} // namespace tsd::rendering

namespace tsd::app {

struct Context;

bool serialize_ApplicationDump(const Context &context, core::DataNode &root);
bool deserialize_ApplicationDump(Context &context, core::DataNode &root);

void serialize_CameraPose(
    const rendering::CameraPose &pose, core::DataNode &node);
void deserialize_CameraPose(core::DataNode &node, rendering::CameraPose &pose);

} // namespace tsd::app
