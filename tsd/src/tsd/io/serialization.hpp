// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Context.hpp"
#include "tsd/rendering/view/Manipulator.hpp"

namespace tsd::io {

using namespace tsd::core;

// clang-format off

void objectToNode(const Object &obj, core::DataNode &node);
void nodeToObject(core::DataNode &node, Object &obj);

void cameraPoseToNode(const rendering::CameraPose &pose, core::DataNode &node);
void nodeToCameraPose(core::DataNode &node, rendering::CameraPose &pose);

void save_Context(Context &ctx, const char *filename);
void save_Context(Context &ctx, core::DataNode &root);
void load_Context(Context &ctx, const char *filename);
void load_Context(Context &ctx, core::DataNode &root);

// clang-format on

} // namespace tsd::io
