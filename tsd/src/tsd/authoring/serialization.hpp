// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Context.hpp"
#include "tsd/view/Manipulator.hpp"

namespace tsd {

// clang-format off

void objectToNode(const Object &obj, serialization::DataNode &node);
void nodeToObject(serialization::DataNode &node, Object &obj);

void cameraPoseToNode(const manipulators::CameraPose &pose, serialization::DataNode &node);
void nodeToCameraPose(serialization::DataNode &node, manipulators::CameraPose &pose);

void save_Context(Context &ctx, const char *filename);
void save_Context(Context &ctx, serialization::DataNode &root);
void load_Context(Context &ctx, const char *filename);
void load_Context(Context &ctx, serialization::DataNode &root);

void save_Context_Conduit(Context &ctx, const char *filename);
void load_Context_Conduit(Context &ctx, const char *filename);

// clang-format on

} // namespace tsd
