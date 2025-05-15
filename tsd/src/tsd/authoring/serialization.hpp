// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Context.hpp"

namespace tsd {

// clang-format off

void save_Context(Context &ctx, const char *filename);
void save_Context(Context &ctx, serialization::DataNode &root);
void load_Context(Context &ctx, const char *filename);
void load_Context(Context &ctx, serialization::DataNode &root);

void save_Context_Conduit(Context &ctx, const char *filename);
void load_Context_Conduit(Context &ctx, const char *filename);

// clang-format on

} // namespace tsd

