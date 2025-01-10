// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Context.hpp"

namespace tsd {

// clang-format off

void save_Context(Context &ctx, const char *filename);
void import_Context(Context &ctx, const char *filename);

// clang-format on

} // namespace tsd

