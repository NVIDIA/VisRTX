// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Context.hpp"

namespace tsd {

// clang-format off

void generate_cylinders(Context &ctx, bool useDefaultMaterial = false);
void generate_icosphere8(Context &ctx);
void generate_material_orb(Context &ctx);
void generate_monkey(Context &ctx, InstanceNodeRef location = {});
IndexedVectorRef<Volume> generate_noiseVolume(Context &ctx,
    IndexedVectorRef<Array> colors = {},
    IndexedVectorRef<Array> opacities = {});
void generate_randomSpheres(Context &ctx, bool useDefaultMaterial = false);
void generate_rtow(Context &ctx);

// clang-format on

} // namespace tsd
