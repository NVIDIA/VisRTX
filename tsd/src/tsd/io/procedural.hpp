// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Context.hpp"

namespace tsd::io {

using namespace tsd::core;

// clang-format off

void generate_cylinders(Context &ctx, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void generate_hdri_dome(Context &ctx, LayerNodeRef location = {});
void generate_hdri_test_image(Context &ctx, LayerNodeRef location = {});
void generate_material_orb(Context &ctx, LayerNodeRef location = {});
void generate_monkey(Context &ctx, LayerNodeRef location = {});
VolumeRef generate_noiseVolume(Context &ctx, LayerNodeRef location = {}, ArrayRef colors = {}, ArrayRef opacities = {});
void generate_randomSpheres(Context &ctx, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void generate_rtow(Context &ctx, LayerNodeRef location = {});

// clang-format on

} // namespace tsd::io
