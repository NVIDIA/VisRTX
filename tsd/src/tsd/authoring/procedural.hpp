// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Context.hpp"

namespace tsd {

// clang-format off

void generate_cylinders(Context &ctx, InstanceNode::Ref location = {}, bool useDefaultMaterial = false);
void generate_material_orb(Context &ctx, InstanceNode::Ref location = {});
void generate_monkey(Context &ctx, InstanceNode::Ref location = {});
VolumeRef generate_noiseVolume(Context &ctx, InstanceNode::Ref location = {}, ArrayRef colors = {}, ArrayRef opacities = {});
void generate_randomSpheres(Context &ctx, InstanceNode::Ref location = {}, bool useDefaultMaterial = false);
void generate_rtow(Context &ctx, InstanceNode::Ref location = {});

// clang-format on

} // namespace tsd
