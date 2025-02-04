// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Context.hpp"

namespace tsd {

// clang-format off

void import_ASSIMP(Context &ctx, const char *filename, InstanceNode::Ref location = {}, bool flatten = false);
void import_DLAF(Context &ctx, const char *filename, InstanceNode::Ref location = {}, bool useDefaultMaterial = false);
void import_HDRI(Context &ctx, const char *filename, InstanceNode::Ref location = {});
void import_NBODY(Context &ctx, const char *filename, InstanceNode::Ref location = {}, bool useDefaultMaterial = false);
void import_OBJ(Context &ctx, const char *filename, InstanceNode::Ref location = {}, bool useDefaultMaterial = false);
void import_PLY(Context &ctx, const char *filename, InstanceNode::Ref location = {});
SpatialFieldRef import_RAW(Context &ctx, const char *filename);
SpatialFieldRef import_FLASH(Context &ctx, const char *filename);
SpatialFieldRef import_NVDB(Context &ctx, const char *filename);

VolumeRef import_volume(Context &ctx,
    const char *filename,
    ArrayRef colors,
    ArrayRef opacities);

// clang-format on

} // namespace tsd
