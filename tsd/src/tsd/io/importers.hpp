// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Context.hpp"

namespace tsd::io {

using namespace tsd::core;

// clang-format off

void import_ASSIMP(Context &ctx, const char *filename, LayerNodeRef location = {}, bool flatten = false);
void import_DLAF(Context &ctx, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_E57XYZ(Context &ctx, const char *filename, LayerNodeRef location = {});
void import_GLTF(Context &ctx, const char *filename, LayerNodeRef location = {});
void import_HDRI(Context &ctx, const char *filename, LayerNodeRef location = {});
void import_HSMESH(Context &ctx, const char *filename, LayerNodeRef location = {});
void import_NBODY(Context &ctx, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_OBJ(Context &ctx, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_USD(Context &ctx, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_PLY(Context &ctx, const char *filename, LayerNodeRef location = {});
void import_PDB(Context &ctx, const char *filename, LayerNodeRef location = {});
void import_SWC(Context &ctx, const char *filename, LayerNodeRef location = {});
void import_XYZDP(Context &ctx, const char *filename, LayerNodeRef location = {});
void import_PT(Context &ctx, const char *filename, LayerNodeRef location = {});
SpatialFieldRef import_RAW(Context &ctx, const char *filename);
SpatialFieldRef import_FLASH(Context &ctx, const char *filename);
SpatialFieldRef import_NVDB(Context &ctx, const char *filename);
SpatialFieldRef import_MHD(Context &ctx, const char *filename);
SpatialFieldRef import_VTI(Context &ctx, const char *filename);
SpatialFieldRef import_VTU(Context &ctx, const char *filename);

VolumeRef import_volume(Context &ctx,
    const char *filename,
    ArrayRef colors = {},
    ArrayRef opacities = {});

// clang-format on

} // namespace tsd::io
