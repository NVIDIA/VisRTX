// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Scene.hpp"

namespace tsd::io {

using namespace tsd::core;

// clang-format off

void import_ASSIMP(Scene &scene, const char *filename, LayerNodeRef location = {}, bool flatten = false);
void import_AXYZ(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_DLAF(Scene &scene, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_E57XYZ(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_GLTF(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_HDRI(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_HSMESH(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_NBODY(Scene &scene, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_OBJ(Scene &scene, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_USD(Scene &scene, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_PLY(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_PDB(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_SWC(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_TRK(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_XYZDP(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_PT(Scene &scene, const char *filename, LayerNodeRef location = {});
SpatialFieldRef import_RAW(Scene &scene, const char *filename);
SpatialFieldRef import_FLASH(Scene &scene, const char *filename);
SpatialFieldRef import_NVDB(Scene &scene, const char *filename);
SpatialFieldRef import_MHD(Scene &scene, const char *filename);
SpatialFieldRef import_VTI(Scene &scene, const char *filename);
SpatialFieldRef import_VTU(Scene &scene, const char *filename);

VolumeRef import_volume(Scene &scene,
    const char *filename,
    ArrayRef colors = {},
    ArrayRef opacities = {});

// clang-format on

} // namespace tsd::io
