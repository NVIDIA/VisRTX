// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/scene/Scene.hpp"
// std
#include <string>
#include <utility>

namespace tsd::io {

using namespace tsd::core;

// clang-format off

// Full scene importers //

void import_AGX(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_ASSIMP(Scene &scene, const char *filename, LayerNodeRef location = {}, bool flatten = false);
void import_AXYZ(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_DLAF(Scene &scene, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_E57XYZ(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_GLTF(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_HDRI(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_HSMESH(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_NBODY(Scene &scene, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_OBJ(Scene &scene, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_PDB(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_PLY(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_POINTSBIN(Scene &scene, const std::vector<std::string> &filepaths, LayerNodeRef location = {});
void import_PT(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_SILO(Scene &scene, const char *filename, LayerNodeRef location);
void import_SMESH(Scene &scene, const char *filename, LayerNodeRef location = {}, bool isAnimation = false);
void import_SWC(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_TRK(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_USD(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_USD2(Scene &scene, const char *filename, LayerNodeRef location = {});
void import_XYZDP(Scene &scene, const char *filename, LayerNodeRef location = {});

// Spatial field importers //

SpatialFieldRef import_RAW(Scene &scene, const char *filename);
SpatialFieldRef import_FLASH(Scene &scene, const char *filename);
SpatialFieldRef import_NVDB(Scene &scene, const char *filename);
SpatialFieldRef import_MHD(Scene &scene, const char *filename);
SpatialFieldRef import_VTI(Scene &scene, const char *filename);
SpatialFieldRef import_VTU(Scene &scene, const char *filename);
SpatialFieldRef import_SILO(Scene &scene, const char *filename);

// clang-format on

///////////////////////////////////////////////////////////////////////////////
// Import volume files (dispatch to different spatial field importers) ////////
///////////////////////////////////////////////////////////////////////////////

VolumeRef import_volume(
    Scene &scene, const char *filename, LayerNodeRef location = {});

VolumeRef import_volume(Scene &scene,
    const char *filename,
    const core::TransferFunction &transferFunction,
    LayerNodeRef location = {});

///////////////////////////////////////////////////////////////////////////////
// Import entire files, dispatches to above importer functions ////////////////
///////////////////////////////////////////////////////////////////////////////

enum class ImporterType
{
  AGX,
  ASSIMP,
  ASSIMP_FLAT,
  AXYZ,
  DLAF,
  E57XYZ,
  GLTF,
  HDRI,
  HSMESH,
  NBODY,
  OBJ,
  PDB,
  PLY,
  POINTSBIN_MULTIFILE,
  PT,
  SILO,
  SMESH,
  SMESH_ANIMATION, // time series version
  SWC,
  TRK,
  USD,
  USD2,
  XYZDP,
  VOLUME,
  TSD,
  XF, // Special case for transfer function files
      // Not an actual scene importer, but used to set transfer function from
      // CLI
  BLANK, // Must be last import type before 'NONE'
  NONE
};

using ImportFile = std::pair<ImporterType, std::string>;
using ImportAnimationFiles = std::pair<ImporterType, std::vector<std::string>>;

void import_file(
    Scene &scene, const ImportFile &file, tsd::core::LayerNodeRef root = {});
void import_file(Scene &scene,
    const ImportFile &file,
    tsd::core::TransferFunction &transferFunction,
    tsd::core::LayerNodeRef root = {});

void import_files(Scene &scene,
    const std::vector<ImportFile> &files,
    tsd::core::LayerNodeRef root = {});
void import_files(Scene &scene,
    const std::vector<ImportFile> &files,
    tsd::core::TransferFunction transferFunction,
    tsd::core::LayerNodeRef root = {});

void import_animations(Scene &scene,
    const std::vector<ImportAnimationFiles> &files,
    tsd::core::LayerNodeRef root = {});

} // namespace tsd::io
