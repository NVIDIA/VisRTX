// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/FlatMap.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <string>
#include <utility>

namespace tsd::animation {
struct AnimationManager;
} // namespace tsd::animation

namespace tsd::core {
class DataNode;
} // namespace tsd::core

namespace tsd::io {

using namespace tsd::scene;

// clang-format off

// Full scene importers //

void import_AGX(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {});
void import_ASSIMP(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {}, bool flatten = false);
void import_AXYZ(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {});
void import_DLAF(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_E57XYZ(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {});
void import_ENSIGHT(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filename,
    LayerNodeRef location = {},
    int timestep = 0);
void import_ENSIGHT(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filename,
    LayerNodeRef location,
    const std::vector<std::string> &fields,
    int timestep = 0);
void import_ENSIGHT(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filename,
    LayerNodeRef location,
    const std::vector<std::string> &fields,
    const tsd::core::DataNode &settings,
    int timestep = 0);
void import_ENSIGHT(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filename,
    LayerNodeRef location,
    const std::vector<std::string> &fields,
    const tsd::core::DataNode &settings,
    MaterialRef overrideMaterial,
    int timestep = 0);
void import_ENSIGHT(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filename,
    LayerNodeRef location,
    const std::vector<std::string> &fields,
    const tsd::core::DataNode &settings,
    MaterialRef overrideMaterial,
    const tsd::core::FlatMap<std::string, MaterialRef> &perPartMaterials,
    int timestep = 0);
void import_GLTF(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {});
void import_HDRI(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {});
void import_HSMESH(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {});
void import_NBODY(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_OBJ(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {}, bool useDefaultMaterial = false);
void import_PDB(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {});
void import_PLY(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {});
void import_POINTSBIN(Scene &scene, tsd::animation::AnimationManager &animMgr, const std::vector<std::string> &filepaths, LayerNodeRef location = {});
void import_PT(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {});
void import_SILO(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location);
void import_SMESH(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {}, bool isAnimation = false);
void import_SWC(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {});
void import_TRK(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {});
void import_USD(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {});
void import_VTP(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filepath, LayerNodeRef location = {});
void import_VTU(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filepath, LayerNodeRef location);
void import_XYZDP(Scene &scene, tsd::animation::AnimationManager &animMgr, const char *filename, LayerNodeRef location = {});

// Spatial field importers //

// Dispatch to the appropriate spatial field importer based on file extension.
// Supports: .raw, .flash/.hdf5, .nvdb, .mhd, .vtu, .silo/.sil
// Note: .vti is not supported here; use import_volume() for VTI files.
SpatialFieldRef import_spatial_field(Scene &scene, const char *filename);

SpatialFieldRef import_RAW(Scene &scene, const char *filename);
SpatialFieldRef import_FLASH(Scene &scene, const char *filename);
SpatialFieldRef import_NVDB(Scene &scene, const char *filename);
SpatialFieldRef import_MHD(Scene &scene, const char *filename);
SpatialFieldRef import_VTI(Scene &scene,
    const char *filename,
    LayerNodeRef location = {},
    std::vector<SpatialFieldRef> *extraFields = nullptr);
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

// Load a sequence of spatial field files as an animated volume.
// Creates a Volume for the first frame and registers a CallbackBinding on a
// new Animation in animMgr so that subsequent frames are loaded on demand.
// Time t=0.0 selects files[0]; t=1.0 selects files[N-1].
VolumeRef import_volume_animation(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const std::vector<std::string> &files,
    const TransferFunction &transferFunction,
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
  ENSIGHT,
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
  VTP,
  VTU,
  XYZDP,
  VOLUME,
  VOLUME_ANIMATION, // time series of spatial field files
  TSD,
  XF, // Special case for transfer function files
      // Not an actual scene importer, but used to set transfer function from
      // CLI
  BLANK, // Must be last import type before 'NONE'
  NONE
};

using ImportFile = std::pair<ImporterType, std::string>;
using ImportAnimationFiles = std::pair<ImporterType, std::vector<std::string>>;

void import_file(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const ImportFile &file,
    tsd::scene::LayerNodeRef root = {});
void import_file(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const ImportFile &file,
    tsd::core::TransferFunction &transferFunction,
    tsd::scene::LayerNodeRef root = {});

void import_files(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const std::vector<ImportFile> &files,
    tsd::scene::LayerNodeRef root = {});
void import_files(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const std::vector<ImportFile> &files,
    tsd::core::TransferFunction transferFunction,
    tsd::scene::LayerNodeRef root = {});

void import_animations(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const std::vector<ImportAnimationFiles> &files,
    tsd::scene::LayerNodeRef root = {});

void import_animations(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const std::vector<ImportAnimationFiles> &files,
    const tsd::core::TransferFunction &transferFunction,
    tsd::scene::LayerNodeRef root = {});

} // namespace tsd::io
