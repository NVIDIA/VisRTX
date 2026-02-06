// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"

// nanovdb
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/GridStats.h>

#include <limits>

namespace tsd::io {

using namespace tsd::core;

SpatialFieldRef import_NVDB(Scene &scene, const char *filepath)
{
  std::string file = fileOf(filepath);
  if (file.empty())
    return {};
  auto field = scene.createObject<SpatialField>(tokens::spatial_field::nanovdb);
  field->setName(file.c_str());

  try {
    auto grid = nanovdb::io::readGrid(filepath);
    auto metadata = grid.gridMetaData();

    bool hasActiveVoxels = false;
    switch (metadata->gridType()) {
    case nanovdb::GridType::Fp4:
      hasActiveVoxels = grid.grid<nanovdb::Fp4>()->activeVoxelCount() > 0;
      break;
    case nanovdb::GridType::Fp8:
      hasActiveVoxels = grid.grid<nanovdb::Fp8>()->activeVoxelCount() > 0;
      break;
    case nanovdb::GridType::Fp16:
      hasActiveVoxels = grid.grid<nanovdb::Fp16>()->activeVoxelCount() > 0;
      break;
    case nanovdb::GridType::FpN:
      hasActiveVoxels = grid.grid<nanovdb::FpN>()->activeVoxelCount() > 0;
      break;
    case nanovdb::GridType::Float:
      hasActiveVoxels = grid.grid<float>()->activeVoxelCount() > 0;
      break;
    default:
      break;
    }

    if (!hasActiveVoxels) {
      logWarning("[import_NVDB] no active voxels in '%s'", filepath);
      if (TSD_NANOVDB_SKIP_INVALID_VOLUMES) {
        logStatus(
            "[import_NVDB] skipping due to TSD_NANOVDB_SKIP_INVALID_VOLUMES");
        scene.removeObject(field.data());
        return {};
      }
    }

    if (!metadata->hasMinMax()) {
      switch (metadata->gridType()) {
      case nanovdb::GridType::Fp4: {
        nanovdb::tools::updateGridStats(
            grid.grid<nanovdb::Fp4>(), nanovdb::tools::StatsMode::MinMax);
        break;
      }
      case nanovdb::GridType::Fp8: {
        nanovdb::tools::updateGridStats(
            grid.grid<nanovdb::Fp8>(), nanovdb::tools::StatsMode::MinMax);
        break;
      }
      case nanovdb::GridType::Fp16: {
        nanovdb::tools::updateGridStats(
            grid.grid<nanovdb::Fp16>(), nanovdb::tools::StatsMode::MinMax);
        break;
      }
      case nanovdb::GridType::FpN: {
        nanovdb::tools::updateGridStats(
            grid.grid<nanovdb::FpN>(), nanovdb::tools::StatsMode::MinMax);
        break;
      }
      case nanovdb::GridType::Float: {
        nanovdb::tools::updateGridStats(
            grid.grid<float>(), nanovdb::tools::StatsMode::MinMax);
        break;
      }
      default:
        break;
      }
    }

    float2 minMax(std::numeric_limits<float>::max(),
        std::numeric_limits<float>::lowest());
    switch (metadata->gridType()) {
    case nanovdb::GridType::Fp4: {
      minMax.x = grid.grid<nanovdb::Fp4>()->tree().root().minimum();
      minMax.y = grid.grid<nanovdb::Fp4>()->tree().root().maximum();
      break;
    }
    case nanovdb::GridType::Fp8: {
      minMax.x = grid.grid<nanovdb::Fp8>()->tree().root().minimum();
      minMax.y = grid.grid<nanovdb::Fp8>()->tree().root().maximum();
      break;
    }
    case nanovdb::GridType::Fp16: {
      minMax.x = grid.grid<nanovdb::Fp16>()->tree().root().minimum();
      minMax.y = grid.grid<nanovdb::Fp16>()->tree().root().maximum();
      break;
    }
    case nanovdb::GridType::FpN: {
      minMax.x = grid.grid<nanovdb::FpN>()->tree().root().minimum();
      minMax.y = grid.grid<nanovdb::FpN>()->tree().root().maximum();
      break;
    }
    case nanovdb::GridType::Float: {
      minMax.x = grid.grid<float>()->tree().root().minimum();
      minMax.y = grid.grid<float>()->tree().root().maximum();
      break;
    }
    default:
      break;
    }

    if (minMax.x <= minMax.y) {
      field->setParameter("range", minMax);
      logStatus("data range %f %f", minMax.x, minMax.y);
    } else {
      logStatus("No data range found.");
    }

    auto gridData = scene.createArray(ANARI_UINT8, grid.size());
    std::memcpy(gridData->map(), grid.data(), grid.size());
    gridData->unmap();
    field->setParameterObject("data", *gridData);

    logStatus("[import_NVDB] ...done!");
  } catch (const std::exception &e) {
    logStatus("[import_NVDB] failed: %s", e.what());
    scene.removeObject(field.data());
    return {};
  }
  return field;
}

} // namespace tsd
