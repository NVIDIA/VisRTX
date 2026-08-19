// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/Logging.hpp"
#include "tsd/io/exporters/NanoVdbSidecar.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"

// nanovdb
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/GridStats.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string_view>

namespace tsd::io {

using namespace tsd::core;

namespace {

// nanovdb::io::readGrid does not bound its own reads: on a file too short to
// hold a header it retries against a stream that is already at EOF and never
// returns. A truncated download or an empty placeholder therefore hangs
// whatever called it, with no timeout to escape through, so check the magic
// ourselves before handing the path over.
bool hasNanoVdbMagic(const char *filepath, uint64_t &magic)
{
  magic = 0;
  std::ifstream file(filepath, std::ios::binary);
  if (!file.read(reinterpret_cast<char *>(&magic), sizeof(magic)))
    return false;
  return magic == NANOVDB_MAGIC_NUMB || magic == NANOVDB_MAGIC_FILE
      || magic == NANOVDB_MAGIC_GRID;
}

} // namespace

SpatialFieldRef import_NVDB(Scene &scene, const char *filepath)
{
  std::string file = fileOf(filepath);

  uint64_t magic = 0;
  if (!hasNanoVdbMagic(filepath, magic)) {
    // '.vdb' names a NanoVDB grid here, not an OpenVDB one: tsdVolumeToNanoVDB
    // writes NanoVDB under that extension. Say so, because a file from any
    // other OpenVDB tool lands on this path and the extension gives no hint.
    if (nanovdb::toMagic(magic) == nanovdb::MagicType::OpenVDB) {
      logError(
          "[import_NVDB] '%s' is an OpenVDB file; TSD reads NanoVDB grids, "
          "which tsdVolumeToNanoVDB can convert one into",
          filepath);
    } else {
      logError("[import_NVDB] '%s' is not a NanoVDB file", filepath);
    }
    return {};
  }

  const std::filesystem::path nvdbPath(filepath);
  const auto sidecarPath = makeSidecarPath(nvdbPath);

  std::optional<NanoVdbSidecar> sidecar;
  std::string sidecarError;

  if (std::filesystem::exists(sidecarPath)) {
    sidecar = readSidecar(sidecarPath, sidecarError);
    if (!sidecar) {
      logWarning("[import_NVDB] Found sidecar but failed to parse: %s (%s)",
          sidecarPath.string().c_str(),
          sidecarError.c_str());
    }
  }

  const bool hasRectCoords = sidecar && sidecar->hasCoords();
  auto field = scene.createObject<SpatialField>(hasRectCoords
          ? tokens::spatial_field::nanovdbRectilinear
          : tokens::spatial_field::nanovdb);
  field->setName(file.c_str());

  try {
    auto grid = nanovdb::io::readGrid(filepath);
    auto metadata = grid.gridMetaData();

    if (!metadata) {
      logError(
          "[import_NVDB] '%s' contains no readable grid metadata", filepath);
      scene.removeObject(field.data());
      return {};
    }

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

    auto gridData = scene.createArray(ANARI_UINT8, grid.bufferSize());
    std::memcpy(gridData->map(), grid.data(), grid.bufferSize());
    gridData->unmap();
    field->setParameterObject("data", *gridData);

    if (sidecar) {
      if (!sidecar->dataCentering.empty())
        field->setParameter("dataCentering", sidecar->dataCentering.c_str());

      if (sidecar->roi)
        field->setParameter("roi", *sidecar->roi);

      if (hasRectCoords) {
        auto makeCoordArray = [&](const std::vector<double> &src) {
          auto arr = scene.createArray(ANARI_FLOAT32, src.size());
          auto *dst = arr->mapAs<float>();
          for (size_t i = 0; i < src.size(); ++i)
            dst[i] = static_cast<float>(src[i]);
          arr->unmap();
          return arr;
        };

        auto coordsX = makeCoordArray(sidecar->coordsX);
        auto coordsY = makeCoordArray(sidecar->coordsY);
        auto coordsZ = makeCoordArray(sidecar->coordsZ);
        field->setParameterObject("coordsX", *coordsX);
        field->setParameterObject("coordsY", *coordsY);
        field->setParameterObject("coordsZ", *coordsZ);
      }
    }

    logStatus("[import_NVDB] ...done!");
  } catch (const std::exception &e) {
    using namespace std::string_view_literals;
    const std::string_view msg(e.what());
    logStatus("[import_NVDB] failed: %s", e.what());
    if (msg.find("compression codec was disabled"sv)
        != std::string_view::npos) {
      logStatus(
          "[import_NVDB] '%s' is stored with a compressed codec but TSD was "
          "built without it. Reconfigure with -DTSD_NANOVDB_USE_ZIP=ON (and "
          "ensure zlib is available) to decompress at read time. ANARI is "
          "always handed the uncompressed in-memory grid buffer.",
          filepath);
    }
    scene.removeObject(field.data());
    return {};
  }
  return field;
}

} // namespace tsd::io
