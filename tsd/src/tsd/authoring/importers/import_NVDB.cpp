// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"
// nanovdb
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/util/IO.h>

namespace tsd {

SpatialFieldRef import_NVDB(Context &ctx, const char *filepath)
{
  std::string file = fileOf(filepath);
  if (file.empty())
    return {};

  auto field = ctx.createObject<SpatialField>(tokens::spatial_field::nanovdb);
  field->setName(file.c_str());

  auto grid = nanovdb::io::readGrid(filepath);
  
  auto gridData = ctx.createArray(ANARI_FLOAT32, grid.size());
  {
    auto *dst = (uint8_t *)gridData->map();
    std::memcpy(dst, grid.data(), grid.size());
  }

  field->setParameterObject("gridData", *gridData);

  logStatus("[import_NVDB] ...done!");

  return field;
}

} // namespace tsd
