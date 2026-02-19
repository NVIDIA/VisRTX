// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <cstdio>

namespace tsd::io {

SpatialFieldRef import_RAW(Scene &scene, const char *filepath)
{
  std::string file = fileOf(filepath);
  if (file.empty())
    return {};

  int dimX = 0, dimY = 0, dimZ = 0;
  anari::DataType type = ANARI_UNKNOWN;

  {
    for (const auto &str : splitString(file, '_')) {
      int x = 0, y = 0, z = 0;
      if (sscanf(str.c_str(), "%ix%ix%i", &x, &y, &z) == 3) {
        dimX = x;
        dimY = y;
        dimZ = z;
      }

      int bits = 0;
      if (sscanf(str.c_str(), "int%i", &bits) == 1) {
        if (bits == 8)
          type = ANARI_UFIXED8;
        else if (bits == 16)
          type = ANARI_UFIXED16;
        else if (bits == 32)
          type = ANARI_UFIXED32;
      }

      if (sscanf(str.c_str(), "uint%i", &bits) == 1) {
        if (bits == 8)
          type = ANARI_UFIXED8;
        else if (bits == 16)
          type = ANARI_UFIXED16;
        else if (bits == 32)
          type = ANARI_UFIXED32;
      }

      if (dimX && dimY && dimZ && type != ANARI_UNKNOWN)
        break;
    }

    if (type == ANARI_UNKNOWN)
      type = ANARI_FLOAT32;

    if (!(dimX && dimY && dimZ)) {
      logError("[import_RAW] unable to parse info from RAW file: '%s'",
          file.c_str());
      return {};
    }
  }

  auto field = scene.createObject<SpatialField>(
      tokens::spatial_field::structuredRegular);
  field->setName(file.c_str());

  auto voxelArray = scene.createArray(type, dimX, dimY, dimZ);
  auto *voxelData = voxelArray->map();

  auto fileHandle = std::fopen(filepath, "rb");
  size_t size = dimX * size_t(dimY) * dimZ * anari::sizeOf(type);
  if (!std::fread((char *)voxelData, size, 1, fileHandle)) {
    logError("[import_RAW] unable to open RAW file: '%s'", file.c_str());
    voxelArray->unmap();
    scene.removeObject(voxelArray.data());
    scene.removeObject(field.data());
    std::fclose(fileHandle);
    return {};
  }

  std::fclose(fileHandle);
  voxelArray->unmap();

  field->setParameterObject("data", *voxelArray);

  return field;
}

} // namespace tsd::io
