// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
// std
#include <cstdio>

namespace tsd {

IndexedVectorRef<Volume> import_RAW(Context &ctx,
    const char *filepath,
    IndexedVectorRef<Array> colorArray,
    IndexedVectorRef<Array> opacityArray)
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
      printf("unable to parse info from RAW file: '%s'\n", file.c_str());
      return {};
    }
  }

  auto field =
      ctx.createObject<SpatialField>(tokens::spatial_field::structuredRegular);
  field->setName(file.c_str());

  auto voxelArray = ctx.createArray(type, dimX, dimY, dimZ);
  auto *voxelData = voxelArray->map();

  auto fileHandle = std::fopen(filepath, "rb");
  size_t size = dimX * size_t(dimY) * dimZ * anari::sizeOf(type);
  if (!std::fread((char *)voxelData, size, 1, fileHandle)) {
    printf("unable to open RAW file: '%s'\n", file.c_str());
    voxelArray->unmap();
    ctx.removeObject(*voxelArray);
    ctx.removeObject(*field);
    std::fclose(fileHandle);
    return {};
  }

  std::fclose(fileHandle);
  voxelArray->unmap();

  field->setParameter("origin"_t, float3(-dimX, -dimY, -dimZ));
  field->setParameter("spacing"_t, 2.f / float3(dimX, dimY, dimZ));
  field->setParameterObject("data"_t, *voxelArray);

  auto volume = ctx.createObject<Volume>(tokens::volume::transferFunction1D);
  volume->setName(file.c_str());
  volume->setParameterObject("value", *field);
  volume->setParameterObject("color", *colorArray);
  volume->setParameterObject("opacity", *opacityArray);
  volume->setParameter("densityScale", 0.1f);

  ctx.tree.insert_last_child(
      ctx.tree.root(), utility::Any(ANARI_VOLUME, volume.index()));

  return volume;
}

} // namespace tsd
