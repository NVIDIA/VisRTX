// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/algorithms/computeScalarRange.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <array>
#include <cstdio>
#include <vector>

namespace tsd::io {

using namespace tsd::core;

static ArrayRef readHsArray(Context &ctx,
    GeometryRef geom,
    const char *param,
    anari::DataType elementType,
    std::FILE *fp)
{
  ArrayRef retval;

  size_t size = 0;
  auto r = std::fread(&size, sizeof(size_t), 1, fp);

  if (size > 0) {
    retval = ctx.createArray(elementType, size);
    auto *dst = retval->map();
    r = std::fread(dst, anari::sizeOf(elementType), size, fp);
    retval->unmap();
    geom->setParameterObject(param, *retval);
  }

  return retval;
}

//
// Importing Haystack meshes: https://github.com/ingowald/haystack
//
void import_HSMESH(Context &ctx, const char *filepath, LayerNodeRef location)
{
  auto *fp = std::fopen(filepath, "rb");
  if (!fp)
    return;

  auto filename = fileOf(filepath);

  auto hs_root = ctx.insertChildTransformNode(
      location ? location : ctx.defaultLayer()->root());

  MaterialRef mat;
  auto geom = ctx.createObject<Geometry>(tokens::geometry::triangle);
  geom->setName(filename.c_str());

  readHsArray(ctx, geom, "vertex.position", ANARI_FLOAT32_VEC3, fp);
  readHsArray(ctx, geom, "vertex.normal", ANARI_FLOAT32_VEC3, fp);
  readHsArray(ctx, geom, "vertex.color", ANARI_FLOAT32_VEC3, fp);
  readHsArray(ctx, geom, "primitive.index", ANARI_UINT32_VEC3, fp);

  auto scalars = readHsArray(ctx, geom, "vertex.attribute0", ANARI_FLOAT32, fp);
  if (scalars) {
    mat = ctx.createObject<Material>(tokens::material::matte);
    auto range = computeScalarRange(*scalars);
    mat->setParameterObject("color", *makeDefaultColorMapSampler(ctx, range));
  } else
    mat = ctx.defaultMaterial();

  auto surface = ctx.createSurface(filename.c_str(), geom, mat);

  auto surfaceLayerRef = ctx.insertChildObjectNode(hs_root, surface);
  (*surfaceLayerRef)->name = filename;

  std::fclose(fp);
}

} // namespace tsd::io
