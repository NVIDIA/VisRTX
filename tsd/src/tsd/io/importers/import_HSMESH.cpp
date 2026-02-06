// Copyright 2024-2026 NVIDIA Corporation
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

static ArrayRef readHsArray(Scene &scene,
    GeometryRef geom,
    const char *param,
    anari::DataType elementType,
    std::FILE *fp)
{
  ArrayRef retval = readArray(scene, elementType, fp);
  if (retval)
    geom->setParameterObject(param, *retval);
  return retval;
}

//
// Importing Haystack meshes: https://github.com/ingowald/haystack
//
void import_HSMESH(Scene &scene, const char *filepath, LayerNodeRef location)
{
  auto *fp = std::fopen(filepath, "rb");
  if (!fp)
    return;

  auto filename = fileOf(filepath);

  auto hs_root = scene.insertChildTransformNode(
      location ? location : scene.defaultLayer()->root());

  MaterialRef mat;
  auto geom = scene.createObject<Geometry>(tokens::geometry::triangle);
  geom->setName(filename.c_str());

  readHsArray(scene, geom, "vertex.position", ANARI_FLOAT32_VEC3, fp);
  readHsArray(scene, geom, "vertex.normal", ANARI_FLOAT32_VEC3, fp);
  readHsArray(scene, geom, "vertex.color", ANARI_FLOAT32_VEC3, fp);
  readHsArray(scene, geom, "primitive.index", ANARI_UINT32_VEC3, fp);

  auto scalars =
      readHsArray(scene, geom, "vertex.attribute0", ANARI_FLOAT32, fp);
  if (scalars) {
    mat = scene.createObject<Material>(tokens::material::matte);
    auto range = computeScalarRange(*scalars);
    mat->setParameterObject("color", *makeDefaultColorMapSampler(scene, range));
  } else
    mat = scene.defaultMaterial();

  auto surface = scene.createSurface(filename.c_str(), geom, mat);

  auto surfaceLayerRef = scene.insertChildObjectNode(hs_root, surface);
  (*surfaceLayerRef)->name() = filename;

  std::fclose(fp);
}

} // namespace tsd::io
