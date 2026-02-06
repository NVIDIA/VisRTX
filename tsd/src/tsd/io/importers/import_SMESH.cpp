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

//
// Importing a bespoke binary dump of just triangles + vertex attribute scalars
//
void import_SMESH(
    Scene &scene, const char *filepath, LayerNodeRef location, bool isAnimation)
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

  size_t size = 1;
  if (isAnimation)
    auto r = std::fread(&size, sizeof(size_t), 1, fp);

  std::vector<TimeStepArrays> arrays;

  arrays.emplace_back(); // primitive.index
  arrays.emplace_back(); // vertex.position
  arrays.emplace_back(); // vertex.attribute0

  for (size_t i = 0; i < size; i++) {
    arrays[0].push_back(readArray(scene, ANARI_UINT32_VEC3, fp));
    arrays[1].push_back(readArray(scene, ANARI_FLOAT32_VEC3, fp));
    arrays[2].push_back(readArray(scene, ANARI_FLOAT32, fp));
  }

  if (size > 1) {
    auto animationName = "SMESH animation for " + std::string(filename);
    auto *anim = scene.addAnimation(animationName.c_str());
    anim->setAsTimeSteps(*geom,
        {Token("primitive.index"),
            Token("vertex.position"),
            Token("vertex.attribute0")},
        arrays);
  }

  geom->setParameterObject("primitive.index", *arrays[0][0]);
  geom->setParameterObject("vertex.position", *arrays[1][0]);
  if (arrays[2][0])
    geom->setParameterObject("vertex.attribute0", *arrays[2][0]);

  auto firstScalarArray = arrays[2][0];
  if (firstScalarArray) {
    mat = scene.createObject<Material>(tokens::material::matte);
    auto range = computeScalarRange(*firstScalarArray);
    mat->setParameterObject("color", *makeDefaultColorMapSampler(scene, range));
  } else
    mat = scene.defaultMaterial();

  auto surface = scene.createSurface(filename.c_str(), geom, mat);

  auto surfaceLayerRef = scene.insertChildObjectNode(hs_root, surface);
  (*surfaceLayerRef)->name() = filename;

  std::fclose(fp);
}

} // namespace tsd::io
