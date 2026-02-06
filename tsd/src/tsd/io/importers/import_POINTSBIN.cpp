// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/algorithms/computeScalarRange.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <array>
#include <vector>

namespace tsd::io {

using namespace tsd::core;

//
// Importing a bespoke binary dump of points + N vertex attribute scalars
//
void import_POINTSBIN(Scene &scene,
    const std::vector<std::string> &filepaths,
    LayerNodeRef location)
{
  if (filepaths.empty())
    return;

  auto hs_root = scene.insertChildTransformNode(
      location ? location : scene.defaultLayer()->root());

  MaterialRef mat;
  auto geom = scene.createObject<Geometry>(tokens::geometry::sphere);
  geom->setName("pointsbin_geometry");

  size_t numTimeSteps = filepaths.size();

  std::vector<TimeStepArrays> arrays;
  arrays.emplace_back(); // vertex.position
  arrays.emplace_back(); // vertex.attribute0

  for (auto &filepath : filepaths) {
    auto *fp = std::fopen(filepath.c_str(), "rb");
    if (!fp)
      continue;

    auto filename = fileOf(filepath);

    arrays[0].push_back(readArray(scene, ANARI_FLOAT32_VEC3, fp));

    // NOTE: only read first attribute array for now
    size_t numAttributes = 0;
    auto r = std::fread(&numAttributes, sizeof(size_t), 1, fp);
    arrays[1].push_back(readArray(scene, ANARI_FLOAT32, fp));

    std::fclose(fp);
  }

  if (numTimeSteps > 1) {
    auto *anim = scene.addAnimation("pointsbin animation");
    anim->setAsTimeSteps(
        *geom, {Token("vertex.position"), Token("vertex.attribute0")}, arrays);
  }

  geom->setParameterObject("vertex.position", *arrays[0][0]);
  if (arrays[1][0])
    geom->setParameterObject("vertex.attribute0", *arrays[1][0]);
  geom->setParameter("radius", 0.01f); // TODO: make configurable

  auto firstScalarArray = arrays[1][0];
  if (firstScalarArray) {
    mat = scene.createObject<Material>(tokens::material::matte);
    auto range = computeScalarRange(*firstScalarArray);
    mat->setParameterObject("color", *makeDefaultColorMapSampler(scene, range));
  } else
    mat = scene.defaultMaterial();

  auto surface = scene.createSurface("pointsbin_surface", geom, mat);
  auto surfaceLayerRef = scene.insertChildObjectNode(hs_root, surface);
}

} // namespace tsd::io
