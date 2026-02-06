// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers.hpp"

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/algorithms/computeScalarRange.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <algorithm>
#include <cstdio>
#include <vector>

namespace tsd::io {

using namespace tsd::core;

void import_AXYZ(Scene &scene, const char *filepath, LayerNodeRef location)
{
  std::string file = fileOf(filepath);
  if (file.empty())
    return;

  // load particle data from file //

  auto *fp = std::fopen(filepath, "rb");
  if (!fp) {
    tsd::core::logError("[import_axyz] could not open file %s", filepath);
    return;
  }

  uint64_t numTimeSteps = 0;
  uint64_t numParticles = 0;
  auto r = std::fread(&numTimeSteps, sizeof(numTimeSteps), 1, fp);
  r = std::fread(&numParticles, sizeof(numParticles), 1, fp);

  if (numTimeSteps == 0 || numParticles == 0) {
    tsd::core::logError(
        "[import_axyz] animation has no points in '%s'", filepath);
    std::fclose(fp);
    return;
  }

  tsd::core::logInfo(
      "[import_axyz] loading [%zu] time steps containing [%zu]"
      " points each from %s",
      numTimeSteps,
      numParticles,
      filepath);

  std::vector<tsd::core::ObjectUsePtr<tsd::core::Array>> timeSteps;

  for (int t = 0; t < numTimeSteps; ++t) {
    auto positionsArray = scene.createArray(ANARI_FLOAT32_VEC3, numParticles);
    positionsArray->setName(
        ("vertex.position_" + file + '_' + std::to_string(t)).c_str());
    positionsArray->setData(fp);
    timeSteps.emplace_back(positionsArray);
  }

  std::fclose(fp);

  // create TSD objects //

  auto axyz_root = scene.insertChildTransformNode(
      location ? location : scene.defaultLayer()->root());
  (*axyz_root)->name() = "axyz_transform_" + file;

  // geometry + material

  auto geom = scene.createObject<tsd::core::Geometry>(
      tsd::core::tokens::geometry::sphere);
  geom->setName(("axyz_geometry" + file).c_str());
  geom->setParameter("radius", 0.1f); // TODO: something smarter
  geom->setParameterObject("vertex.position", *timeSteps[0]);

  auto mat = scene.createObject<tsd::core::Material>(
      tsd::core::tokens::material::matte);
  mat->setName("axyz_material");
  mat->setParameter("color", tsd::math::float3(0.8f, 0.8f, 0.8f));

  // surface

  auto surface = scene.createSurface("axyz_surface", geom, mat);
  scene.insertChildObjectNode(axyz_root, surface);

  // animation

  auto *anim = scene.addAnimation(file.c_str());
  anim->setAsTimeSteps(*geom, "vertex.position", timeSteps);
}

} // namespace tsd::io
