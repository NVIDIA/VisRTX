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

void import_E57XYZ(Scene &scene, const char *filepath, LayerNodeRef location)
{
  std::string file = fileOf(filepath);
  if (file.empty())
    return;

  // load particle data from file //

  uint64_t numParticles = 0;
  auto *fp = std::fopen(filepath, "rb");
  auto r = std::fread(&numParticles, sizeof(numParticles), 1, fp);

  logInfo(
      "[import_e57xyz] loading %zu points from %s", numParticles, file.c_str());

  std::vector<float3> positions(numParticles);
  std::vector<float3> colors(numParticles);
  r = std::fread(positions.data(), sizeof(float3), numParticles, fp);
  r = std::fread(colors.data(), sizeof(float3), numParticles, fp);
  std::fclose(fp);

  for (auto &c : colors)
    c = tsd::core::math::normalizeColor(c);

  // create TSD objects //

  auto xyz_root = scene.insertChildNode(
      location ? location : scene.defaultLayer()->root(), file.c_str());

  auto positionsArray = scene.createArray(ANARI_FLOAT32_VEC3, numParticles);
  auto colorsArray = scene.createArray(ANARI_FLOAT32_VEC3, numParticles);

  positionsArray->setData(positions);
  colorsArray->setData(colors);

  // geometry + material

  auto geom = scene.createObject<Geometry>(tokens::geometry::sphere);
  geom->setName("e57xyz_geometry");
  geom->setParameter("radius", 0.001f); // TODO: something smarter
  geom->setParameterObject("vertex.position", *positionsArray);
  geom->setParameterObject("vertex.color", *colorsArray);

  auto mat = scene.createObject<Material>(tokens::material::matte);
  mat->setName("e57xyz_material");
  mat->setParameter("color", "color");

  // surface

  auto surface = scene.createSurface("e57xyz_surface", geom, mat);
  scene.insertChildObjectNode(xyz_root, surface);
}

} // namespace tsd::io
