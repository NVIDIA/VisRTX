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

void import_XYZDP(Scene &scene, const char *filepath, LayerNodeRef location)
{
  std::string file = fileOf(filepath);
  if (file.empty())
    return;

  // load particle data from file //

  uint64_t numParticles = 0;
  auto *fp = std::fopen(filepath, "rb");
  auto r = std::fread(&numParticles, sizeof(numParticles), 1, fp);

  logInfo(
      "[import_XYZ] loading %zu points from %s", numParticles, file.c_str());

  std::vector<double> positions(numParticles * 3);
  std::vector<double> d(numParticles);
  std::vector<double> phi(numParticles);
  r = std::fread(positions.data(), sizeof(double), numParticles * 3, fp);
  r = std::fread(d.data(), sizeof(double), numParticles, fp);
  r = std::fread(phi.data(), sizeof(double), numParticles, fp);
  std::fclose(fp);

  // create TSD objects //

  auto xyz_root = scene.insertChildNode(
      location ? location : scene.defaultLayer()->root(), file.c_str());

  auto positionsArray = scene.createArray(ANARI_FLOAT32_VEC3, numParticles);
  auto dArray = scene.createArray(ANARI_FLOAT32, numParticles);
  auto phiArray = scene.createArray(ANARI_FLOAT32, numParticles);

  std::copy(positions.begin(), positions.end(), positionsArray->mapAs<float>());
  std::copy(d.begin(), d.end(), dArray->mapAs<float>());
  std::copy(phi.begin(), phi.end(), phiArray->mapAs<float>());

  positionsArray->unmap();
  dArray->unmap();
  phiArray->unmap();

  // geometry

  std::string geomName = "XYZ_geometry";
  auto geom = scene.createObject<Geometry>(tokens::geometry::sphere);
  geom->setName(geomName.c_str());
  geom->setParameter("radius", 0.001f);
  geom->setParameterObject("vertex.position", *positionsArray);
  geom->setParameterObject("vertex.attribute0", *phiArray);
  geom->setParameterObject("vertex.attribute1", *dArray);

  // sampler + material

  auto mat = scene.createObject<Material>(tokens::material::matte);

  auto samplerImageArray = scene.createArray(ANARI_FLOAT32_VEC4, 3);
  auto *colorMapPtr = samplerImageArray->mapAs<math::float4>();
  colorMapPtr[0] = math::float4(0.f, 0.f, 1.f, 1.f);
  colorMapPtr[1] = math::float4(0.f, 1.f, 0.f, 1.f);
  colorMapPtr[2] = math::float4(1.f, 0.f, 0.f, 1.f);
  samplerImageArray->unmap();

  auto phiRange = computeScalarRange(*phiArray);
  logInfo("[import_XYZ] ...range(phi): %f, %f", phiRange.x, phiRange.y);

  auto dRange = computeScalarRange(*dArray);
  logInfo("[import_XYZ] ...range(d)  : %f, %f", dRange.x, dRange.y);

  mat->setParameterObject("color", *makeDefaultColorMapSampler(scene, phiRange));

  // surface

  auto surface = scene.createSurface(geomName.c_str(), geom, mat);
  scene.insertChildObjectNode(xyz_root, surface);
}

} // namespace tsd::io
