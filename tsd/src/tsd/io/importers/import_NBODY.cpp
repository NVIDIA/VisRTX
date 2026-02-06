// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
// std
#include <array>
#include <fstream>
#include <vector>

namespace tsd::io {

using namespace tsd::core;

struct NBODYScene
{
  std::vector<float3> points; // (x, y, z) * numParticles
  float radius{1.5f};
};

void importNBODYFile(const char *filename, NBODYScene &s)
{
  uint64_t numParticles = 0;

  std::ifstream fin(filename);
  if (!fin.is_open())
    return;
  fin >> numParticles;
  s.points.resize(numParticles);
  logStatus("[import_NBODY] np: %zu", numParticles);
  for (uint64_t i = 0; i < numParticles; i++) {
    float3 p;
    fin >> p.x;
    fin >> p.y;
    fin >> p.z;
    logStatus("[import_NBODY] p: %f %f %f", p.x, p.y, p.z);
    s.points.push_back(p);
  }
  fin.close();
}

void import_NBODY(Scene &scene,
    const char *filepath,
    LayerNodeRef location,
    bool useDefaultMaterial)
{
  NBODYScene nbody;
  importNBODYFile(filepath, nbody);

  if (nbody.points.empty())
    return;

  auto nbody_root =
      scene.insertChildNode(location ? location : scene.defaultLayer()->root(),
          fileOf(filepath).c_str());

  auto mat = useDefaultMaterial
      ? scene.getObject<Material>(0)
      : scene.createObject<Material>(tokens::material::matte);

  int64_t numRemainingPoints = nbody.points.size();
  constexpr int64_t CHUNK = 1e7;

  for (int i = 0; numRemainingPoints > 0; numRemainingPoints -= CHUNK, i++) {
    const size_t numPoints =
        std::min(size_t(numRemainingPoints), size_t(CHUNK));
    auto vertexPositionArray = scene.createArray(ANARI_FLOAT32_VEC3, numPoints);
    vertexPositionArray->setData(nbody.points.data() + (i * CHUNK));

    std::string geomName = "NBODY_geometry_" + std::to_string(i);

    auto geom = scene.createObject<Geometry>(tokens::geometry::sphere);
    geom->setName(geomName.c_str());
    geom->setParameter("radius", nbody.radius);
    geom->setParameterObject("vertex.position", *vertexPositionArray);

    auto surface = scene.createSurface(geomName.c_str(), geom, mat);
    scene.insertChildObjectNode(nbody_root, surface);
  }
}

} // namespace tsd
