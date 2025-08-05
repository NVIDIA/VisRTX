// Copyright 2024-2025 NVIDIA Corporation
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

void import_NBODY(Context &ctx,
    const char *filepath,
    LayerNodeRef location,
    bool useDefaultMaterial)
{
  NBODYScene scene;
  importNBODYFile(filepath, scene);

  if (scene.points.empty())
    return;

  auto nbody_root =
      ctx.insertChildNode(location ? location : ctx.defaultLayer()->root(),
          fileOf(filepath).c_str());

  auto mat = useDefaultMaterial
      ? ctx.getObject<Material>(0)
      : ctx.createObject<Material>(tokens::material::matte);

  int64_t numRemainingPoints = scene.points.size();
  constexpr int64_t CHUNK = 1e7;

  for (int i = 0; numRemainingPoints > 0; numRemainingPoints -= CHUNK, i++) {
    const size_t numPoints =
        std::min(size_t(numRemainingPoints), size_t(CHUNK));
    auto vertexPositionArray = ctx.createArray(ANARI_FLOAT32_VEC3, numPoints);
    vertexPositionArray->setData(scene.points.data() + (i * CHUNK));

    std::string geomName = "NBODY_geometry_" + std::to_string(i);

    auto geom = ctx.createObject<Geometry>(tokens::geometry::sphere);
    geom->setName(geomName.c_str());
    geom->setParameter("radius"_t, scene.radius);
    geom->setParameterObject("vertex.position"_t, *vertexPositionArray);

    auto surface = ctx.createSurface(geomName.c_str(), geom, mat);
    ctx.insertChildObjectNode(nbody_root, surface);
  }
}

} // namespace tsd
