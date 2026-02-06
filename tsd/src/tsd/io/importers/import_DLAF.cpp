// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <array>
#include <vector>

namespace tsd::io {

using namespace tsd::core;

struct DLAFScene
{
  std::vector<float> points; // (x, y, z) * numParticles
  std::vector<float> distances;
  float maxDistance{0.f};
  float radius{1.5f};
  std::array<float, 6> bounds = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
};

void importDLAFFile(const char *filename, DLAFScene &s)
{
  uint64_t numParticles = 0;

  auto *fp = std::fopen(filename, "rb");

  auto r = std::fread(&numParticles, sizeof(numParticles), 1, fp);
  r = std::fread(&s.radius, sizeof(s.radius), 1, fp);
  r = std::fread(&s.maxDistance, sizeof(s.maxDistance), 1, fp);
  r = std::fread(s.bounds.data(), sizeof(s.bounds[0]), s.bounds.size(), fp);

  s.points.resize(numParticles * 3);
  r = std::fread(s.points.data(), sizeof(s.points[0]), numParticles * 3, fp);

  s.distances.resize(numParticles);
  r = std::fread(s.distances.data(), sizeof(s.distances[0]), numParticles, fp);

  std::fclose(fp);
}

void import_DLAF(Scene &scene,
    const char *filepath,
    LayerNodeRef location,
    bool useDefaultMaterial)
{
  DLAFScene dlaf;
  importDLAFFile(filepath, dlaf);

  if (dlaf.points.empty())
    return;

  auto dlaf_root =
      scene.insertChildNode(location ? location : scene.defaultLayer()->root(),
          fileOf(filepath).c_str());

  auto mat = useDefaultMaterial
      ? scene.defaultMaterial()
      : scene.createObject<Material>(tokens::material::matte);

  if (!useDefaultMaterial) {
    mat->setParameterObject(
        "color", *makeDefaultColorMapSampler(scene, {0.f, dlaf.maxDistance}));
  }

  int64_t numRemainingPoints = dlaf.distances.size();
  constexpr int64_t CHUNK = 1e7;

  for (int i = 0; numRemainingPoints > 0; numRemainingPoints -= CHUNK, i++) {
    const size_t numPoints =
        std::min(size_t(numRemainingPoints), size_t(CHUNK));
    auto vertexPositionArray = scene.createArray(ANARI_FLOAT32_VEC3, numPoints);
    vertexPositionArray->setData(dlaf.points.data() + (3 * i * CHUNK));

    auto attributeArray = scene.createArray(ANARI_FLOAT32, numPoints);
    attributeArray->setData(dlaf.distances.data() + (i * CHUNK));

    std::string geomName = "DLAF_geometry_" + std::to_string(i);

    auto geom = scene.createObject<Geometry>(tokens::geometry::sphere);
    geom->setName(geomName.c_str());
    geom->setParameter("radius", dlaf.radius);
    geom->setParameterObject("vertex.position", *vertexPositionArray);
    geom->setParameterObject("primitive.attribute0", *attributeArray);

    auto surface = scene.createSurface(geomName.c_str(), geom, mat);

    scene.insertChildObjectNode(dlaf_root, surface);
  }
}

} // namespace tsd::io
