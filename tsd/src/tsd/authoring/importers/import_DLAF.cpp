// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
#include "tsd/core/ColorMapUtil.hpp"
// std
#include <array>
#include <vector>

namespace tsd {

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

void import_DLAF(Context &ctx,
    const char *filepath,
    InstanceNode::Ref location,
    bool useDefaultMaterial)
{
  DLAFScene scene;
  importDLAFFile(filepath, scene);

  if (scene.points.empty())
    return;

  auto dlaf_root = ctx.insertChildNode(
      location ? location : ctx.tree.root(), fileOf(filepath).c_str());

  auto mat = useDefaultMaterial
      ? ctx.defaultMaterial()
      : ctx.createObject<Material>(tokens::material::matte);

  if (!useDefaultMaterial) {
    auto samplerImageArray = ctx.createArray(ANARI_FLOAT32_VEC4, 3);
    auto *colorMapPtr = samplerImageArray->mapAs<math::float4>();
    colorMapPtr[0] = math::float4(0.f, 0.f, 1.f, 1.f);
    colorMapPtr[1] = math::float4(0.f, 1.f, 0.f, 1.f);
    colorMapPtr[2] = math::float4(1.f, 0.f, 0.f, 1.f);
    samplerImageArray->unmap();

    auto sampler = ctx.createObject<Sampler>(tokens::sampler::image1D);
    sampler->setParameter("inAttribute"_t, "attribute0");
    sampler->setParameter(
        "inTransform"_t, makeColorMapTransform(0.f, scene.maxDistance));
    sampler->setParameter("filter"_t, "linear");
    sampler->setParameter("wrapMode"_t, "mirrorRepeat");
    sampler->setParameterObject("image"_t, *samplerImageArray);

    mat->setParameterObject("color"_t, *sampler);
  }

  int64_t numRemainingPoints = scene.distances.size();
  constexpr int64_t CHUNK = 1e7;

  for (int i = 0; numRemainingPoints > 0; numRemainingPoints -= CHUNK, i++) {
    const size_t numPoints =
        std::min(size_t(numRemainingPoints), size_t(CHUNK));
    auto vertexPositionArray = ctx.createArray(ANARI_FLOAT32_VEC3, numPoints);
    vertexPositionArray->setData(scene.points.data() + (3 * i * CHUNK));

    auto attributeArray = ctx.createArray(ANARI_FLOAT32, numPoints);
    attributeArray->setData(scene.distances.data() + (i * CHUNK));

    std::string geomName = "DLAF_geometry_" + std::to_string(i);

    auto geom = ctx.createObject<Geometry>(tokens::geometry::sphere);
    geom->setName(geomName.c_str());
    geom->setParameter("radius"_t, scene.radius);
    geom->setParameterObject("vertex.position"_t, *vertexPositionArray);
    geom->setParameterObject("primitive.attribute0"_t, *attributeArray);

    auto surface = ctx.createSurface(geomName.c_str(), geom, mat);

    ctx.insertChildObjectNode(dlaf_root, surface);
  }
}

} // namespace tsd
