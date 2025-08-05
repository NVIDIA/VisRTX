// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/procedural.hpp"
// std
#include <random>

namespace tsd::io {

void generate_cylinders(
    Context &ctx, LayerNodeRef location, bool useDefaultMaterial)
{
  if (!location)
    location = ctx.defaultLayer()->root();

  // Generate geometry //

  auto cylinders = ctx.createObject<Geometry>(tokens::geometry::cylinder);

  cylinders->setName("random_cylinders_geometry");

  const uint32_t numCylinders = 20;
  const float radius = 0.025f;

  cylinders->setParameter("radius"_t, radius);

  std::mt19937 rng;
  rng.seed(0);
  std::normal_distribution<float> pos_dist(0.f, 1.f);
  std::uniform_real_distribution<float> col_dist(0.f, 1.f);

  auto positionArray = ctx.createArray(ANARI_FLOAT32_VEC3, 2 * numCylinders);
  auto colorArray = ctx.createArray(ANARI_FLOAT32_VEC4, 2 * numCylinders);

  std::vector<float3> positions(2 * numCylinders);
  std::vector<float4> colors(2 * numCylinders);

  for (auto &s : positions) {
    s.x = pos_dist(rng);
    s.y = pos_dist(rng);
    s.z = pos_dist(rng);
  }

  for (auto &s : colors) {
    s.x = col_dist(rng);
    s.y = col_dist(rng);
    s.z = col_dist(rng);
    s.w = 1.f;
  }

  positionArray->setData(positions);
  colorArray->setData(colors);

  cylinders->setParameterObject("vertex.position"_t, *positionArray);
  cylinders->setParameterObject("vertex.color"_t, *colorArray);

  // Populate material with sampler for colormapping //

  auto material = ctx.defaultMaterial();
  if (!useDefaultMaterial) {
    material = ctx.createObject<Material>(tokens::material::matte);
    material->setParameter("color"_t, "color");
    material->setName("random_cylinders_material");
  }

  auto surface = ctx.createSurface("random_cylinders", cylinders, material);
  ctx.insertChildObjectNode(location, surface);
}

} // namespace tsd
