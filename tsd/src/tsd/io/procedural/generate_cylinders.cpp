// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/procedural.hpp"
// std
#include <random>

namespace tsd::io {

void generate_cylinders(
    Scene &scene, LayerNodeRef location, bool useDefaultMaterial)
{
  if (!location)
    location = scene.defaultLayer()->root();

  // Generate geometry //

  auto cylinders = scene.createObject<Geometry>(tokens::geometry::cylinder);

  cylinders->setName("random_cylinders_geometry");

  const uint32_t numCylinders = 20;
  const float radius = 0.025f;

  cylinders->setParameter("radius", radius);

  std::mt19937 rng;
  rng.seed(0);
  std::normal_distribution<float> pos_dist(0.f, 1.f);
  std::uniform_real_distribution<float> col_dist(0.f, 1.f);

  auto positionArray = scene.createArray(ANARI_FLOAT32_VEC3, 2 * numCylinders);
  auto colorArray = scene.createArray(ANARI_FLOAT32_VEC4, 2 * numCylinders);

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

  cylinders->setParameterObject("vertex.position", *positionArray);
  cylinders->setParameterObject("vertex.color", *colorArray);

  // Populate material with sampler for colormapping //

  auto material = scene.defaultMaterial();
  if (!useDefaultMaterial) {
    material = scene.createObject<Material>(tokens::material::matte);
    material->setParameter("color", "color");
    material->setName("random_cylinders_material");
  }

  auto surface = scene.createSurface("random_cylinders", cylinders, material);
  scene.insertChildObjectNode(location, surface);
}

} // namespace tsd
