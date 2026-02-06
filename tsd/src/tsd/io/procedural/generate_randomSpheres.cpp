// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/procedural.hpp"
// std
#include <random>

namespace tsd::io {

void generate_randomSpheres(
    Scene &scene, LayerNodeRef location, bool useDefaultMaterial)
{
  if (!location)
    location = scene.defaultLayer()->root();

  // Generate geometry //

  auto spheres = scene.createObject<Geometry>(tokens::geometry::sphere);

  spheres->setName("random_spheres_geometry");

  const uint32_t numSpheres = 10000;
  const float radius = .01f;

  spheres->setParameter("radius", radius);

  std::mt19937 rng;
  rng.seed(0);
  std::normal_distribution<float> vert_dist(0.f, 0.25f);

  auto positionArray = scene.createArray(ANARI_FLOAT32_VEC3, numSpheres);
  auto distanceArray = scene.createArray(ANARI_FLOAT32, numSpheres);

  auto *positions = positionArray->mapAs<float3>();
  auto *distances = distanceArray->mapAs<float>();

  float maxDist = 0.f;
  for (uint32_t i = 0; i < numSpheres; i++) {
    const auto p = float3(vert_dist(rng), vert_dist(rng), vert_dist(rng));
    positions[i] = p;
    const float d = linalg::length(p);
    distances[i] = d;
    maxDist = std::max(maxDist, d);
  }

  positionArray->unmap();
  distanceArray->unmap();

  spheres->setParameterObject("vertex.position", *positionArray);
  spheres->setParameterObject("vertex.attribute0", *distanceArray);

  // Populate material with sampler for colormapping //

  auto material = scene.defaultMaterial();
  if (!useDefaultMaterial) {
    auto sampler = scene.createObject<Sampler>(tokens::sampler::image1D);

    auto colorArray = scene.createArray(ANARI_FLOAT32_VEC3, 2);
    auto *colors = (float3 *)colorArray->map();
    colors[0] = float3(.8f, .1f, .1f);
    colors[1] = float3(.8f, .8f, .1f);
    colorArray->unmap();

    sampler->setParameterObject("image", *colorArray);
    auto scale = math::scaling_matrix(float3(1.f / maxDist));
    sampler->setParameter("inTransform", scale);
    sampler->setName("random_spheres_colormap");

    material = scene.createObject<Material>(tokens::material::matte);
    material->setParameterObject("color", *sampler);
    material->setName("random_spheres_material");
  }

  auto surface = scene.createSurface("random_spheres", spheres, material);
  scene.insertChildObjectNode(location, surface);
}

} // namespace tsd
