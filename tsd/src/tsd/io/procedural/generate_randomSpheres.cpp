// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/procedural.hpp"
// std
#include <random>

namespace tsd::io {

void generate_randomSpheres(
    Context &ctx, LayerNodeRef location, bool useDefaultMaterial)
{
  if (!location)
    location = ctx.defaultLayer()->root();

  // Generate geometry //

  auto spheres = ctx.createObject<Geometry>(tokens::geometry::sphere);

  spheres->setName("random_spheres_geometry");

  const uint32_t numSpheres = 10000;
  const float radius = .01f;

  spheres->setParameter("radius"_t, radius);

  std::mt19937 rng;
  rng.seed(0);
  std::normal_distribution<float> vert_dist(0.f, 0.25f);

  auto positionArray = ctx.createArray(ANARI_FLOAT32_VEC3, numSpheres);
  auto distanceArray = ctx.createArray(ANARI_FLOAT32, numSpheres);

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

  spheres->setParameterObject("vertex.position"_t, *positionArray);
  spheres->setParameterObject("vertex.attribute0"_t, *distanceArray);

  // Populate material with sampler for colormapping //

  auto material = ctx.defaultMaterial();
  if (!useDefaultMaterial) {
    auto sampler = ctx.createObject<Sampler>(tokens::sampler::image1D);

    auto colorArray = ctx.createArray(ANARI_FLOAT32_VEC3, 2);
    auto *colors = (float3 *)colorArray->map();
    colors[0] = float3(.8f, .1f, .1f);
    colors[1] = float3(.8f, .8f, .1f);
    colorArray->unmap();

    sampler->setParameterObject("image"_t, *colorArray);
    auto scale = math::scaling_matrix(float3(1.f / maxDist));
    sampler->setParameter("inTransform"_t, scale);
    sampler->setName("random_spheres_colormap");

    material = ctx.createObject<Material>(tokens::material::matte);
    material->setParameterObject("color"_t, *sampler);
    material->setName("random_spheres_material");
  }

  auto surface = ctx.createSurface("random_spheres", spheres, material);
  ctx.insertChildObjectNode(location, surface);
}

} // namespace tsd
