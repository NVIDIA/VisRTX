// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/procedural.hpp"
// std
#include <algorithm>
#include <random>

namespace tsd {

VolumeRef generate_noiseVolume(
    Context &ctx, ArrayRef colorArray, ArrayRef opacityArray)
{
  // Generate spatial field //

  static std::mt19937 rng;
  rng.seed(0);
  static std::normal_distribution<float> dist(0.f, 1.0f);

  auto field =
      ctx.createObject<SpatialField>(tokens::spatial_field::structuredRegular);
  field->setName("noise_field");

  auto voxelArray = ctx.createArray(ANARI_UFIXED8, 64, 64, 64);

  auto *voxelsBegin = (uint8_t *)voxelArray->map();
  auto *voxelsEnd = voxelsBegin + (64 * 64 * 64);

  std::for_each(voxelsBegin, voxelsEnd, [&](auto &v) { v = dist(rng) * 255; });

  voxelArray->unmap();

  field->setParameter("origin"_t, float3(-1, -1, -1));
  field->setParameter("spacing"_t, float3(2.f / 64));
  field->setParameterObject("data"_t, *voxelArray);

  // Setup volume //

  auto [inst, volume] = ctx.insertNewChildObjectNode<Volume>(
      ctx.tree.root(), tokens::volume::transferFunction1D);
  volume->setName("noise_volume");

  if (!(colorArray && opacityArray)) {
    colorArray = ctx.createArray(ANARI_FLOAT32_VEC3, 3);
    opacityArray = ctx.createArray(ANARI_FLOAT32, 2);

    std::vector<float3> colors;
    std::vector<float> opacities;

    colors.emplace_back(0.f, 0.f, 1.f);
    colors.emplace_back(0.f, 1.f, 0.f);
    colors.emplace_back(1.f, 0.f, 0.f);

    opacities.emplace_back(0.f);
    opacities.emplace_back(1.f);

    colorArray->setData(colors.data());
    opacityArray->setData(opacities.data());
  }

  volume->setParameterObject("value"_t, *field);
  volume->setParameterObject("color"_t, *colorArray);
  volume->setParameterObject("opacity"_t, *opacityArray);
  volume->setParameter("densityScale"_t, 0.1f);

  return volume;
}

} // namespace tsd
