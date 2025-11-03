// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/io/procedural.hpp"
// std
#include <algorithm>
#include <array>
#include <cmath>

namespace tsd::io {

void generate_sphereSetVolume(Scene &scene, LayerNodeRef location)
{
  if (!location)
    location = scene.defaultLayer()->root();

  location = scene.insertChildTransformNode(location);

  auto root = scene.insertChildNode(location);
  (*root)->name() = "Sphere Set Volume";

  constexpr int volumeSize = 512;
  constexpr float voxelSpacing = 1.0f / volumeSize;
  constexpr float3 volumeOrigin{-0.5f, -0.5f, -0.5f};
  constexpr int maxColorValue = 255;
  constexpr int colorMapSize = 256;

  auto voxelArray =
      scene.createArray(ANARI_UFIXED8, volumeSize, volumeSize, volumeSize);
  auto *voxels = voxelArray->mapAs<uint8_t>();

  // RBF centers and radii, normalized volume space [0,1]
  struct RBFCenter
  {
    float3 center;
    float radius; // support radius for RBF
    float weight; // contribution weight
  };

  constexpr std::array<RBFCenter, 5> rbfCenters{{
      {{0.3f, 0.3f, 0.3f}, 0.15f, 1.0f},
      {{0.7f, 0.7f, 0.3f}, 0.12f, 1.0f},
      {{0.5f, 0.5f, 0.7f}, 0.18f, 1.0f},
      {{0.2f, 0.7f, 0.6f}, 0.10f, 1.0f},
      {{0.8f, 0.3f, 0.7f}, 0.14f, 1.0f},
  }};

  // Gaussian RBF kernel: exp(-r²/h²)
  constexpr auto gaussianRBF = [](float r, float h) noexcept -> float {
    if (r >= h)
      return 0.0f;
    const float normalized = r / h;
    return std::exp(-normalized * normalized);
  };

  // Fill volume using RBF interpolation
  for (int z = 0; z < volumeSize; ++z) {
    for (int y = 0; y < volumeSize; ++y) {
      for (int x = 0; x < volumeSize; ++x) {
        const float3 pos{x / float(volumeSize),
            y / float(volumeSize),
            z / float(volumeSize)};

        // Sum contributions from all RBF centers
        float value = 0.0f;
        for (const auto &rbf : rbfCenters) {
          const float dist = linalg::length(pos - rbf.center);
          value += rbf.weight * gaussianRBF(dist, rbf.radius);
        }

        // Clamp to [0, 1] range
        value = std::clamp(value, 0.0f, 1.0f);

        const size_t idx = z * volumeSize * volumeSize + y * volumeSize + x;
        voxels[idx] = static_cast<uint8_t>(value * maxColorValue);
      }
    }
  }

  voxelArray->unmap();

  // Create spatial field //
  auto field = scene.createObject<SpatialField>(
      tokens::spatial_field::structuredRegular);
  field->setName("sphere_set_volume_field");
  field->setParameter("origin", volumeOrigin);
  field->setParameter(
      "spacing", float3{voxelSpacing, voxelSpacing, voxelSpacing});
  field->setParameterObject("data", *voxelArray);

  // Create TransferFunction1D volume //
  auto colorArray = scene.createArray(ANARI_FLOAT32_VEC4, colorMapSize);
  colorArray->setData(makeDefaultColorMap(colorMapSize).data());

  auto [volumeNode, volume] = scene.insertNewChildObjectNode<Volume>(
      root, tokens::volume::transferFunction1D);
  volume->setName("slice_test_volume");
  volume->setParameterObject("color", *colorArray);
  volume->setParameterObject("value", *field);
}

} // namespace tsd::io
