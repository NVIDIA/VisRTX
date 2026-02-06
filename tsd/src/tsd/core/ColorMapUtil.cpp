// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/ColorMapUtil.hpp"

namespace tsd::core {

namespace detail {

tsd::math::float3 interpolateColor(
    const std::vector<ColorPoint> &controlPoints, float x)
{
  auto first = controlPoints.front();
  if (x <= first.x)
    return tsd::math::float3(first.y, first.z, first.w);

  for (uint32_t i = 1; i < controlPoints.size(); i++) {
    auto current = controlPoints[i];
    auto previous = controlPoints[i - 1];
    if (x <= current.x) {
      const float t = (x - previous.x) / (current.x - previous.x);
      return (1.0f - t) * tsd::math::float3(previous.y, previous.z, previous.w)
          + t * tsd::math::float3(current.y, current.z, current.w);
    }
  }

  auto last = controlPoints.back();
  return tsd::math::float3(last.x, last.y, last.z);
}

float interpolateOpacity(
    const std::vector<OpacityPoint> &controlPoints, float x)

{
  auto first = controlPoints.front();
  if (x <= first.x)
    return first.y;

  for (uint32_t i = 1; i < controlPoints.size(); i++) {
    auto current = controlPoints[i];
    auto previous = controlPoints[i - 1];
    if (x <= current.x) {
      const float t = (x - previous.x) / (current.x - previous.x);
      return (1.0 - t) * previous.y + t * current.y;
    }
  }

  auto last = controlPoints.back();
  return last.y;
}

} // namespace detail

std::vector<math::float4> makeDefaultColorMap(size_t size)
{
  std::vector<math::float4> colors;
  colors.emplace_back(1.f, 0.f, 0.f, 0.0f);
  colors.emplace_back(0.f, 1.f, 0.f, 0.5f);
  colors.emplace_back(0.f, 0.f, 1.f, 1.0f);
  return resampleArray(colors, size);
}

namespace colormap {

// clang-format off

std::vector<float3> jet = {
  {0.f, 0.f, 1.f},
  {0.f, 1.f, 1.f},
  {1.f, 1.f, 0.f},
  {1.f, 0.f, 0.f}
};

std::vector<float3> cool_to_warm = {
  {0.231f, 0.298f, 0.752f},
  {0.552f, 0.690f, 0.996f},
  {0.866f, 0.866f, 0.866f},
  {0.956f, 0.603f, 0.486f},
  {0.705f, 0.015f, 0.149f}
};

std::vector<float3> viridis = {
  {0.267004, 0.004874, 0.329415},
  {0.282656, 0.100196, 0.42216},
  {0.277134, 0.185228, 0.489898},
  {0.253935, 0.265254, 0.529983},
  {0.221989, 0.339161, 0.548752},
  {0.190631, 0.407061, 0.556089},
  {0.163625, 0.471133, 0.558148},
  {0.139147, 0.533812, 0.555298},
  {0.120565, 0.596422, 0.543611},
  {0.134692, 0.658636, 0.517649},
  {0.20803, 0.718701, 0.472873},
  {0.327796, 0.77398, 0.40664},
  {0.477504, 0.821444, 0.318195},
  {0.647257, 0.8584, 0.209861},
  {0.82494, 0.88472, 0.106217},
  {0.993248, 0.906157, 0.143936}
};

std::vector<float3> black_body = {
  {0.f, 0.f, 0.f},
  {1.f, 0.f, 0.f},
  {1.f, 1.f, 0.f},
  {1.f, 1.f, 1.f}
};

std::vector<float3> inferno = {
  {0.f, 0.f, 0.f},
  {0.25f, 0.f, 0.25f},
  {1.f, 0.f, 0.f},
  {1.f, 1.f, 0.f},
  {1.f, 1.f, 1.f}
};

std::vector<float3> ice_fire = {
  {0, 0, 0},
  {0, 0.120394, 0.302678},
  {0, 0.216587, 0.524575},
  {0.0552529, 0.345022, 0.659495},
  {0.128054, 0.492592, 0.720287},
  {0.188952, 0.641306, 0.792096},
  {0.327672, 0.784939, 0.873426},
  {0.60824, 0.892164, 0.935546},
  {0.881376, 0.912184, 0.818097},
  {0.9514, 0.835615, 0.449271},
  {0.904479, 0.690486, 0},
  {0.854063, 0.510857, 0},
  {0.777096, 0.330175, 0.000885023},
  {0.672862, 0.139086, 0.00270085},
  {0.508812, 0, 0},
  {0.299413, 0.000366217, 0.000549325},
  {0.0157473, 0.00332647, 0}
};

std::vector<float3> grayscale = {
  {1.f, 1.f, 1.f},
  {1.f, 1.f, 1.f}
};

// clang-format on

} // namespace colormap

} // namespace tsd::core
