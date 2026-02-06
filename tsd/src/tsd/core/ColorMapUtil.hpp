// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_math
#include "tsd/core/TSDMath.hpp"
// std
#include <vector>

namespace tsd::core {

using ColorPoint = float4;
using OpacityPoint = float2;

struct TransferFunction
{
  std::vector<ColorPoint> colorPoints;
  std::vector<OpacityPoint> opacityPoints;
  math::box1 range = {};
};

std::vector<math::float4> makeDefaultColorMap(size_t size = 256);

template <typename T>
std::vector<T> resampleArray(const std::vector<T> &input, size_t newSize);

namespace detail {

tsd::math::float3 interpolateColor(
    const std::vector<ColorPoint> &controlPoints, float x);

float interpolateOpacity(
    const std::vector<OpacityPoint> &controlPoints, float x);
} // namespace detail

namespace colormap {

extern std::vector<float3> jet;
extern std::vector<float3> cool_to_warm;
extern std::vector<float3> viridis;
extern std::vector<float3> black_body;
extern std::vector<float3> inferno;
extern std::vector<float3> ice_fire;
extern std::vector<float3> grayscale;

} // namespace colormap

///////////////////////////////////////////////////////////////////////////////
// Inlined definitions ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename T>
inline std::vector<T> resampleArray(const std::vector<T> &input, size_t newSize)
{
  std::vector<T> output(newSize);
  const float scale = static_cast<float>(input.size() - 1) / (newSize - 1);
  for (size_t i = 0; i < newSize; i++) {
    const float x = i * scale;
    const int idx = static_cast<int>(x);
    const float t = x - idx;
    if (idx + 1 < input.size()) {
      output[i] = (1.0f - t) * input[idx] + t * input[idx + 1];
    } else {
      output[i] = input[idx];
    }
  }
  return output;
}

} // namespace tsd::core
