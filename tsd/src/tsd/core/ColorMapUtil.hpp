// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TSDMath.hpp"
// std
#include <vector>

namespace tsd::core {

using ColorPoint = float4;
using OpacityPoint = float2;

std::vector<math::float4> makeDefaultColorMap(size_t size = 256);

template <typename T>
std::vector<T> resampleArray(const std::vector<T> &input, size_t newSize);

///////////////////////////////////////////////////////////////////////////////
// Inlined definitions ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace detail {

inline tsd::math::float3 interpolateColor(
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

inline float interpolateOpacity(
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

inline std::vector<math::float4> makeDefaultColorMap(size_t size)
{
  std::vector<math::float4> colors;
  colors.emplace_back(1.f, 0.f, 0.f, 0.0f);
  colors.emplace_back(0.f, 1.f, 0.f, 0.5f);
  colors.emplace_back(0.f, 0.f, 1.f, 1.0f);
  return resampleArray(colors, size);
}

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
