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

/*
 * Transfer function definition composed of RGBA color control points and
 * scalar opacity control points, with an associated data value range.
 *
 * Example:
 *   TransferFunction tf;
 *   tf.range = {0.f, 1.f};
 *   tf.colorPoints.push_back({1.f, 0.f, 0.f, 1.f});
 *   tf.opacityPoints.push_back({0.5f, 1.f});
 */
struct TransferFunction
{
  std::vector<ColorPoint> colorPoints;
  std::vector<OpacityPoint> opacityPoints;
  math::box1 range = {};
};

// Default TF (viridis colors, linear opacity ramp, unset range).
TransferFunction makeDefaultTransferFunction();

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

namespace palette {

// Qualitative palettes: maximally-distinct hues for discrete per-primitive
// coloring (e.g. one color per isovalue). RGBA so they map directly onto a
// 'color' / 'primitive.color' parameter.
extern std::vector<float4> tab10;
extern std::vector<float4> tab20;
extern std::vector<float4> set1;
extern std::vector<float4> set2;
extern std::vector<float4> dark2;
extern std::vector<float4> paired;

} // namespace palette

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
