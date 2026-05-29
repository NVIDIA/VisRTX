// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/anari_cpp.hpp>
// helium
#include <helium/helium_math.h>
// std
#include <algorithm>
#include <cmath>
#include <limits>

namespace tsd {
namespace core {
namespace math {

using namespace anari::math;
using namespace helium::math;

static constexpr mat3 IDENTITY_MAT3 = identity;
static constexpr mat4 IDENTITY_MAT4 = identity;
static constexpr float inf = std::numeric_limits<float>::infinity();

static constexpr tsd::core::math::float3 to_float3(
    const tsd::core::math::float4 &v)
{
  return tsd::core::math::float3(v.x, v.y, v.z);
};

template <typename T>
static constexpr bool neql(T a, T b, T eps = 1e-6)
{
  return std::abs(a - b) <= eps;
}

static constexpr float radians(float degrees)
{
  return degrees * M_PI / 180.f;
};

static constexpr tsd::core::math::float3 radians(tsd::core::math::float3 v)
{
  return tsd::core::math::float3(radians(v.x), radians(v.y), radians(v.z));
};

static constexpr float degrees(float radians)
{
  return radians * 180.f / float(M_PI);
};

static constexpr tsd::core::math::float3 degrees(tsd::core::math::float3 v)
{
  return tsd::core::math::float3(degrees(v.x), degrees(v.y), degrees(v.z));
};

inline float normalizeDegrees(float v)
{
  v = std::fmod(v, 360.f);
  return v < 0.f ? v + 360.f : v;
}

inline tsd::core::math::float3 normalizeDegrees(tsd::core::math::float3 v)
{
  return tsd::core::math::float3(
      normalizeDegrees(v.x), normalizeDegrees(v.y), normalizeDegrees(v.z));
}

static constexpr tsd::core::math::float3 azelToDir(tsd::core::math::float2 azel)
{
  const float az = radians(azel.x);
  const float el = radians(azel.y);
  return tsd::core::math::float3(
      std::sin(az) * std::cos(el), std::sin(el), std::cos(az) * std::cos(el));
}

inline tsd::core::math::float3 normalizeColor(
    const tsd::core::math::float3 &color)
{
  const float maxValue = tsd::core::math::maxelem(color);
  return maxValue > 1.f ? color / maxValue : color;
}

inline tsd::core::math::mat4 makeValueRangeTransform(float lower, float upper)
{
  const auto scale = tsd::core::math::scaling_matrix(
      tsd::core::math::float3(1.f / (upper - lower)));
  const auto translation = tsd::core::math::translation_matrix(
      tsd::core::math::float3(-lower, 0, 0));
  return tsd::core::math::mul(scale, translation);
}

inline tsd::core::math::mat4 makeValueRangeTransform(
    const tsd::core::math::float2 &range)
{
  return makeValueRangeTransform(range.x, range.y);
}

inline void decomposeMatrix(const tsd::core::math::mat4 &m,
    tsd::core::math::float3 &scale,
    tsd::core::math::mat4 &rotation,
    tsd::core::math::float3 &translation)
{
  // Step 1: Extract translation from the 4th column of the matrix
  translation = to_float3(m[3]);

  // Step 2: Extract scale factors from the lengths of the basis vectors
  auto basisX = to_float3(m[0]); // First column (X-axis basis vector)
  auto basisY = to_float3(m[1]); // Second column (Y-axis basis vector)
  auto basisZ = to_float3(m[2]); // Third column (Z-axis basis vector)

  scale.x = tsd::core::math::length(basisX);
  scale.y = tsd::core::math::length(basisY);
  scale.z = tsd::core::math::length(basisZ);

  // Step 3: Remove scale from the basis vectors to get pure rotation
  auto rotationMatrix = m; // Copy the full 4x4 matrix
  if (scale.x != 0.0f)
    rotationMatrix[0] /= scale.x; // Normalize X-axis
  if (scale.y != 0.0f)
    rotationMatrix[1] /= scale.y; // Normalize Y-axis
  if (scale.z != 0.0f)
    rotationMatrix[2] /= scale.z; // Normalize Z-axis

  // Keep the 4th column as (0, 0, 0, 1) for the rotation matrix
  rotationMatrix[3] = tsd::core::math::float4(0.0f, 0.0f, 0.0f, 1.0f);
  rotation = rotationMatrix; // Assign normalized rotation matrix
}

inline tsd::core::math::float3 matrixToAzElRoll(const tsd::core::math::mat4 &r)
{
  const float m00 = r[0][0];
  const float m01 = r[1][0];
  const float m02 = r[2][0];
  const float m10 = r[0][1];
  const float m11 = r[1][1];
  const float m12 = r[2][1];
  const float m22 = r[2][2];

  const float elevation = std::asin(std::clamp(-m12, -1.f, 1.f));
  float azimuth = 0.f;
  float roll = 0.f;

  if (std::abs(std::cos(elevation)) > 1e-5f) {
    azimuth = std::atan2(m02, m22);
    roll = std::atan2(m10, m11);
  } else {
    roll = std::atan2(-m01, m00);
  }

  return {azimuth, elevation, roll};
}

} // namespace math

using namespace linalg::aliases;
using mat4 = float4x4;

} // namespace core

namespace math = core::math;

} // namespace tsd

namespace anari {
// box1 is already exposed by helium.
ANARI_TYPEFOR_SPECIALIZATION(tsd::core::math::box2, ANARI_FLOAT32_BOX2);
ANARI_TYPEFOR_SPECIALIZATION(tsd::core::math::box3, ANARI_FLOAT32_BOX3);
} // namespace anari
