// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/anari_cpp.hpp>
// helium
#include <helium/helium_math.h>

namespace tsd {
namespace math {

using namespace anari::math;
using namespace helium::math;

static constexpr mat4 IDENTITY_MAT4 = identity;

static constexpr tsd::math::float3 to_float3(const tsd::math::float4 &v)
{
  return tsd::math::float3(v.x, v.y, v.z);
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

static constexpr tsd::math::float3 azelToDir(tsd::math::float2 azel)
{
  const float az = radians(azel.x);
  const float el = radians(azel.y);
  return tsd::math::float3(
      std::sin(az) * std::cos(el), std::sin(el), std::cos(az) * std::cos(el));
}

static constexpr tsd::math::float3 radians(tsd::math::float3 v)
{
  return tsd::math::float3(radians(v.x), radians(v.y), radians(v.z));
};

static constexpr float degrees(float radians)
{
  return radians * 180.f / float(M_PI);
};

static constexpr tsd::math::float3 degrees(tsd::math::float3 v)
{
  return tsd::math::float3(degrees(v.x), degrees(v.y), degrees(v.z));
};

inline tsd::math::mat4 makeValueRangeTransform(float lower, float upper)
{
  const auto scale =
      tsd::math::scaling_matrix(tsd::math::float3(1.f / (upper - lower)));
  const auto translation =
      tsd::math::translation_matrix(tsd::math::float3(-lower, 0, 0));
  return tsd::math::mul(scale, translation);
}

inline tsd::math::mat4 makeValueRangeTransform(const tsd::math::float2 &range)
{
  return makeValueRangeTransform(range.x, range.y);
}

inline void decomposeMatrix(const tsd::math::mat4 &m,
    tsd::math::float3 &scale,
    tsd::math::mat4 &rotation,
    tsd::math::float3 &translation)
{
  // Step 1: Extract translation from the 4th column of the matrix
  translation = to_float3(m[3]);

  // Step 2: Extract scale factors from the lengths of the basis vectors
  auto basisX = to_float3(m[0]); // First column (X-axis basis vector)
  auto basisY = to_float3(m[1]); // Second column (Y-axis basis vector)
  auto basisZ = to_float3(m[2]); // Third column (Z-axis basis vector)

  scale.x = tsd::math::length(basisX);
  scale.y = tsd::math::length(basisY);
  scale.z = tsd::math::length(basisZ);

  // Step 3: Remove scale from the basis vectors to get pure rotation
  auto rotationMatrix = m; // Copy the full 4x4 matrix
  if (scale.x != 0.0f)
    rotationMatrix[0] /= scale.x; // Normalize X-axis
  if (scale.y != 0.0f)
    rotationMatrix[1] /= scale.y; // Normalize Y-axis
  if (scale.z != 0.0f)
    rotationMatrix[2] /= scale.z; // Normalize Z-axis

  // Keep the 4th column as (0, 0, 0, 1) for the rotation matrix
  rotationMatrix[3] = tsd::math::float4(0.0f, 0.0f, 0.0f, 1.0f);
  rotation = rotationMatrix; // Assign normalized rotation matrix
}

inline tsd::math::float3 matrixToAzElRoll(const tsd::math::mat4 &r)
{
  const float r00 = r[0][0], r01 = r[0][1], r02 = r[0][2];
  const float r10 = r[1][0], r11 = r[1][1], r12 = r[1][2];
  const float r20 = r[2][0], r21 = r[2][1], r22 = r[2][2];

  const float elevation = std::asin(-r21);
  const float azimuth = std::atan2(r20, r22);
  const float roll = std::atan2(r01, r11);

  return {azimuth, elevation, roll};
}

} // namespace math

using namespace linalg::aliases;
using mat4 = float4x4;

} // namespace tsd
