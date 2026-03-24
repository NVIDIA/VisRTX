// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/TSDMath.hpp"
// std
#include <algorithm>
#include <cstddef>
#include <vector>

namespace tsd::animation {

/*
 * Strategy used to blend between adjacent keyframe values: step (hold
 * previous), linear, or spherical-linear (SLERP) for quaternion rotations.
 */
enum class InterpolationRule
{
  STEP,
  LINEAR,
  SLERP,
};

/*
 * Result of looking up a time value against a sorted time-base: the indices
 * of the two bracketing keyframes and the normalized interpolation factor.
 *
 * Example:
 *   TimeSample s = findSample(timeBase, t);
 *   auto v = lerp(values[s.lo], values[s.hi], s.alpha);
 */
struct TimeSample
{
  size_t lo{0};
  size_t hi{0};
  float alpha{0.f};
};

inline TimeSample findTimeSample(const std::vector<float> &times, float t)
{
  if (times.size() == 0)
    return {};
  if (times.size() == 1 || t <= times[0])
    return {0, 0, 0.f};
  if (t >= times.back())
    return {times.size() - 1, times.size() - 1, 0.f};

  auto it = std::upper_bound(cbegin(times), cend(times), t);
  auto hi = std::distance(cbegin(times), it);
  auto lo = hi - 1;

  float span = times[hi] - times[lo];
  float alpha = (span > 0.f) ? (t - times[lo]) / span : 0.f;
  return {size_t(lo), size_t(hi), alpha};
}

inline math::mat4 composeTransform(
    math::float4 quat, math::float3 trans, math::float3 scl)
{
  float x = quat.x, y = quat.y, z = quat.z, w = quat.w;
  float x2 = x + x, y2 = y + y, z2 = z + z;
  float xx = x * x2, xy = x * y2, xz = x * z2;
  float yy = y * y2, yz = y * z2, zz = z * z2;
  float wx = w * x2, wy = w * y2, wz = w * z2;

  math::mat4 m;
  m[0] = {(1.f - (yy + zz)) * scl.x, (xy + wz) * scl.x, (xz - wy) * scl.x, 0.f};
  m[1] = {(xy - wz) * scl.y, (1.f - (xx + zz)) * scl.y, (yz + wx) * scl.y, 0.f};
  m[2] = {(xz + wy) * scl.z, (yz - wx) * scl.z, (1.f - (xx + yy)) * scl.z, 0.f};
  m[3] = {trans.x, trans.y, trans.z, 1.f};
  return m;
}

} // namespace tsd::animation
