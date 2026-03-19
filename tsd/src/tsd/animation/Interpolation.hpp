// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

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

} // namespace tsd::animation
