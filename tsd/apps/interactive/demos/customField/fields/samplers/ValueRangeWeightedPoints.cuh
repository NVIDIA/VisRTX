// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../WeightedPointsFieldData.h"
#include "gpu/gpu_math.h"

// Conservative value interval over an object-space AABB for a non-negative
// weighted-points field. The lower bound is always 0; the upper bound sums each
// node's largest Gaussian contribution over the box (evaluated at the box point
// closest to the node center). Conservative because the sampler's value at any
// point is a sum over an LOD cut, which is a SUBSET of all nodes, and each
// node's box maximum upper-bounds its contribution there.
inline __host__ __device__ visrtx::box1 valueRangeWeightedPoints(
    const WeightedPointsFieldData &field,
    const visrtx::vec3 &boxLo,
    const visrtx::vec3 &boxHi)
{
  if (!field.values || field.numNodes <= 0)
    return visrtx::box1{0.f, field.maxValue};

  const float k = field.inv2SigmaSq; // = 1/(2 sigma^2) > 0
  float hi = 0.f;
  for (int i = 0; i < field.numNodes; ++i) {
    const float w = field.values[i * 4 + 3];
    if (w <= 0.f)
      continue;
    const visrtx::vec3 c(field.values[i * 4 + 0],
        field.values[i * 4 + 1],
        field.values[i * 4 + 2]);
    // Closest box point to the node center -> largest exp(-d^2 k) over the box.
    const visrtx::vec3 dNear = c - glm::clamp(c, boxLo, boxHi);
    hi += w * expf(-glm::dot(dNear, dNear) * k);
  }
  return visrtx::box1{0.f, hi};
}
