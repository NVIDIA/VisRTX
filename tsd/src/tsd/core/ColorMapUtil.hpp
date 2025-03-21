// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/TSDMath.hpp"

namespace tsd {

inline math::mat4 makeColorMapTransform(float lower, float upper)
{
  const auto scale = math::scaling_matrix(math::float3(1.f / (upper - lower)));
  const auto translation = math::translation_matrix(math::float3(-lower, 0, 0));
  return math::mul(scale, translation);
}

} // namespace tsd
