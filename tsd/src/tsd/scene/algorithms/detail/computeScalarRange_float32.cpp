// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "computeScalarRangeImpl.hpp"

namespace tsd::scene::detail {

tsd::math::float2 computeScalarRange_float32(const Array &a)
{
  return computeScalarRangeImpl<ANARI_FLOAT32>(a);
}

} // namespace tsd::scene::detail
