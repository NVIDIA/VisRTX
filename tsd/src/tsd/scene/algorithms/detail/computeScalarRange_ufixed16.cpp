// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "computeScalarRangeImpl.hpp"

namespace tsd::scene::detail {

tsd::math::float2 computeScalarRange_ufixed16(const Array &a)
{
  return computeScalarRangeImpl<ANARI_UFIXED16>(a);
}

} // namespace tsd::scene::detail
