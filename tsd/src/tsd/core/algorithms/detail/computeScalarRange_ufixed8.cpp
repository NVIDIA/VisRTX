// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "computeScalarRangeImpl.hpp"

namespace tsd::core::detail {

tsd::math::float2 computeScalarRange_ufixed8(const Array &a)
{
  return computeScalarRangeImpl<ANARI_UFIXED8>(a);
}

} // namespace tsd::core::detail
