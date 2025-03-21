// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "computeScalarRangeImpl.hpp"

namespace tsd::algorithm::detail {

tsd::float2 computeScalarRange_ufixed16(const Array &a)
{
  return computeScalarRangeImpl<ANARI_UFIXED16>(a);
}

} // namespace tsd::algorithm::detail
