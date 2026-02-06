// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "parallel_for.h"

namespace tsd::rendering::detail {

template <typename IN_T, typename OUT_T, typename FCN>
inline void parallel_transform(ComputeStream stream,
    const IN_T *begin,
    const IN_T *end,
    OUT_T *out,
    FCN &&fcn)
{
  const uint32_t size = end - begin;
  parallel_for(
      stream, 0u, size, [=] DEVICE_FCN(uint32_t i) { out[i] = fcn(begin[i]); });
}

} // namespace tsd::rendering::detail
