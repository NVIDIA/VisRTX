// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "parallel_for.h"

#ifdef ENABLE_CUDA
#include <thrust/transform_reduce.h>
#elif defined(ENABLE_TBB)
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#endif

namespace tsd::rendering::detail {

template <typename T, typename TRANSFORM_FCN, typename REDUCE_FCN>
inline T parallel_reduce(ComputeStream stream,
    uint32_t start,
    uint32_t end,
    T init,
    TRANSFORM_FCN &&transform,
    REDUCE_FCN &&reduce)
{
#ifdef ENABLE_CUDA
  return thrust::transform_reduce(thrust::cuda::par.on(stream),
      thrust::make_counting_iterator(start),
      thrust::make_counting_iterator(end),
      transform,
      init,
      reduce);
#elif defined(ENABLE_TBB)
  return tbb::parallel_reduce(
      tbb::blocked_range<uint32_t>(start, end),
      init,
      [&](const tbb::blocked_range<uint32_t> &r, T partial) {
        for (auto i = r.begin(); i < r.end(); i++)
          partial = reduce(partial, transform(i));
        return partial;
      },
      reduce);
#else
  T result = init;
  for (auto i = start; i < end; i++)
    result = reduce(result, transform(i));
  return result;
#endif
}

} // namespace tsd::rendering::detail
