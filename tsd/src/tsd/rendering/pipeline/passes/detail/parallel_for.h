// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ENABLE_CUDA
// thrust
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#define DEVICE_FCN __device__
#define DEVICE_FCN_INLINE __forceinline__ __device__
#elif defined(ENABLE_TBB)
#include <tbb/parallel_for.h>
#define DEVICE_FCN
#define DEVICE_FCN_INLINE inline
#else
#define DEVICE_FCN
#define DEVICE_FCN_INLINE inline
#endif

#include "ComputeStream.h"

namespace tsd::rendering::detail {

template <typename FCN>
inline void parallel_for(
    ComputeStream stream, uint32_t start, uint32_t end, FCN &&fcn)
{
#ifdef ENABLE_CUDA
  thrust::for_each(thrust::cuda::par.on(stream),
      thrust::make_counting_iterator(start),
      thrust::make_counting_iterator(end),
      fcn);
#elif defined(ENABLE_TBB)
  tbb::parallel_for(start, end, fcn);
#else
  for (auto i = start; i < end; i++)
    fcn(i);
#endif
}

} // namespace tsd::rendering::detail
