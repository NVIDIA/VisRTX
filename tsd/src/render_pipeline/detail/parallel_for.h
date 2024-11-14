// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ENABLE_CUDA
// thrust
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#define DEVICE_FCN __device__
#define DEVICE_FCN_INLINE __forceinline__ __device__
#else
#include <tbb/parallel_for.h>
#define DEVICE_FCN
#define DEVICE_FCN_INLINE inline
#endif

namespace tsd::detail {

template <typename FCN>
inline void parallel_for(uint32_t start, uint32_t end, FCN &&fcn)
{
#if ENABLE_CUDA
  thrust::for_each(thrust::device,
      thrust::make_counting_iterator(start),
      thrust::make_counting_iterator(end),
      fcn);
#else
  tbb::parallel_for(start, end, fcn);
#endif
}

} // namespace tsd::detail
