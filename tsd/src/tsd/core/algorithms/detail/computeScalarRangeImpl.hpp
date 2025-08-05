// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef TSD_USE_CUDA
#define TSD_USE_CUDA 1
#endif

#include "tsd/core/TSDMath.hpp"
#include "tsd/core/scene/objects/Array.hpp"
// std
#include <algorithm>
#include <limits>
#if TSD_USE_CUDA
// thrust
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#endif

namespace tsd::core::detail {

template <int ANARI_ENUM_T>
inline tsd::math::float2 computeScalarRangeImpl(const Array &a)
{
  using properties_t = anari::ANARITypeProperties<ANARI_ENUM_T>;
  using base_t = typename properties_t::base_type;
  tsd::math::float4 min_out{0.f, 0.f, 0.f, 0.f};
  tsd::math::float4 max_out{0.f, 0.f, 0.f, 0.f};

  const auto *begin = a.dataAs<base_t>();
  const auto *end = begin + a.size();
#if TSD_USE_CUDA
  if (a.kind() == Array::MemoryKind::CUDA) {
    const auto minmax = thrust::minmax_element(
        thrust::device_pointer_cast(begin), thrust::device_pointer_cast(end));
    const base_t min_v = *minmax.first;
    const base_t max_v = *minmax.second;
    properties_t::toFloat4(&min_out.x, &min_v);
    properties_t::toFloat4(&max_out.x, &max_v);
  } else {
#endif
    const auto minmax = std::minmax_element(begin, end);
    const auto min_v = *minmax.first;
    const auto max_v = *minmax.second;
    properties_t::toFloat4(&min_out.x, &min_v);
    properties_t::toFloat4(&max_out.x, &max_v);
#if TSD_USE_CUDA
  }
#endif

  return {min_out.x, max_out.x};
}

// NOTE(jda) - Expand the template in separate TUs due to long Thrust compile
//             times.
tsd::math::float2 computeScalarRange_ufixed8(const Array &a);
tsd::math::float2 computeScalarRange_ufixed16(const Array &a);
tsd::math::float2 computeScalarRange_fixed8(const Array &a);
tsd::math::float2 computeScalarRange_fixed16(const Array &a);
tsd::math::float2 computeScalarRange_float32(const Array &a);
tsd::math::float2 computeScalarRange_float64(const Array &a);

} // namespace tsd::core::detail