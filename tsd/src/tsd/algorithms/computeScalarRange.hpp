// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef TSD_USE_CUDA
#define TSD_USE_CUDA 1
#endif

#include "tsd/core/Context.hpp"
// std
#include <algorithm>
#include <limits>
#if TSD_USE_CUDA
// thrust
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#endif

namespace tsd::algorithm {

namespace detail {

// NOTE(jda): This is a reduced version of anari::anariTypeInvoke() to lower
//            Thrust/CUDA compile times
template <typename R, template <int> class F, typename... Args>
inline R scalarTypeInvoke(ANARIDataType type, Args &&...args)
{
  // clang-format off
  switch (type) {
  case ANARI_UFIXED8: return F<ANARI_UFIXED8>()(std::forward<Args>(args)...);
  case ANARI_UFIXED16: return F<ANARI_UFIXED16>()(std::forward<Args>(args)...);
  case ANARI_FIXED8: return F<ANARI_FIXED8>()(std::forward<Args>(args)...);
  case ANARI_FIXED16: return F<ANARI_FIXED16>()(std::forward<Args>(args)...);
  case ANARI_FLOAT32: return F<ANARI_FLOAT32>()(std::forward<Args>(args)...);
  case ANARI_FLOAT64: return F<ANARI_FLOAT64>()(std::forward<Args>(args)...);
  default:
    return F<ANARI_UNKNOWN>()(std::forward<Args>(args)...);
  }
  // clang-format off
}

template <int ANARI_ENUM_T>
struct ComputeScalarRange
{
  using properties_t = anari::ANARITypeProperties<ANARI_ENUM_T>;
  using base_t = typename properties_t::base_type;

  tsd::float2 operator()(const Array &a)
  {
    tsd::float4 min_out{0.f, 0.f, 0.f, 0.f};
    tsd::float4 max_out{0.f, 0.f, 0.f, 0.f};

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
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

inline tsd::float2 computeScalarRange(const Array &a)
{
  constexpr float maxFloat = std::numeric_limits<float>::max();
  tsd::float2 retval{maxFloat, -maxFloat};

  const anari::DataType type = a.elementType();
  const bool elementsAreArrays = anari::isArray(type);
  const bool elementsAreScalars =
      !anari::isObject(type) && anari::componentsOf(type) == 1;

  if (auto *ctx = a.context(); elementsAreArrays && ctx) {
    const auto *begin = (uint64_t *)a.data();
    const auto *end = begin + a.size();
    std::for_each(begin, end, [&](uint64_t idx) {
      tsd::float2 subRange{maxFloat, -maxFloat};
      if (auto subArray = ctx->getObject<Array>(idx); subArray)
        subRange = computeScalarRange(*subArray);
      retval.x = std::min(retval.x, subRange.x);
      retval.y = std::max(retval.y, subRange.y);
    });
  } else if (elementsAreScalars) {
    retval = detail::scalarTypeInvoke<tsd::float2, detail::ComputeScalarRange>(
        type, a);
  }

  return retval;
}

} // namespace tsd::algorithm
