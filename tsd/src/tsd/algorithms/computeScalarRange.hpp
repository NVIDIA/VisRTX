// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TSDMath.hpp"
#include "tsd/objects/Array.hpp"
// std
#include <algorithm>

namespace tsd::algorithm {

namespace detail {

template <int ANARI_ENUM_T>
struct ComputeScalarRange
{
  using properties_t = anari::ANARITypeProperties<ANARI_ENUM_T>;
  using base_t = typename properties_t::base_type;

  float2 operator()(const Array &a)
  {
    const auto *begin = a.dataAs<base_t>();
    const auto *end = begin + a.size();
    const auto minmax = std::minmax_element(begin, end);
    const auto min_v = *minmax.first;
    const auto max_v = *minmax.second;

    float4 min_out{0.f, 0.f, 0.f, 0.f};
    properties_t::toFloat4(&min_out.x, &min_v);
    float4 max_out{0.f, 0.f, 0.f, 0.f};
    properties_t::toFloat4(&max_out.x, &max_v);

    return {min_out.x, max_out.x};
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

inline float2 computeScalarRange(const Array &a)
{
  const anari::DataType type = a.elementType();
  if (anari::isObject(type) || anari::componentsOf(type) != 1)
    return {0.f, 0.f};
  else
    return anari::anariTypeInvoke<float2, detail::ComputeScalarRange>(type, a);
}

} // namespace tsd::algorithm
