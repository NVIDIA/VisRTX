/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "gpu/gpu_decl.h"

#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
// std
#include <limits>
#include <variant>
// cuda
#include <cuda_runtime.h>

namespace visrtx {

using namespace glm;

template <typename T>
struct range_t
{
  using element_t = T;

  range_t() = default;
  VISRTX_HOST_DEVICE range_t(const T &t) : lower(t), upper(t) {}
  VISRTX_HOST_DEVICE range_t(const T &_lower, const T &_upper)
      : lower(_lower), upper(_upper)
  {}

  VISRTX_HOST_DEVICE range_t<T> &extend(const T &t)
  {
    lower = min(lower, t);
    upper = max(upper, t);
    return *this;
  }

  VISRTX_HOST_DEVICE range_t<T> &extend(const range_t<T> &t)
  {
    lower = min(lower, t.lower);
    upper = max(upper, t.upper);
    return *this;
  }

  T lower{T(std::numeric_limits<float>::max())};
  T upper{T(-std::numeric_limits<float>::max())};
};

using box1 = range_t<float>;
using box2 = range_t<vec2>;
using box3 = range_t<vec3>;

// Operations on range_t //

VISRTX_HOST_DEVICE box1 make_box1(const vec2 &v)
{
  return box1(v.x, v.y);
}

template <typename T>
VISRTX_HOST_DEVICE typename range_t<T>::element_t size(const range_t<T> &r)
{
  return r.upper - r.lower;
}

template <typename T>
VISRTX_HOST_DEVICE typename range_t<T>::element_t center(const range_t<T> &r)
{
  return .5f * (r.lower + r.upper);
}

template <typename T>
VISRTX_HOST_DEVICE typename range_t<T>::element_t clamp(
    const typename range_t<T>::element_t &t, const range_t<T> &r)
{
  return max(r.lower, min(t, r.upper));
}

VISRTX_HOST_DEVICE bool contains(float v, const box1 &r)
{
  return v >= r.lower && v <= r.upper;
}

VISRTX_HOST_DEVICE float position(float v, const box1 &r)
{
  v = clamp(v, r);
  return (v - r.lower) * (1.f / size(r));
}

template <typename T>
VISRTX_HOST_DEVICE size_t largest_axis(const range_t<T> &r)
{
  return 0;
}

template <>
VISRTX_HOST_DEVICE size_t largest_axis(const box3 &r)
{
  auto d = size(r);
  size_t axis = 0;
  if (d[0] < d[1])
    axis = 1;
  if (d[axis] < d[2])
    axis = 2;
  return axis;
}

VISRTX_HOST_DEVICE float largest_extent(const box3 &r)
{
  return size(r)[largest_axis(r)];
}

VISRTX_HOST_DEVICE float half_area(const box3 &r)
{
  auto d = size(r);
  return (d[0] + d[1]) * d[2] + d[0] * d[1];
}

VISRTX_HOST_DEVICE float volume(const box3 &r)
{
  auto d = size(r);
  return d[0] * d[1] * d[2];
}

VISRTX_HOST_DEVICE bool empty(const box3 &r)
{
  return r.lower.x > r.upper.x || r.lower.y > r.upper.y
      || r.lower.z > r.upper.z;
}

// Helper functions ///////////////////////////////////////////////////////////

VISRTX_HOST_DEVICE int64_t iDivUp(int64_t a, int64_t b)
{
  return (a + b - 1) / b;
}

VISRTX_HOST_DEVICE uint32_t cvt_uint32(const float &f)
{
  return (uint32_t)(255.f * glm::clamp(f, 0.f, 1.f));
}

VISRTX_HOST_DEVICE uint32_t cvt_uint32(const vec4 &v)
{
  return (cvt_uint32(v.x) << 0) | (cvt_uint32(v.y) << 8)
      | (cvt_uint32(v.z) << 16) | (cvt_uint32(v.w) << 24);
}

VISRTX_HOST_DEVICE float cvt_float(const uint8_t &i)
{
  return i / 255.f;
}

VISRTX_HOST_DEVICE vec4 cvt_float4(const uint32_t &i)
{
  return vec4(cvt_float(static_cast<uint8_t>(i >> 0)),
      cvt_float(static_cast<uint8_t>(i >> 8)),
      cvt_float(static_cast<uint8_t>(i >> 16)),
      cvt_float(static_cast<uint8_t>(i >> 24)));
}

} // namespace visrtx
