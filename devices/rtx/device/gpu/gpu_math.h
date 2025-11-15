/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu_decl.h"

// glm
#include <glm/ext/matrix_float3x3.hpp>
#include <glm/ext/matrix_float3x4.hpp>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/fwd.hpp>
// std
#include <limits>

namespace visrtx {

using glm::fquat;
using glm::ivec1;
using glm::ivec2;
using glm::ivec3;
using glm::ivec4;
using glm::mat2;
using glm::mat2x2;
using glm::mat2x3;
using glm::mat2x4;
using glm::mat3;
using glm::mat3x2;
using glm::mat3x3;
using glm::mat3x4;
using glm::mat4;
using glm::mat4x2;
using glm::mat4x3;
using glm::mat4x4;
using glm::quat;
using glm::uvec1;
using glm::uvec2;
using glm::uvec3;
using glm::uvec4;
using glm::vec1;
using glm::vec2;
using glm::vec3;
using glm::vec4;

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

struct Ray
{
  vec3 org;
  vec3 dir;
  box1 t{0.f, std::numeric_limits<float>::max()};
};

struct InstanceSurfaceGPUData;
struct InstanceVolumeGPUData;
struct GeometryGPUData;
struct MaterialGPUData;
struct VolumeGPUData;

struct SurfaceHit
{
  mat3x4 worldToObject;
  mat3x4 objectToWorld;

  vec3 hitpoint;
  float t;
  vec3 Ng;
  uint32_t primID{~0u};
  vec3 Ns;
  uint32_t objID{~0u};
  vec3 uvw;
  uint32_t instID{~0u};
  vec3 tU;
  float epsilon;
  vec3 tV;
  bool isFrontFace;
  bool foundHit;

  const InstanceSurfaceGPUData *instance{nullptr};
  const GeometryGPUData *geometry{nullptr};
  const MaterialGPUData *material{nullptr};
};

struct VolumeHit
{
  mat3x4 worldToObject;
  mat3x4 objectToWorld;

  const VolumeGPUData *volume{nullptr};
  const InstanceVolumeGPUData *instance{nullptr};
  Ray localRay;

  bool foundHit;
};

using Hit = SurfaceHit;

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
  using glm::min, glm::max;
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

VISRTX_HOST_DEVICE float pow2(float f)
{
  return f * f;
}

VISRTX_HOST_DEVICE float pow5(float f)
{
  return f * f * f * f * f;
}

VISRTX_HOST_DEVICE float clampf(float f, float lo, float hi)
{
  return glm::max(lo, glm::min(hi, f));
}

VISRTX_HOST_DEVICE vec2 sphericalCoordsFromDirection(vec3 dir)
{
  float theta = acosf(glm::clamp(dir.z, -1.0f, 1.0f));
  float p = atan2f(dir.y, dir.x);
  float phi = p < 0.f ? p + 2.f * float(M_PI) : p;

  return vec2(theta, phi);
}

VISRTX_HOST_DEVICE vec3 sphericalCoordsToDirection(vec2 thetaPhi)
{
  const float sinTheta = sinf(thetaPhi.x);

  return vec3(sinTheta * cosf(thetaPhi.y),
      sinTheta * sinf(thetaPhi.y),
      cosf(thetaPhi.x));
}

template <typename T>
VISRTX_HOST_DEVICE T heaviside(const T &x)
{
  return x < T(0.0) ? T(0.0) : T(1.0);
}

VISRTX_HOST_DEVICE bool intersectBox(
    const box3 &b, const vec3 &org, const vec3 &dir, box1 &inout)
{
  const vec3 mins = (b.lower - org) * (1.f / dir);
  const vec3 maxs = (b.upper - org) * (1.f / dir);
  const vec3 nears = glm::min(mins, maxs);
  const vec3 fars = glm::max(mins, maxs);
  const float tin = glm::max(nears.x, glm::max(nears.y, nears.z));
  const float tout = glm::min(fars.x, glm::min(fars.y, fars.z));
  if (tin < tout)
    inout = box1(tin, tout);
  return tin < tout;
}

VISRTX_HOST_DEVICE vec2 uniformSampleDisk(float radius, const vec2 &s)
{
  const float r = sqrtf(s.x) * radius;
  const float phi = 2.f * float(M_PI) * s.y;
  return vec2{r * cosf(phi), r * sinf(phi)};
}

VISRTX_HOST_DEVICE vec3 uniformSampleCone(
    float cosThetaMax, // cosine of cone opening angle
    vec3 s) // rand uniforms in [0,1)
{
  // Sample direction in cone
  float phi = 2.0f * M_PI * s.x;
  float cosTheta = (1.0f - s.y) + s.y * cosThetaMax;
  float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

  // Sample distance with cube-root for uniform volume
  float d = powf(s.z, 1.0f / 3.0f);

  return vec3(d * sinTheta * cosf(phi), d * sinTheta * sinf(phi), d * cosTheta);
}

VISRTX_HOST_DEVICE vec3 xfmVec(const mat4 &m, const vec3 &p)
{
  return mat3(m) * p;
}

VISRTX_HOST_DEVICE vec3 xfmPoint(const mat4 &m, const vec3 &p)
{
  return m * vec4(p, 1.0f);
}

} // namespace visrtx
