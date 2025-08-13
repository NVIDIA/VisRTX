/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <curand_uniform.h>
#include <device_atomic_functions.h>
#include <algorithm>
#include <cmath>
#include <glm/ext/matrix_float3x3.hpp>
#include <glm/ext/vector_float3.hpp>
#include <limits>
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"

#include <glm/gtx/color_space.hpp>

#include <cub/thread/thread_search.cuh>

namespace visrtx {

struct LightSample
{
  vec3 radiance;
  vec3 dir;
  float dist;
  float pdf;
};

namespace detail {

VISRTX_DEVICE LightSample sampleDirectionalLight(
    const LightGPUData &ld, const mat4 &xfm)
{
  LightSample ls;
  ls.dir = xfmVec(xfm, -ld.distant.direction);
  ls.dist = std::numeric_limits<float>::infinity();
  ls.radiance = ld.color * ld.distant.irradiance;
  ls.pdf = 1.f;

  return ls;
}

VISRTX_DEVICE LightSample samplePointLight(
    const LightGPUData &ld, const mat4 &xfm, const Hit &hit)
{
  LightSample ls;
  ls.dir = xfmPoint(xfm, ld.point.position) - hit.hitpoint;
  ls.dist = length(ls.dir);
  ls.dir = glm::normalize(ls.dir);
  ls.radiance = ld.color * ld.point.intensity;
  ls.pdf = 1.f;
  return ls;
}

VISRTX_DEVICE LightSample sampleSpotLight(
    const LightGPUData &ld, const mat4 &xfm, const Hit &hit)
{
  LightSample ls;
  ls.dir = glm::normalize(xfmPoint(xfm, ld.spot.position) - hit.hitpoint);
  ls.dist = length(ls.dir);
  float spot = dot(normalize(ld.spot.direction), -ls.dir);
  if (spot < ld.spot.cosOuterAngle)
    spot = 0.f;
  else if (spot > ld.spot.cosInnerAngle)
    spot = 1.f;
  else {
    spot = (spot - ld.spot.cosOuterAngle)
        / (ld.spot.cosInnerAngle - ld.spot.cosOuterAngle);
    spot = spot * spot * (3.f - 2.f * spot);
  }
  ls.radiance = ld.color * ld.spot.intensity * spot;
  ls.pdf = 1.f;
  return ls;
}

VISRTX_DEVICE int inverseSampleCDF(const float *cdf, int size, float u)
{
  return cub::LowerBound(cdf, size, u);
}

VISRTX_DEVICE LightSample sampleHDRILight(
    const LightGPUData &ld, const mat4 &xfm, const vec3 &dir)
{
  // Compute the UV coordinates from the direction
  auto thetaPhi = sphericalCoordsFromDirection(ld.hdri.xfm * dir);
  auto uv = glm::vec2(thetaPhi.y, thetaPhi.x)
      / glm::vec2(float(M_PI) * 2.0f, float(M_PI));

  auto radiance = sampleHDRI(ld, uv);
  auto pdf = dot(radiance, {0.2126f, 0.7152f, 0.0722f}) * sinf(thetaPhi.x) * ld.hdri.pdfWeight;

  // Sample the HDRI texture
  LightSample ls;
  ls.dir = xfmVec(xfm, dir);
  ls.dist = std::numeric_limits<float>::infinity();
  ls.radiance = radiance * ld.hdri.scale;
  ls.pdf = pdf;

  return ls;
}

VISRTX_DEVICE LightSample sampleHDRILight(
    const LightGPUData &ld, const mat4 &xfm, const Hit &hit, RandState &rs)
{
  // Row and column sampling
  auto y = inverseSampleCDF(
      ld.hdri.marginalCDF, ld.hdri.size.y, curand_uniform(&rs));
  auto x = inverseSampleCDF(ld.hdri.conditionalCDF + y * ld.hdri.size.x,
      ld.hdri.size.x,
      curand_uniform(&rs));

  auto xy = glm::uvec2(x, y);

#ifdef VISRTX_ENABLE_HDRI_SAMPLING_DEBUG
  if (ld.hdri.samples) {
    atomicInc(ld.hdri.samples + y * ld.hdri.size.x + x, ~0u);
  }
#endif
  auto jitter = glm::vec2(curand_uniform(&rs), curand_uniform(&rs));
  auto uv =
      glm::clamp((glm::vec2(xy) + jitter) / glm::vec2(ld.hdri.size), 0.f, 1.f);

  // And spherical coordinates
  auto thetaPhi = float(M_PI) * glm::vec2(uv.y, 2.0f * (uv.x));

  // Get world direction

  // Compute PDF
  auto radiance = sampleHDRI(ld, uv);
  auto pdf = dot(radiance, {0.2126f, 0.7152f, 0.0722f}) * sinf(thetaPhi.x) * ld.hdri.pdfWeight;

  LightSample ls;
  // ld.hdri.xfm is computed in HDRI.cpp and is made so it is an orthogonal
  // matrix. So transposing is actually the same as inverting. Use RHS
  // multiplication so we don't have to transpose the matrix.
  ls.dir = sphericalCoordsToDirection(thetaPhi) * ld.hdri.xfm;
  ls.dist = 1e20f;
  ls.radiance = radiance * ld.hdri.scale;
  ls.pdf = pdf;

  return ls;
}

} // namespace detail

VISRTX_DEVICE LightSample sampleLight(
    ScreenSample &ss, const Hit &hit, DeviceObjectIndex idx, const mat4 &xfm)
{
  auto &ld = ss.frameData->registry.lights[idx];

  switch (ld.type) {
  case LightType::DIRECTIONAL:
    return detail::sampleDirectionalLight(ld, xfm);
  case LightType::POINT:
    return detail::samplePointLight(ld, xfm, hit);
  case LightType::SPOT:
    return detail::sampleSpotLight(ld, xfm, hit);
  case LightType::HDRI:
    return detail::sampleHDRILight(ld, xfm, hit, ss.rs);
  default:
    break;
  }

  return {};
}

} // namespace visrtx
