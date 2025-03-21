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

#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"

namespace visrtx {

struct LightSample
{
  vec3 radiance;
  vec3 dir;
  float dist;
  float pdf;
};

namespace detail {

VISRTX_DEVICE LightSample sampleDirectionalLight(const LightGPUData &ld)
{
  LightSample ls;
  ls.dir = -ld.distant.direction;
  ls.dist = FLT_MAX;
  ls.radiance = ld.color * ld.distant.irradiance;
  ls.pdf = 1.f;
  return ls;
}

VISRTX_DEVICE LightSample samplePointLight(const LightGPUData &ld, const Hit &hit)
{
  LightSample ls;
  ls.dir = ld.point.position - hit.hitpoint;
  ls.dist = length(ls.dir);
  ls.dir = glm::normalize(ls.dir);
  ls.radiance = ld.color * ld.point.intensity;
  ls.pdf = 1.f;
  return ls;
}

VISRTX_DEVICE LightSample sampleSpotLight(const LightGPUData &ld, const Hit &hit)
{
  LightSample ls;
  ls.dir = ld.spot.position - hit.hitpoint;
  ls.dist = length(ls.dir);
  ls.dir = glm::normalize(ls.dir);
  float spot = dot(normalize(ld.spot.direction), -ls.dir);
  if (spot < ld.spot.cosOuterAngle) spot = 0.f;
  else if (spot > ld.spot.cosInnerAngle) spot = 1.f;
  else {
    spot = (spot - ld.spot.cosOuterAngle) / (ld.spot.cosInnerAngle - ld.spot.cosOuterAngle);
    spot = spot * spot * (3.f - 2.f * spot);
  }
  ls.radiance = ld.color * ld.spot.intensity * spot;
  ls.pdf = 1.f;
  return ls;
}

VISRTX_DEVICE LightSample sampleHDRILight(
  const LightGPUData &ld,
  const Hit &hit,
  RandState &rs)
{
  LightSample ls;
  ls.dir = randomDir(rs);
  ls.dist = 1e20f;
  ls.radiance = sampleHDRI(ld, ls.dir);
  ls.pdf = 0.f;
  return ls;
}

} // namespace detail

VISRTX_DEVICE LightSample sampleLight(
    ScreenSample &ss, const Hit &hit, DeviceObjectIndex idx)
{
  auto &ld = ss.frameData->registry.lights[idx];

  switch (ld.type) {
  case LightType::DIRECTIONAL:
    return detail::sampleDirectionalLight(ld);
  case LightType::POINT:
    return detail::samplePointLight(ld, hit);
  case LightType::SPOT:
    return detail::sampleSpotLight(ld, hit);
  case LightType::HDRI:
    return detail::sampleHDRILight(ld, hit, ss.rs);
  default:
    break;
  }

  return {};
}

} // namespace visrtx
