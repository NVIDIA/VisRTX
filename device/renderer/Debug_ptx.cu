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

#include "Debug.h"
#include "gpu/shading_api.h"

namespace visrtx {

enum class RayType
{
  SURFACE,
  VOLUME
};

struct RayData
{
  vec3 outColor{0.f};
  Hit hit{};
};

struct VolumeRayData
{
  vec3 outColor{0.f};
  VolumeHit hit{};
};

DECLARE_FRAME_DATA(frameData)

RT_PROGRAM void __closesthit__surface()
{
  auto &rd = ray::rayData<RayData>();

  ray::populateSurfaceHit(rd.hit);

  auto method =
      static_cast<Debug::Method>(frameData.renderer.params.debug.method);

  ray::computeNormal(*rd.hit.geometry, ray::primID(), rd.hit);

  switch (method) {
  case Debug::Method::PRIM_ID:
    rd.outColor = makeRandomColor(ray::primID());
    break;
  case Debug::Method::GEOM_ID:
    rd.outColor = makeRandomColor(ray::objID());
    break;
  case Debug::Method::INST_ID:
    rd.outColor = makeRandomColor(ray::instID());
    break;
  case Debug::Method::RAY_UVW:
    rd.outColor = ray::uvw(rd.hit.geometry->type);
    break;
  case Debug::Method::IS_TRIANGLE:
    rd.outColor = boolColor(rd.hit.geometry->type == GeometryType::TRIANGLE);
    break;
  case Debug::Method::BACKFACE:
    rd.outColor = boolColor(optixIsFrontFaceHit());
    break;
  case Debug::Method::NG:
    rd.outColor = rd.hit.Ng;
    break;
  case Debug::Method::NG_ABS:
    rd.outColor = abs(rd.hit.Ng);
    break;
  case Debug::Method::NS:
    rd.outColor = rd.hit.Ns;
    break;
  case Debug::Method::NS_ABS:
    rd.outColor = abs(rd.hit.Ns);
    break;
  default: {
    rd.outColor = vec3(1.f);
    break;
  }
  }
}

RT_PROGRAM void __closesthit__volume()
{
  auto &vrd = ray::rayData<VolumeRayData>();
  ray::populateVolumeHit(vrd.hit);
}

RT_PROGRAM void __miss__()
{
  // no-op
}

RT_PROGRAM void __raygen__()
{
  /////////////////////////////////////////////////////////////////////////////
  // TODO: clean this up! need to split out Ray/RNG, don't need screen samples
  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;
  auto ray = makePrimaryRay(ss);
  /////////////////////////////////////////////////////////////////////////////

  auto method =
      static_cast<Debug::Method>(frameData.renderer.params.debug.method);
  if (method == Debug::Method::IS_VOLUME) {
    VolumeRayData vrd{};
    intersectVolume(ss, ray, RayType::VOLUME, &vrd);
    if (vrd.hit.foundHit) {
      accumResults(frameData.fb,
          ss.pixel,
          vec4(boolColor(true), 1.f),
          vrd.hit.localRay.t.lower,
          vec3(boolColor(true)),
          -ray.dir);
      return;
    }
  } else {
    RayData rd{};
    intersectSurface(ss, ray, RayType::SURFACE, &rd);
    if (rd.hit.foundHit) {
      accumResults(frameData.fb,
          ss.pixel,
          vec4(rd.outColor, 1.f),
          rd.hit.t,
          vec3(rd.outColor),
          rd.hit.Ng);
      return;
    }
  }

  auto color = frameData.renderer.bgColor;
  accumResults(
      frameData.fb, ss.pixel, color, ray.t.upper, vec3(color), ray.dir);
}

} // namespace visrtx
