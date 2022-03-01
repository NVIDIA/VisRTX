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

#include "gpu/shading_api.h"

namespace visrtx {

enum class RayType
{
  DIFFUSE_RADIANCE
};

struct PathData
{
  int depth{0};
  float Lw{1.f};
  Hit currentHit{};
};

DECLARE_FRAME_DATA(frameData)

// OptiX programs /////////////////////////////////////////////////////////////

RT_PROGRAM void __closesthit__()
{
  ray::populateHit();
}

RT_PROGRAM void __miss__()
{
  // no-op
}

RT_PROGRAM void __raygen__()
{
  const auto &rendererParams = frameData.renderer.params.dpt;

  PathData pathData;

  auto &hit = pathData.currentHit;

  /////////////////////////////////////////////////////////////////////////////
  // TODO: clean this up! need to split out Ray/RNG, don't need screen samples
  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;
  auto ray = makePrimaryRay(ss);
  auto tmax = ray.t.upper;
  /////////////////////////////////////////////////////////////////////////////

  vec3 outColor(0.f);
  vec3 outNormal = ray.dir;
  float outDepth = tmax;

  float outOpacity(0.f);

  do {
    hit.foundHit = false;
    intersectSurface(ss, ray, RayType::DIFFUSE_RADIANCE, &hit);

    float volumeOpacity = 0.f;
    vec3 volumeColor(0.f);
    const float volumeDepth = rayMarchAllVolumes(ss,
        ray,
        RayType::DIFFUSE_RADIANCE,
        hit.foundHit ? hit.t : ray.t.upper,
        volumeColor,
        volumeOpacity);

    if (!hit.foundHit) {
      if (pathData.depth == 0) {
        outColor = volumeColor;
        outDepth = volumeDepth;
        outNormal = -ray.dir;
        outOpacity = volumeOpacity;
        accumulateValue(outColor, vec3(frameData.renderer.bgColor), outOpacity);
        accumulateValue(outOpacity, frameData.renderer.bgColor.w, outOpacity);
      }
      break;
    }

    ray.dir = randomDir(ss.rs, hit.normal);
    ray.org = hit.hitpoint + (hit.epsilon * hit.normal);
    ray.t.lower = hit.epsilon;
    ray.t.upper = tmax;

    if (pathData.depth == 0) {
      const auto &material = *hit.material;
      const auto mat_baseColor =
          getMaterialParameter(frameData, material.baseColor, hit);
      outColor = volumeColor;
      accumulateValue(outColor, mat_baseColor, volumeOpacity);
      outOpacity = 1.f;
      outDepth = min(hit.t, volumeDepth);
      outNormal = hit.normal;
    }

    if (pathData.depth != 0)
      pathData.Lw *= rendererParams.R * (1.f - volumeOpacity);
    pathData.depth++;
  } while (pathData.depth < rendererParams.maxDepth);

  accumResults(frameData.fb,
      ss.pixel,
      vec4(pathData.Lw * outColor, 1.f),
      outDepth,
      outColor,
      outNormal);
}

} // namespace visrtx
