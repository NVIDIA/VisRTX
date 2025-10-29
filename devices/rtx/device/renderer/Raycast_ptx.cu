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

#include "gpu/evalShading.h"
#include "gpu/shadingState.h"
#include "gpu/shading_api.h"

namespace visrtx {

enum class RayType
{
  PRIMARY
};

DECLARE_FRAME_DATA(frameData)

VISRTX_GLOBAL void __anyhit__primary()
{
  ray::cullbackFaces();
}

VISRTX_GLOBAL void __closesthit__primary()
{
  ray::populateHit();
}

VISRTX_GLOBAL void __miss__()
{
  // no-op
}

VISRTX_GLOBAL void __raygen__()
{
  auto &rendererParams = frameData.renderer;

  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;
  auto ray = makePrimaryRay(ss, true /*pixel centered*/);
  float tmax = ray.t.upper;

  vec3 outputColor(0.f);
  vec3 outputNormal = ray.dir;
  float outputOpacity = 0.f;
  float depth = 1e30f;
  uint32_t primID = ~0u;
  uint32_t objID = ~0u;
  uint32_t instID = ~0u;
  bool firstHit = true;

  while (outputOpacity < 0.99f) {
    SurfaceHit surfaceHit;
    ray.t.upper = tmax;
    surfaceHit.foundHit = false;
    intersectSurface(ss,
        ray,
        RayType::PRIMARY,
        &surfaceHit,
        primaryRayOptiXFlags(rendererParams));

    vec3 color(0.f);
    float opacity = 0.f;

    if (surfaceHit.foundHit) {
      uint32_t vObjID = ~0u;
      uint32_t vInstID = ~0u;
      const float vDepth = rayMarchAllVolumes(ss,
          ray,
          RayType::PRIMARY,
          surfaceHit.t,
          rendererParams.inverseVolumeSamplingRate,
          color,
          opacity,
          vObjID,
          vInstID);

      if (firstHit) {
        const bool volumeFirst = vDepth < surfaceHit.t;
        if (volumeFirst) {
          outputNormal = -ray.dir;
          depth = vDepth;
          primID = 0;
          objID = vObjID;
          instID = vInstID;
        } else {
          outputNormal = surfaceHit.Ng;
          depth = surfaceHit.t;
          primID = computeGeometryPrimId(surfaceHit);
          objID = surfaceHit.objID;
          instID = surfaceHit.instID;
        }
        firstHit = false;
      }

      MaterialShadingState shadingState;
      materialInitShading(
          &shadingState, frameData, *surfaceHit.material, surfaceHit);
      auto materialBaseColor = materialEvaluateTint(shadingState);
      auto materialOpacity = materialEvaluateOpacity(shadingState);

      const auto lighting = glm::abs(glm::dot(ray.dir, surfaceHit.Ns))
          * rendererParams.ambientColor;
      accumulateValue(color, materialBaseColor * lighting, opacity);
      accumulateValue(opacity, materialOpacity, opacity);

      color *= opacity;
      accumulateValue(outputColor, color, outputOpacity);
      accumulateValue(outputOpacity, opacity, outputOpacity);

      ray.t.lower = surfaceHit.t + surfaceHit.epsilon;
    } else {
      uint32_t vObjID = ~0u;
      uint32_t vInstID = ~0u;
      const float volumeDepth = rayMarchAllVolumes(ss,
          ray,
          RayType::PRIMARY,
          ray.t.upper,
          rendererParams.inverseVolumeSamplingRate,
          color,
          opacity,
          vObjID,
          vInstID);

      if (firstHit) {
        depth = min(depth, volumeDepth);
        primID = 0;
        objID = vObjID;
        instID = vInstID;
      }

      color *= opacity;

      const auto bg = getBackground(frameData, ss.screen, ray.dir);
      accumulateValue(color, vec3(bg) * bg.a, opacity);
      accumulateValue(opacity, bg.w, opacity);
      accumulateValue(outputColor, color, outputOpacity);
      accumulateValue(outputOpacity, opacity, outputOpacity);
      break;
    }
  }

  accumResults(frameData,
      ss.pixel,
      vec4(outputColor, outputOpacity),
      depth,
      outputColor,
      outputNormal,
      primID,
      objID,
      instID);
}

} // namespace visrtx
