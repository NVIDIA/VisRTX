/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu/evalShading.h"
#include "gpu/gpu_util.h"
#include "gpu/intersectRay.h"
#include "gpu/renderer/common.h"
#include "gpu/shadingState.h"
#include "gpu/volumeIntegration.h"

// Shared helper functions for ray generation across renderers

namespace visrtx {

// Surface SHADOW ray → accumulated opacity (0 = unblocked, 1 = blocked).
// Float payload init 0, accumulateValue(o, α, o) in __anyhit__shadow. The
// path-tracing renderer uses vec3 transmittance instead — see
// shadowTransmittance.h.
VISRTX_DEVICE float surfaceShadowOpacity(ScreenSample &ss, const Ray &r)
{
  float a = 0.0f;
  intersectSurface(
      ss, r, RayType::SHADOW, &a, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return a;
}

// Volume SHADOW ray opacity. Same payload convention as surfaceShadowOpacity.
VISRTX_DEVICE float volumeShadowOpacity(ScreenSample &ss, const Ray &r)
{
  float a = 0.0f;
  intersectVolume(
      ss, r, RayType::SHADOW, &a, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return a;
}

// Templated rendering loop
// ShadingPolicy must implement:
//   static VISRTX_DEVICE vec3 shadeSurface(
//       const MaterialShadingState &shadingState,
//       ScreenSample &ss,
//       const Ray &ray,
//       const SurfaceHit &hit)
// returning a vec3 of reflected + emitted radiance (coverage/transmission are
// queried by the loop, not returned)
template <typename ShadingPolicy>
VISRTX_DEVICE void renderPixel(FrameGPUData &frameData, ScreenSample ss)
{
  auto &rendererParams = frameData.renderer;

  for (int i = 0; i < frameData.renderer.numIterations; i++) {
    bool isVeryFirstRay = i == 0 && ss.frameData->fb.frameID == 0;
    const uint32_t sampleIdx = uint32_t(ss.frameData->fb.frameID)
            * uint32_t(rendererParams.numIterations)
        + uint32_t(i);
    auto ray = makePrimaryRay(ss, sampleIdx, isVeryFirstRay);
    applyCuttingPlane(rendererParams.cutPlane, ray);
    float tmax = ray.t.upper;

    // Output accumulators. outputOpacity is the scalar coverage track driving
    // AOVs + framebuffer alpha; remainingT is the colored straight-through
    // transmittance driving radiance compositing + loop termination.
    vec3 outputColor(0.f);
    vec3 outputAlbedo(0.f);
    vec3 outputNormal(0.f);
    float outputOpacity = 0.f;
    vec3 remainingT(1.f);

    // First hit metadata (for picking)
    float depth = std::numeric_limits<float>::max();
    uint32_t primID = ~0u;
    uint32_t objID = ~0u;
    uint32_t instID = ~0u;

    constexpr float MIN_NORMAL_LENGTH = 1e-6f;
    const vec3 FALLBACK_NORMAL(0.f, 0.f, 0.f);

    // Transparency traversal loop. For T=0 (opaque), luminance(remainingT) ==
    // 1 - outputOpacity.
    while (luminance(remainingT) > 1.f - OPACITY_THRESHOLD) {
      ray.t.upper = tmax;

      SurfaceHit surfaceHit;
      surfaceHit.foundHit = false;
      intersectSurface(ss,
          ray,
          RayType::PRIMARY,
          &surfaceHit,
          primaryRayOptiXFlags(rendererParams));

      float hitDist = surfaceHit.foundHit ? surfaceHit.t : ray.t.upper;

      vec3 volumeColor(0.f);
      vec3 volumeNormal(0.f);
      float volumeOpacity = 0.f;
      uint32_t volObjID = ~0u;
      uint32_t volInstID = ~0u;

      float volumeDepth = rayMarchAllVolumes(ss,
          ray,
          RayType::PRIMARY,
          hitDist,
          rendererParams.inverseVolumeSamplingRate,
          volumeColor,
          volumeOpacity,
          volObjID,
          volInstID,
          &volumeNormal);

      bool volumeHit = volumeDepth < hitDist;
      if (volumeHit) {
        const float volumeNormalLength = glm::length(volumeNormal);
        const vec3 outputVolumeNormal = volumeNormalLength > MIN_NORMAL_LENGTH
            ? volumeNormal * (1.f / volumeNormalLength)
            : FALLBACK_NORMAL;
        outputColor += remainingT * volumeColor;
        accumulateValue(outputAlbedo, volumeColor, outputOpacity);
        accumulateNormal(outputNormal, outputVolumeNormal, outputOpacity);
        accumulateValue(outputOpacity, volumeOpacity, outputOpacity);
        remainingT *= (1.f - volumeOpacity);

        depth = volumeDepth;
        objID = volObjID;
        instID = volInstID;
        primID = volObjID;
      }

      if (surfaceHit.foundHit) {
        MaterialShadingState shadingState;
        materialInitShading(
            &shadingState, frameData, *surfaceHit.material, surfaceHit);

        const float alpha = materialEvaluateOpacity(shadingState);
        const vec3 transmission = materialEvaluateTransmission(shadingState);

        // Reflected + emitted radiance from this surface.
        const vec3 refl =
            ShadingPolicy::shadeSurface(shadingState, ss, ray, surfaceHit);

        outputColor += remainingT * (alpha * refl);
        // AOVs stay on the scalar coverage track (not the colored remainingT).
        accumulateValue(outputAlbedo,
            materialEvaluateTint(shadingState) * alpha,
            outputOpacity);
        const vec3 materialNormal = materialEvaluateNormal(shadingState);
        const float materialNormalLength = glm::length(materialNormal);
        const vec3 outputMaterialNormal =
            materialNormalLength > MIN_NORMAL_LENGTH
            ? materialNormal * (1.f / materialNormalLength)
            : FALLBACK_NORMAL;
        accumulateNormal(outputNormal, outputMaterialNormal, outputOpacity);
        accumulateValue(outputOpacity, alpha, outputOpacity);
        remainingT *= (1.f - alpha * (1.f - transmission));

        if (!volumeHit) {
          depth = surfaceHit.t;
          primID = surfaceHit.primID;
          objID = surfaceHit.objID;
          instID = surfaceHit.instID;
        }

        // Advance straight through to the next surface (no bending).
        ray.t.lower = surfaceHit.t + surfaceHit.epsilon;
      }

      if (isVeryFirstRay) {
        setPixelIds(frameData.fb, ss.pixel, depth, primID, objID, instID);
      }

      if (!surfaceHit.foundHit)
        break;
    }

    // HDRI background fills the remaining transmittance and marks the pixel
    // covered for the AOV/alpha track.
    if (vec3 hdri; getBackgroundLight(frameData, ray.dir, hdri)) {
      outputColor += remainingT * hdri;
      remainingT = vec3(0.f);
      accumulateValue(outputOpacity, 1.f, outputOpacity);
    }

    accumPixelSample(frameData,
        ss.pixel,
        vec4(outputColor, outputOpacity),
        outputAlbedo,
        outputNormal);
  }
}

} // namespace visrtx
