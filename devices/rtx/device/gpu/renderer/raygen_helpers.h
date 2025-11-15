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
#include "gpu/intersectRay.h"
#include "gpu/renderer/common.h"
#include "gpu/shadingState.h"

// Shared helper functions for ray generation across renderers

namespace visrtx {

// Compute surface attenuation for shadow/occlusion rays
// Uses the standard SHADOW ray type for consistency across all renderers
VISRTX_DEVICE float surfaceAttenuation(ScreenSample &ss, const Ray &r)
{
  float a = 0.0f;
  intersectSurface(
      ss, r, RayType::SHADOW, &a, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return a;
}

// Compute volume attenuation for shadow rays
// Uses the standard SHADOW ray type for consistency across all renderers
VISRTX_DEVICE float volumeAttenuation(ScreenSample &ss, const Ray &r)
{
  float attenuation = 0.0f;
  intersectVolume(
      ss, r, RayType::SHADOW, &attenuation, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return attenuation;
}

// Evaluate opacity including transmission
VISRTX_DEVICE float evaluateOpacity(const MaterialShadingState &shadingState)
{
  return materialEvaluateOpacity(shadingState)
      * (1.0f - glm::luminosity(materialEvaluateTransmission(shadingState)));
}

// Templated rendering loop
// ShadingPolicy must implement:
//   static VISRTX_DEVICE vec4 shadeSurface(
//       const MaterialShadingState &shadingState,
//       ScreenSample &ss,
//       const Ray &ray,
//       const SurfaceHit &hit)
template <typename ShadingPolicy>
VISRTX_DEVICE void renderPixel(FrameGPUData &frameData, ScreenSample ss)
{
  auto &rendererParams = frameData.renderer;

  for (int i = 0; i < frameData.renderer.numIterations; i++) {
    // First go with the main ray, pixel centered if first frame.
    // Jittered samples are produced by next iterations.
    bool isVeryFirstRay = i == 0 && ss.frameData->fb.frameID == 0;
    auto ray = makePrimaryRay(ss, isVeryFirstRay);
    float tmax = ray.t.upper;

    // Output accumulators
    vec3 outputColor(0.f);
    vec3 outputAlbedo(0.f);
    vec3 outputNormal(0.f);
    float outputOpacity = 0.f;

    // First hit metadata (for picking)
    float depth = std::numeric_limits<float>::max();
    uint32_t primID = ~0u;
    uint32_t objID = ~0u;
    uint32_t instID = ~0u;

    // Transparency traversal loop
    while (outputOpacity < OPACITY_THRESHOLD) {
      ray.t.upper = tmax;

      // Find next surface
      SurfaceHit surfaceHit;
      surfaceHit.foundHit = false;
      intersectSurface(ss,
          ray,
          RayType::PRIMARY,
          &surfaceHit,
          primaryRayOptiXFlags(rendererParams));

      float hitDist = surfaceHit.foundHit ? surfaceHit.t : ray.t.upper;

      // Ray march volumes up to this surface hit
      vec3 volumeColor(0.f);
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
          volInstID);

      // Accumulate volume contribution if any
      if (volumeDepth < depth) {
        accumulateValue(outputColor, volumeColor, outputOpacity);
        accumulateValue(outputAlbedo, volumeColor, outputOpacity);
        accumulateNormal(outputNormal, -ray.dir, outputOpacity);
        accumulateValue(outputOpacity, volumeOpacity, outputOpacity);

        depth = volumeDepth;
        objID = volObjID;
        instID = volInstID;
        primID = volObjID;
      }

      // Handle surface hit if any
      if (surfaceHit.foundHit) {
        MaterialShadingState shadingState;
        materialInitShading(
            &shadingState, frameData, *surfaceHit.material, surfaceHit);

        // Call the renderer-specific shading function
        const vec4 surfaceColor =
            ShadingPolicy::shadeSurface(shadingState, ss, ray, surfaceHit);

        // Accumulate surface contribution
        accumulateValue(
            outputColor, vec3(surfaceColor) * surfaceColor.a, outputOpacity);
        accumulateValue(outputAlbedo,
            materialEvaluateTint(shadingState) * surfaceColor.a,
            outputOpacity);
        accumulateNormal(
            outputNormal, materialEvaluateNormal(shadingState), outputOpacity);
        accumulateValue(outputOpacity, surfaceColor.a, outputOpacity);

        // Track first hit from surface
        if (surfaceHit.t < depth) {
          depth = surfaceHit.t;
          primID = surfaceHit.primID;
          objID = surfaceHit.objID;
          instID = surfaceHit.instID;
        }

        // Advance ray past this surface for next iteration
        ray.t.lower = surfaceHit.t + surfaceHit.epsilon;
      }

      // Record first hit metadata
      if (isVeryFirstRay) {
        setPixelIds(frameData.fb, ss.pixel, depth, primID, objID, instID);
      }

      // Exit if the current ray left the scene
      if (!surfaceHit.foundHit)
        break;

      // Otherwise, continue through transparent surface
    }

    // Accumulate background for remaining transparency
    const auto bg = getBackground(frameData, ss.screen, ray.dir);
    const bool premultiplyBg = rendererParams.premultiplyBackground;
    vec3 bgColor = premultiplyBg ? vec3(bg) * bg.a : vec3(bg);

    accumulateValue(outputColor, bgColor, outputOpacity);
    accumulateValue(outputOpacity, bg.a, outputOpacity);

    // Write accumulated sample to framebuffer
    accumPixelSample(frameData,
        ss.pixel,
        vec4(outputColor, outputOpacity),
        outputAlbedo,
        outputNormal,
        i);
  }
}

} // namespace visrtx
