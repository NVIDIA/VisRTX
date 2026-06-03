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

// visrtx
#include "gpu/evalShading.h"
#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"
#include "gpu/intersectRay.h"
#include "gpu/renderer/common.h"
#include "gpu/renderer/raygen_helpers.h"
#include "gpu/renderer/shadowTransmittance.h"
#include "gpu/sampleLight.h"
#include "gpu/shadingState.h"
#include "gpu/shading_api.h"

// glm
#include <glm/common.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/geometric.hpp>
#include <glm/vector_relational.hpp>

// std
#include <cmath>
#include <limits>

namespace visrtx {

DECLARE_FRAME_DATA(frameData)

// AO occlusion from surface shadow transmittance (1 = fully blocked).
static VISRTX_DEVICE float surfaceShadowOcclusion(
    ScreenSample &ss, const Ray &r)
{
  return 1.0f - luminance(surfaceShadowTransmittance(ss, r));
}

// Interactive shading policy for templated rendering loop //////////////////

struct InteractiveShadingPolicy
{
  static VISRTX_DEVICE vec3 shadeSurface(
      const MaterialShadingState &shadingState,
      ScreenSample &ss,
      const Ray &ray,
      const SurfaceHit &hit)
  {
    const auto &rendererParams = frameData.renderer;
    const auto &interactiveParams = rendererParams.params.interactive;
    auto &world = frameData.world;

    // Ambient occlusion (uses vec3 surface shadow transmittance via adapter).
    const float aoFactor = interactiveParams.aoSamples > 0
        ? computeAO(ss,
              ray,
              hit,
              rendererParams.occlusionDistance,
              interactiveParams.aoSamples,
              &surfaceShadowOcclusion)
        : 1.f;

    vec3 contrib = materialEvaluateEmission(shadingState, -ray.dir);
    contrib += rendererParams.ambientColor * rendererParams.ambientIntensity
        * materialEvaluateTint(shadingState);

    const vec3 shadowOrigin = shadingHitpoint(hit) + hit.Ng * hit.epsilon;
    for (size_t i = 0; i < world.numLightInstances; i++) {
      const auto &light = world.lightInstances[i];
      const auto lightSample =
          sampleLight(ss, shadowOrigin, light.lightIndex, light.xfm);

      if (lightSample.pdf == 0.0f)
        continue;

      const Ray shadowRay = {
          shadowOrigin,
          lightSample.dir,
          {hit.epsilon, lightSample.dist},
      };

      // Surface shadows are tinted (vec3); volume shadows stay scalar.
      const vec3 surfaceTransmittance =
          surfaceShadowTransmittance(ss, shadowRay);
      const float volumeTransmittance =
          1.0f - volumeShadowOpacity(ss, shadowRay);
      const vec3 attenuation = surfaceTransmittance * volumeTransmittance;

      if (glm::all(
              glm::lessThanEqual(attenuation, vec3(MIN_CONTRIBUTION_EPSILON))))
        continue;

      vec3 thisLightContrib =
          materialShadeSurface(shadingState, hit, lightSample, -ray.dir);

      // Environment MIS (balance heuristic): the HDRI is the only light the
      // indirect bounce's escape can also reach, so combine the NEE and escape
      // estimators instead of summing them (which double-counted the env).
      // Interactive loops all lights with no pick, so pLight = envPdf (NO
      // 1/numLights). Non-env lights keep wNee = 1 (behaviour unchanged).
      if (frameData.registry.lights[light.lightIndex].type == LightType::HDRI) {
        const float pLight = envPdf(frameData, lightSample.dir);
        const float pBsdf =
            materialEvalPdf(shadingState, -ray.dir, lightSample.dir);
        thisLightContrib *= pLight / (pLight + pBsdf);
      }

      contrib += thisLightContrib * attenuation;
    }

    contrib *= aoFactor;

    // Single indirect bounce — REFLECTION only. Transmission/refraction is
    // owned by the flat compositing loop, so a through-surface continuation is
    // discarded here to avoid double-counting the transmitted background.
    NextRay nextRay = materialNextRay(shadingState, ray, ss.rs);
    if (!continuesThroughSurface(nextRay)
        && glm::any(glm::greaterThan(
            nextRay.contributionWeight, glm::vec3(MIN_CONTRIBUTION_EPSILON)))) {
      Ray bounceRay = {hit.hitpoint + hit.Ng * hit.epsilon,
          normalize(nextRay.direction)};

      SurfaceHit bounceHit;
      bounceHit.foundHit = false;
      intersectSurface(ss, bounceRay, RayType::PRIMARY, &bounceHit);

      if (bounceHit.foundHit) {
        MaterialShadingState bounceShadingState;
        materialInitShading(
            &bounceShadingState, frameData, *bounceHit.material, bounceHit);

        auto sampleDir = randomDir(ss.rs, bounceHit.Ns);
        auto cosineT = dot(bounceHit.Ns, sampleDir);
        auto color = materialEvaluateTint(bounceShadingState) * cosineT
            * rendererParams.ambientColor * rendererParams.ambientIntensity;
        contrib += color * nextRay.contributionWeight;
      } else {
        vec3 hdri;
        if (getBackgroundLight(frameData, bounceRay.dir, hdri)) {
          // Env MIS escape side: weight the BSDF-sampled escape by the same
          // balance heuristic as the NEE loop (pLight = envPdf, no 1/numLights).
          // A delta / through-surface lobe reports +inf => wBsdf = 1; here the
          // bounce is reflection-only so nextRay.pdf is finite.
          const float pLight = envPdf(frameData, bounceRay.dir);
          const float wBsdf = isinf(nextRay.pdf)
              ? 1.0f
              : nextRay.pdf / (nextRay.pdf + pLight);
          contrib += wBsdf * hdri * nextRay.contributionWeight;
        }
      }
    }

    return contrib;
  }
};

// OptiX programs /////////////////////////////////////////////////////////////

VISRTX_GLOBAL void __closesthit__shadow()
{
  // no-op
}

VISRTX_GLOBAL void __miss__shadow()
{
  // no-op
}

VISRTX_GLOBAL void __anyhit__shadow()
{
  auto &rendererParams = frameData.renderer.params;

  if (ray::isIntersectingSurfaces()) {
    ray::cullCutPlane();
    SurfaceHit hit;
    ray::populateSurfaceHit(hit);

    auto &transmittance = ray::rayData<vec3>();

    // Fully opaque material: skip the init/opacity callable chain.
    if (hit.material->isFullyOpaque) {
      transmittance = vec3(0.0f);
      optixTerminateRay();
      return;
    }

    MaterialShadingState shadingState;
    materialInitShading(&shadingState, frameData, *hit.material, hit);
    const float alpha = materialEvaluateOpacity(shadingState);
    const vec3 T = materialEvaluateTransmission(shadingState);

    transmittance *= (1.0f - alpha * (1.0f - T));

    if (glm::all(glm::lessThanEqual(transmittance, vec3(1.f - OPACITY_THRESHOLD))))
      optixTerminateRay();
    else
      optixIgnoreIntersection();
  } else {
    // Volume shadows are a separate trace with a scalar float payload
    // (volumeShadowOpacity); not interchangeable with the vec3 surface payload
    // above. See gpu/renderer/shadowTransmittance.h.
    auto &attenuation = ray::rayData<float>();
    VolumeHit hit;
    ray::populateVolumeHit(hit);
    rayMarchVolume(ray::screenSample(),
        hit,
        attenuation,
        rendererParams.interactive.inverseVolumeSamplingRateShadows);
    if (attenuation < OPACITY_THRESHOLD)
      optixIgnoreIntersection();
  }
}

VISRTX_GLOBAL void __anyhit__shading()
{
  ray::cullbackFaces();
  ray::cullCutPlane();
}

VISRTX_GLOBAL void __closesthit__shading()
{
  ray::populateHit();
}

VISRTX_GLOBAL void __miss__shading()
{
  if (ray::isIntersectingSurfaces()) {
    auto &hit = ray::rayData<SurfaceHit>();
    hit.foundHit = false;
  } else {
    auto &hit = ray::rayData<VolumeHit>();
    hit.foundHit = false;
  }
}

VISRTX_GLOBAL void __raygen__()
{
  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;

  renderPixel<InteractiveShadingPolicy>(frameData, ss);
}

} // namespace visrtx
