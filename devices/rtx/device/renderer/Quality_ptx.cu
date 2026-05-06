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

#include <curand_mtgp32_kernel.h>
#include <optix_device.h>
#include "gpu/createScreenSample.h"
#include "gpu/evalShading.h"
#include "gpu/gpu_debug.h"
#include "gpu/gpu_decl.h"
#include "gpu/gpu_math.h"
#include "gpu/gpu_util.h"
#include "gpu/intersectRay.h"
#include "gpu/populateHit.h"
#include "gpu/renderer/common.h"
#include "gpu/sampleLight.h"
#include "gpu/shadingState.h"
#include "gpu/volumeIntegration.h"

#include <limits>

namespace visrtx {

constexpr float PATH_CONTRIBUTION_EPSILON = 1.0e-8f;
constexpr float ATTENUATION_EPSILON = std::numeric_limits<float>::epsilon();
constexpr int RUSSIAN_ROULETTE_START_DEPTH = 3;
constexpr float VOLUME_SCATTER_EPSILON = 1.0e-4f;

DECLARE_FRAME_DATA(frameData)

struct VolumeDistanceSample
{
  bool didScatter;
  vec3 albedo;
  float depth;
  vec3 normal;
  float extinction;
  uint32_t objID;
  uint32_t instID;
};

struct SampleDetails
{
  vec3 color;
  float opacity;
  vec3 albedo;
  float depth;
  vec3 normal;
};

VISRTX_DEVICE void accumPixelSample(const FrameGPUData &frame,
    const uvec2 &pixel,
    const SampleDetails &sample)
{
  accumPixelSample(frame,
      pixel,
      vec4(sample.color, sample.opacity),
      sample.albedo,
      sample.normal);
}

VISRTX_DEVICE vec3 surfaceAttenuation(ScreenSample &ss, Ray r)
{
  vec3 attenuation = vec3(1.0f);
  intersectSurface(
      ss, r, RayType::SHADOW, &attenuation, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return attenuation;
}

VISRTX_DEVICE vec3 volumeAttenuation(ScreenSample &ss, Ray r)
{
  vec3 attenuation = vec3(1.0f);
  intersectVolume(
      ss, r, RayType::SHADOW, &attenuation, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return attenuation;
}

VISRTX_DEVICE vec3 evaluateOpacity(const MaterialShadingState &shadingState)
{
  return materialEvaluateOpacity(shadingState)
      * (1.0f - materialEvaluateTransmission(shadingState));
}

VISRTX_DEVICE bool shouldTerminatePath(
    ScreenSample &ss, int depth, vec3 &contribution, bool useRussianRoulette)
{
  if (glm::all(glm::lessThan(contribution, vec3(PATH_CONTRIBUTION_EPSILON))))
    return true;

  if (!useRussianRoulette || depth < RUSSIAN_ROULETTE_START_DEPTH)
    return false;

  const float maxContribution =
      glm::max(contribution.x, glm::max(contribution.y, contribution.z));
  const float survivalProb = glm::min(0.95f, maxContribution);
  if (curand_uniform(&ss.rs) > survivalProb)
    return true;

  contribution /= survivalProb;
  return false;
}

VISRTX_DEVICE LightSample sampleLights(ScreenSample &ss,
    const FrameGPUData &frameData,
    const vec3 &origin,
    const vec3 &normal)
{
  const auto &world = frameData.world;
  bool hasAmbientLight = frameData.renderer.ambientIntensity > 0.0f;
  auto numLights = world.numLightInstances + hasAmbientLight;

  if (numLights == 0)
    return {};

  // curand_uniform returns (0,1], invert to get [0,numLights).
  // Clamp to handle float rounding when curand returns a subnormal.
  const size_t selectedIdx =
      glm::min(size_t((1.0f - curand_uniform(&ss.rs)) * float(numLights)),
          numLights - 1);

  // Uniform light pick: P(light) = 1/numLights. Fold that into the returned
  // pdf rather than into radiance so MIS weights see the full joint pdf
  // P(dir, light) = P(dir | light) * (1/numLights).
  const float lightPickPdf = 1.0f / float(numLights);

  // last index is reserved for ambient light if it exists
  if (selectedIdx == world.numLightInstances) {
    const auto &rendererParams = frameData.renderer;
    return LightSample{
        rendererParams.ambientColor * rendererParams.ambientIntensity,
        sampleHemisphere(ss.rs, normal),
        std::numeric_limits<float>::max(),
        lightPickPdf / (2.0f * float(M_PI)),
    };
  } else {
    const auto &lightInstance = world.lightInstances[selectedIdx];
    auto ls =
        sampleLight(ss, origin, lightInstance.lightIndex, lightInstance.xfm);
    ls.pdf *= lightPickPdf;
    return ls;
  }
}

VISRTX_DEVICE
VolumeDistanceSample sampleVolumeDistance(ScreenSample &ss, Ray ray)
{
  VolumeDistanceSample volumeHit = {
      false, vec3(0.0f), ray.t.upper, vec3(0.0f), 0.0f, ~0u, ~0u};

  volumeHit.depth = sampleDistanceAllVolumes(ss,
      ray,
      RayType::PRIMARY,
      ray.t.upper,
      volumeHit.albedo,
      volumeHit.extinction,
      volumeHit.didScatter,
      volumeHit.objID,
      volumeHit.instID,
      &volumeHit.normal);
  return volumeHit;
}

VISRTX_GLOBAL void __closesthit__shading()
{
  ray::populateHit();
}

VISRTX_GLOBAL void __anyhit__shading()
{
  ray::cullCutPlane();
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

VISRTX_GLOBAL void __closesthit__shadow() {}

VISRTX_GLOBAL void __anyhit__shadow()
{
  auto &attenuation = ray::rayData<vec3>();

  if (ray::isIntersectingSurfaces()) {
    ray::cullCutPlane();
    SurfaceHit hit;
    ray::populateSurfaceHit(hit);

    auto ss = ray::screenSample();
    MaterialShadingState shadingState;
    materialInitShading(&shadingState, frameData, *hit.material, hit);
    auto opacity = evaluateOpacity(shadingState);

    attenuation *= (1.0f - opacity);

    if (glm::all(glm::lessThanEqual(attenuation, vec3(ATTENUATION_EPSILON))))
      optixTerminateRay();
    else
      optixIgnoreIntersection();
  } else {
    VolumeHit hit;
    ray::populateVolumeHit(hit);

    vec3 albedo = vec3(0.0f);
    float sampledExtinction = 0.0f;
    bool sampledDidScatter = false;
    sampleDistanceVolume(
        ray::screenSample(), hit, albedo, sampledExtinction, sampledDidScatter);

    if (sampledDidScatter)
      attenuation *= albedo;

    if (glm::all(glm::lessThanEqual(attenuation, vec3(ATTENUATION_EPSILON))))
      optixTerminateRay();
    else
      optixIgnoreIntersection();
  }
}

VISRTX_GLOBAL void __miss__shadow() {}

VISRTX_GLOBAL void __raygen__()
{
  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;

  const auto &rendererParams = frameData.renderer;
  const auto &qualityParams = rendererParams.params.quality;

  for (int i = 0; i < rendererParams.numIterations; ++i) {
    bool isVeryFirstRay = i == 0 && ss.frameData->fb.frameID == 0;
    auto ray = makePrimaryRay(ss, isVeryFirstRay);

    applyCuttingPlane(rendererParams.cutPlane, ray);

    SampleDetails sample = {
        vec3(0.0f), 0.0f, vec3(0.0f), ray.t.upper, vec3(0.0f)};

    auto sampleContribution = vec3(1.0f);

    for (int d = 0; d < qualityParams.maxRayDepth; ++d) {
      const bool isFirstBounce = d == 0;

      SurfaceHit surfaceHit = {};
      intersectSurface(ss,
          ray,
          RayType::PRIMARY,
          &surfaceHit,
          primaryRayOptiXFlags(rendererParams));

      float volumeUpperBound = surfaceHit.foundHit ? surfaceHit.t : ray.t.upper;
      auto volumeRay = Ray{ray.org, ray.dir, {ray.t.lower, volumeUpperBound}};

      auto volumeSample = sampleVolumeDistance(ss, volumeRay);

      if (volumeSample.didScatter) {
        const vec3 scatterPos = ray.org + ray.dir * volumeSample.depth;

        {
          LightSample lightSample =
              sampleLights(ss, frameData, scatterPos, volumeSample.normal);
          if (lightSample.pdf >= ATTENUATION_EPSILON
              && lightSample.dist > 0.0f) {
            const float eps = VOLUME_SCATTER_EPSILON;
            const Ray shadowRay = {
                scatterPos + lightSample.dir * eps,
                lightSample.dir,
                {eps, lightSample.dist},
            };
            const auto attenuation = surfaceAttenuation(ss, shadowRay)
                * volumeAttenuation(ss, shadowRay);

            constexpr float INV_4PI = 1.0f / (4.0f * float(M_PI));
            const vec3 directLight = volumeSample.albedo * lightSample.radiance
                * INV_4PI / lightSample.pdf;
            sample.color += sampleContribution * directLight * attenuation;
          }
        }

        accumulateValue(sample.opacity, 1.0f, sample.opacity);
        sampleContribution *= volumeSample.albedo;
        if (shouldTerminatePath(ss, d, sampleContribution, true))
          break;

        if (isFirstBounce) {
          setPixelIds(frameData.fb,
              ss.pixel,
              volumeSample.depth,
              volumeSample.objID,
              volumeSample.objID,
              volumeSample.instID);
          sample.depth = volumeSample.depth;
          sample.albedo = volumeSample.albedo;
          const vec3 volumeNormal = glm::length(volumeSample.normal) > 0.01f
              ? volumeSample.normal
              : -ray.dir;
          sample.normal = volumeNormal;
        }

        const vec3 scatterDir = randomDir(ss.rs);
        ray = Ray{scatterPos + scatterDir * VOLUME_SCATTER_EPSILON, scatterDir};
        continue;
      }

      if (surfaceHit.foundHit) {
        MaterialShadingState shadingState;
        materialInitShading(
            &shadingState, frameData, *surfaceHit.material, surfaceHit);

        const vec3 materialEmission =
            materialEvaluateEmission(shadingState, -ray.dir);
        const vec3 materialTint = materialEvaluateTint(shadingState);
        const float materialOpacity = materialEvaluateOpacity(shadingState);

        if (isFirstBounce) {
          setPixelIds(frameData.fb,
              ss.pixel,
              surfaceHit.t,
              surfaceHit.primID,
              surfaceHit.objID,
              surfaceHit.instID);
          sample.depth = surfaceHit.t;
          sample.normal = materialEvaluateNormal(shadingState);
          sample.albedo = materialTint;
        }

        sample.color += sampleContribution * materialEmission * materialOpacity;
        LightSample lightSample =
            sampleLights(ss, frameData, shadowOrigin, surfaceHit.Ns);
        if (lightSample.pdf >= ATTENUATION_EPSILON && lightSample.dist > 0.0f) {
          // Gate on the shading normal so the terminator follows the smooth
          // surface; gating on Ng would carve the per-triangle facet shape
          // into the lit/unlit boundary at grazing light angles.
          const float lightDotNs = dot(lightSample.dir, surfaceHit.Ns);
          if (lightDotNs > 0.0f) {
            const Ray shadowRay = {
                surfaceHit.hitpoint + surfaceHit.Ng * surfaceHit.epsilon,
                lightSample.dir,
                {surfaceHit.epsilon, lightSample.dist},
            };
            const auto attenuation = surfaceAttenuation(ss, shadowRay)
                * volumeAttenuation(ss, shadowRay);
            const vec3 directLight = materialShadeSurface(
                shadingState, surfaceHit, lightSample, -ray.dir);

            sample.color += sampleContribution * materialOpacity * directLight
                * attenuation;
          }
        }

        auto nextRay = materialNextRay(shadingState, ray, ss.rs);
        sampleContribution *= nextRay.contributionWeight;

        if (!continuesThroughSurface(nextRay))
          accumulateValue(sample.opacity, 1.0f, sample.opacity);

        if (shouldTerminatePath(ss, d, sampleContribution, true))
          break;

        const float side = continuesThroughSurface(nextRay) ? -1.0f : 1.0f;
        ray =
            Ray{surfaceHit.hitpoint + surfaceHit.Ng * surfaceHit.epsilon * side,
                normalize(vec3(nextRay.direction))};
      }

      if (!surfaceHit.foundHit && !volumeSample.didScatter) {
        if (vec3 hdri; getBackgroundLight(frameData, ray.dir, hdri)) {
          sample.color += sampleContribution * hdri;
          accumulateValue(sample.opacity, 1.f, sample.opacity);
        }

        if (isFirstBounce) {
          setPixelIds(frameData.fb, ss.pixel, ray.t.upper, ~0u, ~0u, ~0u);
        }

        break;
      }
    }

    accumPixelSample(frameData, ss.pixel, sample);
  }
}

} // namespace visrtx
