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
#include "gpu/renderer/shadowTransmittance.h"
#include "gpu/sampleLight.h"
#include "gpu/shadingState.h"
#include "gpu/volumeIntegration.h"

#include <limits>

namespace visrtx {

constexpr float PATH_CONTRIBUTION_EPSILON = 1.0e-8f;
constexpr float ATTENUATION_EPSILON = std::numeric_limits<float>::epsilon();
// RR start depth. Volume scatter engages earlier — throughput shrinks by
// medium albedo each scatter, so dense regions need RR sooner. Surface
// bounces don't shrink throughput so reliably; keep their conservative
// threshold.
constexpr int RUSSIAN_ROULETTE_START_DEPTH = 3;
constexpr int RUSSIAN_ROULETTE_START_DEPTH_VOLUME = 1;
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

VISRTX_DEVICE void accumPixelSample(
    const FrameGPUData &frame, const uvec2 &pixel, const SampleDetails &sample)
{
  accumPixelSample(frame,
      pixel,
      vec4(sample.color, sample.opacity),
      sample.albedo,
      sample.normal);
}

VISRTX_DEVICE vec3 evaluateOpacity(const MaterialShadingState &shadingState)
{
  return materialEvaluateOpacity(shadingState)
      * (1.0f - materialEvaluateTransmission(shadingState));
}

VISRTX_DEVICE bool shouldTerminatePath(ScreenSample &ss,
    int depth,
    vec3 &contribution,
    bool useRussianRoulette,
    int rrStartDepth = RUSSIAN_ROULETTE_START_DEPTH,
    float maxSurvivalProb = 0.95f)
{
  if (glm::all(glm::lessThan(contribution, vec3(PATH_CONTRIBUTION_EPSILON))))
    return true;

  if (!useRussianRoulette || depth < rrStartDepth)
    return false;

  // Survival cap. Surface default 0.95 keeps contributing paths alive
  // (bouncing is cheap). White-smoke / dense-cloud volumes keep
  // max(contribution) near 1 forever; the lower 0.5 cap forces probabilistic
  // termination so the warp doesn't pin on the longest lane.
  const float maxContribution =
      glm::max(contribution.x, glm::max(contribution.y, contribution.z));
  const float survivalProb = glm::min(maxSurvivalProb, maxContribution);
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
    // Fold the hemisphere-sample pdf cos(theta)/pi with the uniform light pick.
    const vec3 dir = sampleHemisphere(ss.rs, normal);
    const float cosNs = fmaxf(0.f, dot(dir, normal));
    return LightSample{
        rendererParams.ambientColor * rendererParams.ambientIntensity,
        dir,
        std::numeric_limits<float>::max(),
        lightPickPdf * cosNs * kInvPi,
    };
  } else {
    const auto &lightInstance = world.lightInstances[selectedIdx];
    auto ls =
        sampleLight(ss, origin, lightInstance.lightIndex, lightInstance.xfm);
    ls.pdf *= lightPickPdf;
    return ls;
  }
}

VISRTX_DEVICE LightSample sampleLightsVolume(
    ScreenSample &ss, const FrameGPUData &frameData, const vec3 &origin)
{
  const auto &world = frameData.world;
  const bool hasAmbientLight = frameData.renderer.ambientIntensity > 0.0f;
  const auto numLights = world.numLightInstances + hasAmbientLight;

  if (numLights == 0)
    return {};

  const size_t selectedIdx =
      glm::min(size_t((1.0f - curand_uniform(&ss.rs)) * float(numLights)),
          numLights - 1);

  const float lightPickPdf = 1.0f / float(numLights);

  if (selectedIdx == world.numLightInstances) {
    const auto &rendererParams = frameData.renderer;
    constexpr float INV_4PI = 1.0f / (4.0f * kPi);
    const vec3 dir = randomDir(ss.rs);
    return LightSample{
        rendererParams.ambientColor * rendererParams.ambientIntensity,
        dir,
        std::numeric_limits<float>::max(),
        lightPickPdf * INV_4PI,
    };
  } else {
    // Ambient sampled uniform-sphere (pdf 1/(4π)) to match the isotropic phase.
    const auto &lightInstance = world.lightInstances[selectedIdx];
    auto ls =
        sampleLight(ss, origin, lightInstance.lightIndex, lightInstance.xfm);
    ls.pdf *= lightPickPdf;
    return ls;
  }
}

VISRTX_DEVICE
VolumeDistanceSample sampleVolumeDistance(
    ScreenSample &ss, Ray ray, bool needNormal)
{
  VolumeDistanceSample volumeHit = {
      false, vec3(0.0f), ray.t.upper, vec3(0.0f), 0.0f, ~0u, ~0u};

  // Skip the gradient-based normal computation on non-primary bounces.
  volumeHit.depth = sampleDistanceAllVolumes(ss,
      ray,
      RayType::PRIMARY,
      ray.t.upper,
      volumeHit.albedo,
      volumeHit.extinction,
      volumeHit.didScatter,
      volumeHit.objID,
      volumeHit.instID,
      needNormal ? &volumeHit.normal : nullptr);
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

    // Fully opaque material: skip the init / opacity / transmission callable
    // dispatch chain and just block the ray.
    if (hit.material->isFullyOpaque) {
      attenuation = vec3(0.0f);
      optixTerminateRay();
      return;
    }

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

    // Unbiased ratio-tracking transmittance over this volume segment.
    // Scalar σ_t (TF is monochrome) broadcast to vec3 in the callee.
    ratioTrackTransmittanceVolume(ray::screenSample(), hit, attenuation);

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

      auto volumeSample = sampleVolumeDistance(ss, volumeRay, isFirstBounce);

      if (volumeSample.didScatter) {
        const vec3 scatterPos = ray.org + ray.dir * volumeSample.depth;

        {
          LightSample lightSample =
              sampleLightsVolume(ss, frameData, scatterPos);
          if (lightSample.pdf >= ATTENUATION_EPSILON
              && lightSample.dist > 0.0f) {
            constexpr float INV_4PI = 1.0f / (4.0f * kPi);
            const vec3 directLight = volumeSample.albedo * lightSample.radiance
                * INV_4PI / lightSample.pdf;
            const vec3 contribUpper = sampleContribution * directLight;
            const float maxContrib = glm::max(
                contribUpper.x, glm::max(contribUpper.y, contribUpper.z));
            // Pre-shadow skip: a contribution below SHADOW_SKIP_EPSILON
            // can't survive RGB quantisation even unattenuated. Costs nothing
            // to skip the trace entirely.
            constexpr float SHADOW_SKIP_EPSILON = 1.0e-5f;
            if (maxContrib >= SHADOW_SKIP_EPSILON) {
              const float eps = VOLUME_SCATTER_EPSILON;
              const Ray shadowRay = {
                  scatterPos + lightSample.dir * eps,
                  lightSample.dir,
                  {eps, lightSample.dist},
              };
              // Adaptive RR knob: w in (0, 1] = maxContrib / 0.5. Dim rays
              // raise the in-trace RR threshold so ratio-tracking kills them
              // sooner. RR estimator stays unbiased; cap inside RR bounds
              // amplification.
              ss.shadowContribWeight = glm::min(1.0f, maxContrib * 2.0f);
              const auto attenuation = surfaceShadowTransmittance(ss, shadowRay)
                  * volumeShadowTransmittance(ss, shadowRay);
              ss.shadowContribWeight = 1.0f;
              sample.color += contribUpper * attenuation;
            }
          }
        }

        accumulateValue(sample.opacity, 1.0f, sample.opacity);
        sampleContribution *= volumeSample.albedo;
        if (shouldTerminatePath(ss,
                d,
                sampleContribution,
                true,
                RUSSIAN_ROULETTE_START_DEPTH_VOLUME,
                /*maxSurvivalProb=*/0.5f))
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
        // Sample around the shading normal so the cosine-weighted hemisphere's
        // pdf matches the BRDF's NdotL (which uses Ns). Sampling around Ng
        // would bias the Lambertian estimator by cos_Ns/cos_Ng on smooth or
        // bump-mapped surfaces.
        const vec3 shadowOrigin =
            shadingHitpoint(surfaceHit) + surfaceHit.Ng * surfaceHit.epsilon;
        LightSample lightSample =
            sampleLights(ss, frameData, shadowOrigin, surfaceHit.Ns);
        if (lightSample.pdf >= ATTENUATION_EPSILON && lightSample.dist > 0.0f) {
          // Gate on the shading normal so the terminator follows the smooth
          // surface; gating on Ng would carve the per-triangle facet shape
          // into the lit/unlit boundary at grazing light angles.
          const float lightDotNs = dot(lightSample.dir, surfaceHit.Ns);
          if (lightDotNs > 0.0f) {
            const vec3 directLight = materialShadeSurface(
                shadingState, surfaceHit, lightSample, -ray.dir);
            const vec3 contribUpper =
                sampleContribution * materialOpacity * directLight;
            const float maxContrib = glm::max(
                contribUpper.x, glm::max(contribUpper.y, contribUpper.z));
            constexpr float SHADOW_SKIP_EPSILON = 1.0e-5f;
            if (maxContrib >= SHADOW_SKIP_EPSILON) {
              const Ray shadowRay = {
                  shadowOrigin,
                  lightSample.dir,
                  {surfaceHit.epsilon, lightSample.dist},
              };
              ss.shadowContribWeight = glm::min(1.0f, maxContrib * 2.0f);
              const auto attenuation = surfaceShadowTransmittance(ss, shadowRay)
                  * volumeShadowTransmittance(ss, shadowRay);
              ss.shadowContribWeight = 1.0f;
              sample.color += contribUpper * attenuation;
            }
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
        // Primary-ray HDRI = rendered background. Bounce misses skip it:
        // NEE at the previous vertex already sampled HDRI as a light, so
        // adding it on miss double-counts for non-Dirac BSDFs (almost all
        // scivis materials). Dirac mirrors / specular transmission lose
        // HDRI under this rule — no MIS yet to recover them.
        if (isFirstBounce) {
          if (vec3 hdri; getBackgroundLight(frameData, ray.dir, hdri)) {
            sample.color += sampleContribution * hdri;
            accumulateValue(sample.opacity, 1.f, sample.opacity);
          }
          setPixelIds(frameData.fb, ss.pixel, ray.t.upper, ~0u, ~0u, ~0u);
        }

        break;
      }
    }

    accumPixelSample(frameData, ss.pixel, sample);
  }
}

} // namespace visrtx
