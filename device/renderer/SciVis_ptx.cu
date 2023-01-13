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
  PRIMARY = 0,
  SHADOW = 1
};

struct RayAttenuation
{
  const Ray *ray{nullptr};
  float attenuation{0.f};
};

DECLARE_FRAME_DATA(frameData)

// Helper functions ///////////////////////////////////////////////////////////

RT_FUNCTION bool isOccluded(ScreenSample &ss, Ray r)
{
  uint32_t o = 0;
  intersectSurface(
      ss, r, RayType::SHADOW, &o, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return static_cast<bool>(o);
}

RT_FUNCTION float computeAO(
    ScreenSample &ss, const Ray &primaryRay, const Hit &currentHit)
{
  int hits = 0;

  const int numSamples = frameData.renderer.params.scivis.aoSamples;
  for (int i = 0; i < numSamples; i++) {
    Ray aoRay;
    aoRay.org = currentHit.hitpoint + (currentHit.epsilon * currentHit.Ng);
    aoRay.dir = randomDir(ss.rs, currentHit.Ns);
    if (isOccluded(ss, aoRay))
      hits++;
  }

  return 1.f - hits / float(numSamples);
}

RT_FUNCTION float attenuation(ScreenSample &ss, Ray r)
{
  RayAttenuation ra;
  ra.ray = &r;
  intersectVolume(
      ss, r, RayType::SHADOW, &ra, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return ra.attenuation;
}

RT_FUNCTION vec3 computeLightConrib(ScreenSample &ss, const SurfaceHit &hit)
{
  const auto &rendererParams = frameData.renderer.params.scivis;

  auto &world = frameData.world;

  const vec3 shadePoint = hit.hitpoint + (hit.epsilon * hit.Ng);

  vec3 contrib(0.f);
  for (size_t i = 0; i < world.numLightInstances; i++) {
    auto *inst = world.lightInstances + i;
    if (!inst)
      continue;

    for (size_t l = 0; l < inst->numLights; l++) {
      auto ls = sampleLight(ss, hit, inst->indices[l]);
      Ray r;
      r.org = shadePoint;
      r.dir = ls.dir;
      r.t.upper = ls.dist;
      if (!isOccluded(ss, r)) {
        contrib += ls.radiance * dot(ls.dir, hit.Ns)
            * rendererParams.lightFalloff * (1.f - attenuation(ss, r));
      }
    }
  }
  return contrib;
}

// OptiX programs /////////////////////////////////////////////////////////////

RT_PROGRAM void __closesthit__shadow()
{
  // no-op
}

RT_PROGRAM void __anyhit__shadow()
{
  if (ray::isIntersectingSurfaces()) {
    SurfaceHit hit;
    ray::populateSurfaceHit(hit);
    const auto &material = *hit.material;
    const auto mat_baseColor =
        getMaterialParameter(frameData, material.baseColor, hit);
    const auto mat_opacity =
        getMaterialParameter(frameData, material.opacity, hit);
    if (mat_opacity >= 0.99f) {
      auto &occluded = ray::rayData<uint32_t>();
      occluded = true;
      optixTerminateRay();
    } else
      optixIgnoreIntersection();
  } else {
    auto &ra = ray::rayData<RayAttenuation>();
    VolumeHit hit;
    ray::populateVolumeHit(hit);
    rayMarchVolume(ray::screenSample(), hit, ra.attenuation);
    if (ra.attenuation < 0.99f)
      optixIgnoreIntersection();
  }
}

RT_PROGRAM void __closesthit__primary()
{
  ray::populateHit();
}

RT_PROGRAM void __miss__()
{
  // TODO
}

RT_PROGRAM void __raygen__()
{
  const auto &rendererParams = frameData.renderer.params.scivis;

  /////////////////////////////////////////////////////////////////////////////
  // TODO: clean this up! need to split out Ray/RNG, don't need screen samples
  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;
  auto ray = makePrimaryRay(ss);
  float tmax = ray.t.upper;
  /////////////////////////////////////////////////////////////////////////////

  SurfaceHit surfaceHit;
  VolumeHit volumeHit;
  vec3 outputColor(0.f);
  vec3 outputNormal = ray.dir;
  float outputOpacity = 0.f;
  float depth = 1e30f;
  bool firstHit = true;

  while (outputOpacity < 0.99f) {
    ray.t.upper = tmax;
    surfaceHit.foundHit = false;
    intersectSurface(ss, ray, RayType::PRIMARY, &surfaceHit);

    vec3 color(0.f);
    float opacity = 0.f;

    if (surfaceHit.foundHit) {
      depth = min(depth,
          rayMarchAllVolumes(
              ss, ray, RayType::PRIMARY, surfaceHit.t, color, opacity));

      if (firstHit) {
        outputNormal = surfaceHit.Ng;
        depth = min(depth, surfaceHit.t);
        firstHit = false;
      }

      const auto &material = *surfaceHit.material;
      const auto mat_baseColor =
          getMaterialParameter(frameData, material.baseColor, surfaceHit);
      const auto mat_opacity =
          getMaterialParameter(frameData, material.opacity, surfaceHit);

      const float aoFactor = rendererParams.aoSamples > 0
          ? computeAO(ss, ray, surfaceHit) * rendererParams.aoIntensity
          : 0.f;

      accumulateValue(color,
          (mat_baseColor * computeLightConrib(ss, surfaceHit))
              + (rendererParams.aoColor * aoFactor
                  * rendererParams.lightFalloff),
          opacity);
      accumulateValue(opacity, mat_opacity, opacity);

      color *= opacity;
      accumulateValue(outputColor, color, outputOpacity);
      accumulateValue(outputOpacity, opacity, outputOpacity);

      ray.t.lower = surfaceHit.t + surfaceHit.epsilon;
    } else {
      const auto bgColor = vec3(frameData.renderer.bgColor);
      const auto bgOpacity = frameData.renderer.bgColor.w;

      const float volumeDepth = rayMarchAllVolumes(
          ss, ray, RayType::PRIMARY, ray.t.upper, color, opacity);

      if (firstHit)
        depth = min(depth, volumeDepth);

      color *= opacity;
      accumulateValue(color, bgColor, opacity);
      accumulateValue(opacity, bgOpacity, opacity);
      accumulateValue(outputColor, color, outputOpacity);
      accumulateValue(outputOpacity, opacity, outputOpacity);
      break;
    }
  }

  accumResults(frameData.fb,
      ss.pixel,
      vec4(outputColor, outputOpacity),
      depth,
      outputColor,
      outputNormal);
}

} // namespace visrtx
