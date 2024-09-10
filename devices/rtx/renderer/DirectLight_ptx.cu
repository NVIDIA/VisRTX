/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

VISRTX_DEVICE float volumeAttenuation(ScreenSample &ss, Ray r)
{
  RayAttenuation ra;
  ra.ray = &r;
  intersectVolume(
      ss, r, RayType::SHADOW, &ra, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return ra.attenuation;
}

VISRTX_DEVICE vec4 shadeSurface(ScreenSample &ss, Ray &ray, const SurfaceHit &hit)
{
  const auto &rendererParams = frameData.renderer;
  const auto &directLightParams = rendererParams.params.directLight;

  auto &world = frameData.world;

  const vec3 shadePoint = hit.hitpoint + (hit.epsilon * hit.Ns);

  // Compute ambient light contribution //

  const float aoFactor = directLightParams.aoSamples > 0
      ? computeAO(ss,
            ray,
            RayType::SHADOW,
            hit,
            rendererParams.occlusionDistance,
            directLightParams.aoSamples)
      : 1.f;
  const vec4 matAoResult = evalMaterial(frameData,
      *hit.material,
      hit,
      -ray.dir,
      -ray.dir,
      rendererParams.ambientIntensity * rendererParams.ambientColor);

  // Compute contribution from other lights //

  vec3 contrib = vec3(matAoResult) * (aoFactor * float(M_PI));
  float opacity = matAoResult.w;
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
      const float surface_o = 1.f - surfaceAttenuation(ss, r, RayType::SHADOW);
      const float volume_o = 1.f - volumeAttenuation(ss, r);
      const float attenuation = surface_o * volume_o;
      const vec4 matResult = evalMaterial(
          frameData, *hit.material, hit, -ray.dir, ls.dir, ls.radiance);
      contrib += vec3(matResult) * dot(ls.dir, hit.Ns)
          * directLightParams.lightFalloff * attenuation;
    }
  }
  return {contrib, opacity};
}

// OptiX programs /////////////////////////////////////////////////////////////

VISRTX_GLOBAL void __closesthit__shadow()
{
  // no-op
}

VISRTX_GLOBAL void __anyhit__shadow()
{
  if (ray::isIntersectingSurfaces()) {
    SurfaceHit hit;
    ray::populateSurfaceHit(hit);

    const auto &fd = frameData;
    const auto &md = *hit.material;
    vec4 color = getMaterialParameter(fd, md.values[MV_BASE_COLOR], hit);
    float opacity = getMaterialParameter(fd, md.values[MV_OPACITY], hit).x;
    opacity = adjustedMaterialOpacity(opacity, md) * color.w;

    auto &o = ray::rayData<float>();
    accumulateValue(o, opacity, o);
    if (o >= 0.99f)
      optixTerminateRay();
    else
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
  // TODO
}

VISRTX_GLOBAL void __raygen__()
{
  auto &rendererParams = frameData.renderer;

  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;
  auto ray = makePrimaryRay(ss);
  float tmax = ray.t.upper;

  SurfaceHit surfaceHit;
  VolumeHit volumeHit;
  vec3 outputColor(0.f);
  vec3 outputNormal = ray.dir;
  float outputOpacity = 0.f;
  float depth = 1e30f;
  uint32_t primID = ~0u;
  uint32_t objID = ~0u;
  uint32_t instID = ~0u;
  bool firstHit = true;

  while (outputOpacity < 0.99f) {
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
          primID = surfaceHit.primID;
          objID = surfaceHit.objID;
          instID = surfaceHit.instID;
        }
        firstHit = false;
      }

      const vec4 shadingResult = shadeSurface(ss, ray, surfaceHit);
      accumulateValue(color, vec3(shadingResult), opacity);
      accumulateValue(opacity, shadingResult.w, opacity);

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

      const auto bg = getBackground(frameData.renderer, ss.screen);
      accumulateValue(color, vec3(bg), opacity);
      accumulateValue(opacity, bg.w, opacity);
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
      outputNormal,
      primID,
      objID,
      instID);
}

} // namespace visrtx
