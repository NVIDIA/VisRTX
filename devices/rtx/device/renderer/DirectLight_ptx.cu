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

#include <cmath>
#include <glm/common.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/geometric.hpp>
#include <glm/vector_relational.hpp>
#include <limits>
#include "gpu/evalShading.h"
#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"
#include "gpu/intersectRay.h"
#include "gpu/sampleLight.h"
#include "gpu/shadingState.h"
#include "gpu/shading_api.h"
#include "gpu/renderer/common.h"
#include "gpu/renderer/raygen_helpers.h"

namespace visrtx {

DECLARE_FRAME_DATA(frameData)

// DirectLight shading policy for templated rendering loop //////////////////

struct DirectLightShadingPolicy
{
  static VISRTX_DEVICE vec4 shadeSurface(const MaterialShadingState &shadingState,
    ScreenSample &ss,
    const Ray &ray,
    const SurfaceHit &hit)
{
  const auto &rendererParams = frameData.renderer;
  const auto &directLightParams = rendererParams.params.directLight;

  auto &world = frameData.world;

  // Compute ambient light contribution //
  const float aoFactor = directLightParams.aoSamples > 0
      ? computeAO(ss,
            ray,
            hit,
            rendererParams.occlusionDistance,
            directLightParams.aoSamples,
            &surfaceAttenuation)
      : 1.f;

  vec3 contrib = materialEvaluateEmission(shadingState, -ray.dir);

  // Handle ambient light contribution
  if (rendererParams.ambientIntensity > 0.0f) {
    contrib += rendererParams.ambientColor * rendererParams.ambientIntensity
        * materialEvaluateTint(shadingState);
  }

  // Handle all lights contributions
  for (size_t i = 0; i < world.numLightInstances; i++) {
    auto *inst = world.lightInstances + i;
    if (!inst)
      continue;

    for (size_t l = 0; l < inst->numLights; l++) {
      const auto lightSample =
          sampleLight(ss, hit, inst->indices[l], inst->xfm);
      if (lightSample.pdf == 0.0f)
        continue;

      // Shadowing
      const Ray shadowRay = {
          hit.hitpoint + hit.Ng * hit.epsilon,
          lightSample.dir,
          {hit.epsilon, lightSample.dist},
      };

      const float surface_attenuation =
          1.0f - surfaceAttenuation(ss, shadowRay);
      const float volume_attenuation = 1.0f - volumeAttenuation(ss, shadowRay);
      const float attenuation = surface_attenuation * volume_attenuation;

      // Complete occlusion?
      if (attenuation <= MIN_CONTRIBUTION_EPSILON)
        continue;

      const vec3 thisLightContrib =
          materialShadeSurface(shadingState, hit, lightSample, -ray.dir);

      contrib += thisLightContrib * attenuation;
    }
  }

  // Take AO in account
  contrib *= aoFactor;

  // Then proceed with single bounce ray for indirect lighting
  SurfaceHit bounceHit = hit;
  NextRay nextRay = materialNextRay(shadingState, ray, ss.rs);
  if (glm::any(glm::greaterThan(
          nextRay.contributionWeight, glm::vec3(MIN_CONTRIBUTION_EPSILON)))) {
    Ray bounceRay = {
        bounceHit.hitpoint
            + bounceHit.Ng
                * std::copysignf(
                    bounceHit.epsilon, dot(bounceHit.Ns, nextRay.direction)),
        nextRay.direction,
    };

    // Only check for intersecting surfaces and background as secondary light
    // interactions
    bounceHit.foundHit = false;
    intersectSurface(ss, bounceRay, RayType::PRIMARY, &bounceHit);

    if (bounceHit.foundHit) {
      // We hit something. Gather its contribution, cosine weighted diffuse
      // only, we want this to be lightweight.
      MaterialShadingState bounceShadingState;
      materialInitShading(
          &bounceShadingState, frameData, *bounceHit.material, bounceHit);

      auto sampleDir = randomDir(ss.rs, bounceHit.Ns);
      auto cosineT = dot(bounceHit.Ns, sampleDir);
      auto color = materialEvaluateTint(bounceShadingState) * cosineT;
      contrib += color * nextRay.contributionWeight;
    } else {
      // No hit, get background contribution directly (no surface to weight
      // against)
      const auto color = getBackground(frameData, ss.screen, bounceRay.dir);
      contrib += vec3(color) * nextRay.contributionWeight;
    }
  }

  float opacity = evaluateOpacity(shadingState);
  return vec4(contrib, opacity);
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
    SurfaceHit hit;
    ray::populateSurfaceHit(hit);

    MaterialShadingState shadingState;
    materialInitShading(&shadingState, frameData, *hit.material, hit);
    auto opacity = evaluateOpacity(shadingState);

    auto &o = ray::rayData<float>();

    accumulateValue(o, opacity, o);

    if (o >= OPACITY_THRESHOLD)
      optixTerminateRay();
    else
      optixIgnoreIntersection();
  } else {
    auto &attenuation = ray::rayData<float>();
    VolumeHit hit;
    ray::populateVolumeHit(hit);
    rayMarchVolume(ray::screenSample(),
        hit,
        attenuation,
        rendererParams.directLight.inverseVolumeSamplingRateShadows);
    if (attenuation < OPACITY_THRESHOLD)
      optixIgnoreIntersection();
  }
}

VISRTX_GLOBAL void __anyhit__shading()
{
  ray::cullbackFaces();
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

  renderPixel<DirectLightShadingPolicy>(frameData, ss);
}

} // namespace visrtx
