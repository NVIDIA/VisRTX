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

#define VISRTX_DEBUGGING 0

#include "gpu/evalShading.h"
#include "gpu/gpu_debug.h"
#include "gpu/shadingState.h"
#include "gpu/shading_api.h"
namespace visrtx {

enum class RayType
{
  DIFFUSE_RADIANCE
};

struct PathData
{
  int depth{0};
  vec3 Lw{1.f};
  Hit currentHit{};
};

DECLARE_FRAME_DATA(frameData)

// OptiX programs /////////////////////////////////////////////////////////////

VISRTX_GLOBAL void __closesthit__()
{
  ray::populateHit();
}

VISRTX_GLOBAL void __anyhit__()
{
  SurfaceHit hit;
  ray::populateSurfaceHit(hit);

  const auto &md = *hit.material;

  MaterialShadingState shadingState;
  materialInitShading(&shadingState, frameData, md, hit);
  auto materialOpacity = materialEvaluateOpacity(shadingState);

  float opacity = materialOpacity;
  if (opacity < 0.99f && curand_uniform(&ray::screenSample().rs) > opacity)
    optixIgnoreIntersection();
}

VISRTX_GLOBAL void __miss__()
{
  // no-op
}

VISRTX_GLOBAL void __raygen__()
{
  auto &rendererParams = frameData.renderer;
  auto &dptParams = rendererParams.params.dpt;

  PathData pathData;

  auto &hit = pathData.currentHit;

  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;

  for (int i = 0; i < frameData.renderer.numIterations; i++) {
    auto ray = makePrimaryRay(ss);
    auto tmax = ray.t.upper;

    if (debug())
      printf("========== BEGIN: FrameID %i ==========\n", frameData.fb.frameID);

    const auto bg = getBackground(frameData, ss.screen, ray.dir);
    vec3 outColor(bg);
    vec3 outNormal = ray.dir;
    float outDepth = tmax;
    uint32_t primID = ~0u;
    uint32_t objID = ~0u;
    uint32_t instID = ~0u;

    while (true) {
      if (debug())
        printf("-------- BOUNCE: %i --------\n", pathData.depth);
      hit.foundHit = false;
      intersectSurface(ss,
          ray,
          RayType::DIFFUSE_RADIANCE,
          &hit,
          primaryRayOptiXFlags(rendererParams));

      float volumeOpacity = 0.f;
      vec3 volumeColor(0.f);
      float Tr = 0.f;
      uint32_t vObjID = ~0u;
      uint32_t vInstID = ~0u;
      const float volumeDepth = sampleDistanceAllVolumes(ss,
          ray,
          RayType::DIFFUSE_RADIANCE,
          hit.foundHit ? hit.t : ray.t.upper,
          volumeColor,
          volumeOpacity,
          Tr,
          vObjID,
          vInstID);

      const bool volumeHit = Tr < 1.f && (!hit.foundHit || volumeDepth < hit.t);

      if (!hit.foundHit && !volumeHit)
        break;

      if (pathData.depth++ >= dptParams.maxDepth) {
        pathData.Lw = vec3(0.f);
        break;
      }

      vec3 albedo(1.f);
      vec3 pos(0.f);

      if (!volumeHit) {
        pos = hit.hitpoint + (hit.epsilon * hit.Ng);
        const auto &material = *hit.material;
        MaterialShadingState shadingState;
        materialInitShading(&shadingState, frameData, material, hit);
        albedo = materialEvaluateTint(shadingState);
      } else {
        pos = ray.org + volumeDepth * ray.dir;
        albedo = volumeColor;
      }
      pathData.Lw *= albedo;

      // RR absorption
      float P = glm::compMax(pathData.Lw);
      if (P < .2f /*lp.rouletteProb*/) {
        if (curand_uniform(&ss.rs) > P) {
          pathData.Lw = vec3(0.f);
          break;
        }
        pathData.Lw /= P;
      }

      // pathData.Lw += Le; // TODO: emission

      vec3 scatterDir(0.f);
      if (!volumeHit) {
        scatterDir = randomDir(ss.rs, hit.Ns);
        pathData.Lw *= fmaxf(0.f, dot(scatterDir, hit.Ng));
      } else
        scatterDir = sampleUnitSphere(ss.rs, -ray.dir);

      ray.org = pos;
      ray.dir = scatterDir;
      ray.t.lower = 0.f;
      ray.t.upper = rendererParams.occlusionDistance;

      if (pathData.depth == 0) {
        const bool volumeFirst = volumeDepth < hit.t;
        outDepth = volumeFirst ? volumeDepth : hit.t;
        outNormal = hit.Ng; // TODO: for volume (gradient?)
        primID = volumeFirst ? 0 : computeGeometryPrimId(hit);
        objID = volumeFirst ? vObjID : hit.objID;
        instID = volumeFirst ? vInstID : hit.instID;
      }
    }

    vec3 Ld(rendererParams.ambientIntensity); // ambient light!
    // if (numLights > 0) {
    //   Ld = ...;
    // }

    vec3 color = pathData.depth ? pathData.Lw * Ld : vec3(bg);
    if (crosshair())
      color = vec3(1) - color;
    if (debug())
      printf("========== END: FrameID %i ==========\n", frameData.fb.frameID);
    accumResults(frameData.fb,
        ss.pixel,
        vec4(color, 1.f),
        outDepth,
        outColor,
        outNormal,
        primID,
        objID,
        instID,
        i);
  }
}

} // namespace visrtx
