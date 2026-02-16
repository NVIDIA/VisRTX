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

#include <limits>
#include "gpu/evalShading.h"
#include "gpu/gpu_decl.h"
#include "gpu/sampleLight.h"
#include "gpu/shadingState.h"
#include "gpu/shading_api.h"
#include "gpu/renderer/common.h"
#include "gpu/renderer/raygen_helpers.h"

namespace visrtx {

DECLARE_FRAME_DATA(frameData)

// OptiX programs /////////////////////////////////////////////////////////////

VISRTX_GLOBAL void __closesthit__shadow()
{
  // no-op
}

VISRTX_GLOBAL void __anyhit__shadow()
{
  SurfaceHit hit;
  ray::populateSurfaceHit(hit);

  const auto &fd = frameData;
  const auto &md = *hit.material;
  MaterialShadingState shadingState;
  materialInitShading(&shadingState, fd, md, hit);

  auto &o = ray::rayData<float>();
  accumulateValue(o, materialEvaluateOpacity(shadingState), o);
  if (o >= OPACITY_THRESHOLD)
    optixTerminateRay();
  else
    optixIgnoreIntersection();
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
  // no-op
}

// Fast shading policy for templated rendering loop /////////////////////////

struct FastShadingPolicy
{
  static VISRTX_DEVICE vec4 shadeSurface(const MaterialShadingState &shadingState,
      ScreenSample &ss,
      const Ray &ray,
      const SurfaceHit &hit)
  {
    const auto &rendererParams = frameData.renderer;
    const auto &aoParams = rendererParams.params.fast;

    const float ndotl = glm::abs(glm::dot(ray.dir, hit.Ns));

    const bool traceAO = aoParams.aoBlend > 0.f && aoParams.aoSamples > 0;
    const float aoFactor = traceAO
        ? computeAO(ss,
              ray,
              hit,
              rendererParams.occlusionDistance,
              aoParams.aoSamples,
              &surfaceAttenuation)
        : 1.f;

    auto materialBaseColor = materialEvaluateTint(shadingState);
    auto materialOpacity = materialEvaluateOpacity(shadingState);

    const float lighting = glm::mix(ndotl,
        aoFactor * rendererParams.ambientIntensity,
        aoParams.aoBlend);

    return vec4(materialBaseColor * lighting * rendererParams.ambientColor,
        materialOpacity);
  }
};

VISRTX_GLOBAL void __raygen__()
{
  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;

  renderPixel<FastShadingPolicy>(frameData, ss);
}

} // namespace visrtx
