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

#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/shadingState.h"
#include "gpu/shading_api.h"

using namespace visrtx;

VISRTX_CALLABLE void __direct_callable__init(MatteShadingState *shadingState,
    const FrameGPUData *fd,
    const SurfaceHit *hit,
    const MaterialGPUData::Matte *md)
{
  vec4 color = getMaterialParameter(*fd, md->color, *hit);
  float opacity = getMaterialParameter(*fd, md->opacity, *hit).x;

  shadingState->baseColor = vec3(color);
  shadingState->opacity =
      adjustedMaterialOpacity(color.w * opacity, md->alphaMode, md->cutoff);
}

VISRTX_CALLABLE NextRay __direct_callable__nextRay(
    const MatteShadingState *shadingState,
    const Ray *ray,
    RandState *rs)
{
  return NextRay{vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f)};
}

VISRTX_CALLABLE
vec3 __direct_callable__evaluateTint(const MatteShadingState *shadingState)
{
  return shadingState->baseColor;
}

VISRTX_CALLABLE
float __direct_callable__evaluateOpacity(const MatteShadingState *shadingState)
{
  return shadingState->opacity;
}

VISRTX_CALLABLE
vec3 __direct_callable__evaluateEmission(const MatteShadingState *shadingState, const vec3* outgoingDir)
{
  return vec3(0.0f, 0.0f, 0.0f);
}

// Signature must match the call inside shaderMatteSurface in MatteShader.cuh.
VISRTX_CALLABLE vec3 __direct_callable__shadeSurface(
    const MatteShadingState *shadingState,
    const SurfaceHit *hit,
    const LightSample *lightSample,
    const vec3 *outgoingDir)
{
  float NdotL = fmaxf(0.0f, dot(hit->Ns, lightSample->dir));
  return shadingState->baseColor * float(M_1_PI) * NdotL * lightSample->radiance / lightSample->pdf;
}
