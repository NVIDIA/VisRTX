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

#pragma once

#include <optix_device.h>
#include "gpu/gpu_objects.h"
#include "gpu/sampleLight.h"
#include "gpu/sbt.h"
#include "shadingState.h"

namespace visrtx {

VISRTX_DEVICE bool materialInitShading(MaterialShadingState *shadingState,
    const FrameGPUData &fd,
    const MaterialGPUData &md,
    const SurfaceHit &hit)
{
  if (md.callableBaseIndex == ~DeviceObjectIndex(0)) {
    shadingState->callableBaseIndex = ~0;
    return false;
  }

  shadingState->callableBaseIndex = md.callableBaseIndex;

  return optixDirectCall<bool>(shadingState->callableBaseIndex,
      &shadingState->data,
      &fd,
      &hit,
      &md.materialData);
}

VISRTX_DEVICE vec3 materialEvaluateTint(
    const MaterialShadingState &shadingState)
{
  if (shadingState.callableBaseIndex == ~DeviceObjectIndex(0))
    return vec3(0.8f, 0.8f, 0.8f); // Default tint color

  return optixDirectCall<vec3>(shadingState.callableBaseIndex
          + int(SurfaceShaderEntryPoints::EvaluateTint),
      &shadingState.data);
}

VISRTX_DEVICE float materialEvaluateOpacity(
    const MaterialShadingState &shadingState)
{
  if (shadingState.callableBaseIndex == ~DeviceObjectIndex(0))
    return 1.0f; // Default opacity

  return optixDirectCall<float>(shadingState.callableBaseIndex
          + int(SurfaceShaderEntryPoints::EvaluateOpacity),
      &shadingState.data);
}

VISRTX_DEVICE vec3 materialEvaluateEmission(
    const MaterialShadingState &shadingState, const vec3& outgoingDir)
{
  if (shadingState.callableBaseIndex == ~DeviceObjectIndex(0))
    return vec3(0.0f, 0.0f, 0.0f); // Default emission color

  return optixDirectCall<vec3>(shadingState.callableBaseIndex
          + int(SurfaceShaderEntryPoints::EvaluateEmission),
      &shadingState.data, &outgoingDir);
}

VISRTX_DEVICE vec3 materialEvaluateTransmission(
    const MaterialShadingState &shadingState)
{
  if (shadingState.callableBaseIndex == ~DeviceObjectIndex(0))
    return vec3(0.0f); // Default transmission

  return optixDirectCall<vec3>(shadingState.callableBaseIndex
          + int(SurfaceShaderEntryPoints::EvaluateTransmission),
      &shadingState.data);
}

VISRTX_DEVICE vec3 materialEvaluateNormal(
    const MaterialShadingState &shadingState)
{
  if (shadingState.callableBaseIndex == ~DeviceObjectIndex(0))
    return vec3(0.8f, 0.8f, 0.8f); // Default tint color

  return optixDirectCall<vec3>(shadingState.callableBaseIndex
          + int(SurfaceShaderEntryPoints::EvaluateNormal),
      &shadingState.data);
}

VISRTX_DEVICE NextRay materialNextRay(const MaterialShadingState &shadingState,
    const Ray &ray, RandState& rs)
{
  if (shadingState.callableBaseIndex == ~DeviceObjectIndex(0)) // No next ray by defaut
    return NextRay{vec4(0.0f), vec4(0.0f)};

  return optixDirectCall<NextRay>(shadingState.callableBaseIndex
          + int(SurfaceShaderEntryPoints::EvaluateNextRay),
      &shadingState.data,
      &ray,
      &rs);
}

VISRTX_DEVICE vec3 materialShadeSurface(
    const MaterialShadingState &shadingState,
    const SurfaceHit &hit,
    const LightSample &lightSample,
    const vec3 &outgoingDir)
{
  if (shadingState.callableBaseIndex == ~DeviceObjectIndex(0))
    return vec3(0.0f, 0.0f, 0.0f); // No shading by default

  return optixDirectCall<vec3>(
      shadingState.callableBaseIndex + int(SurfaceShaderEntryPoints::Shade),
      &shadingState.data,
      &hit,
      &lightSample,
      &outgoingDir);
}

} // namespace visrtx
