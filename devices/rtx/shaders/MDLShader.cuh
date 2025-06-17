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

#pragma once

#include "gpu/gpu_objects.h"
#include "gpu/sampleLight.h"
#include "gpu/shadingState.h"

#include <optix.h>

#include <glm/ext/matrix_float3x4.hpp>
#include <glm/ext/vector_float3.hpp>

#include <mi/neuraylib/target_code_types.h>
#include <optix_device.h>

namespace visrtx {

VISRTX_DEVICE bool mdlInitShading(MDLShadingState *shadingState,
    const FrameGPUData &fd,
    const SurfaceHit &hit,
    const MaterialGPUData::MDL &md)
{
  if (md.implementationIndex == ~DeviceObjectIndex(0))
    return false;

  shadingState->callableBaseIndex = static_cast<unsigned int>(MaterialType::MDL)
      + md.implementationIndex * int(SurfaceShaderEntryPoints::Count);
  // Call must match the implementation in MDLShader_ptx.cu
  optixDirectCall<void>(shadingState->callableBaseIndex + int(SurfaceShaderEntryPoints::Initialize),
      shadingState,
      &fd,
      &hit,
      &md);

  return true;
}

VISRTX_DEVICE vec3 mdlShadeSurface(const MDLShadingState &shadingState,
    const SurfaceHit &hit,
    const LightSample &ls,
    const vec3& outgoingDir)
{
  // Call signature must match the actual implementation in MDLShader_ptx.cu

  return optixDirectCall<vec3>(
      shadingState.callableBaseIndex + int(SurfaceShaderEntryPoints::Shade),
      &shadingState,
      &hit,
      &ls,
      &outgoingDir);
}

VISRTX_DEVICE NextRay mdlNextRay(
    const MDLShadingState& shadingState, const Ray &ray, const ScreenSample &ss)
{
  // Call signature must match the actual implementation in MDLShader_ptx.cu
  return optixDirectCall<NextRay>(
      shadingState.callableBaseIndex + int(SurfaceShaderEntryPoints::EvaluateNextRay),
      &shadingState,
      &ray,
      &ss);
}

VISRTX_DEVICE vec3 mdlEvaluateTint(
  const MDLShadingState& shadingState)
{
  return optixDirectCall<vec3>(
    shadingState.callableBaseIndex + int(SurfaceShaderEntryPoints::EvaluateTint),
      &shadingState
  );
}

VISRTX_DEVICE float mdlEvaluateOpacity(
  const MDLShadingState& shadingState)
{
  return optixDirectCall<float>(
    shadingState.callableBaseIndex + int(SurfaceShaderEntryPoints::EvaluateOpacity),
      &shadingState
  );
}

} // namespace visrtx
