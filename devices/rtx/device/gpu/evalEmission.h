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

// Material emission entry points, factored out of evalShading.h so the light
// sampler (sampleLight.h) can evaluate a Geometry Light's emission at the
// sampled point through the SAME callables the hit-side deposit uses — keeping
// next-event and path-hit radiance identical (MIS stays unbiased). This header
// sits below sampleLight.h in the include graph; evalShading.h re-exports it.

#include <optix_device.h>
#include "gpu/gpu_objects.h"
#include "gpu/sbt.h"
#include "shadingState.h"

namespace visrtx {

// Every material's __direct_callable__init returns void (see the *Shader_ptx.cu
// files). Calling them through optixDirectCall<bool> read an indeterminate
// return value; nothing consumed it — the emission/shading entry points gate on
// callableBaseIndex, set here — so it just invited a caller to branch on garbage.
// Return void and match the callables' ABI.
VISRTX_DEVICE void materialInitShading(MaterialShadingState *shadingState,
    const FrameGPUData &fd,
    const MaterialGPUData &md,
    const SurfaceHit &hit)
{
  if (md.callableBaseIndex == ~DeviceObjectIndex(0)) {
    shadingState->callableBaseIndex = ~0;
    return;
  }

  shadingState->callableBaseIndex = md.callableBaseIndex;

  optixDirectCall<void>(shadingState->callableBaseIndex,
      &shadingState->data,
      &fd,
      &hit,
      &md.materialData);
}

VISRTX_DEVICE vec3 materialEvaluateEmission(
    const MaterialShadingState &shadingState, const vec3 &outgoingDir)
{
  if (shadingState.callableBaseIndex == ~DeviceObjectIndex(0))
    return vec3(0.0f, 0.0f, 0.0f); // Default emission color

  return optixDirectCall<vec3>(shadingState.callableBaseIndex
          + int(SurfaceShaderEntryPoints::EvaluateEmission),
      &shadingState.data,
      &outgoingDir);
}

// Emitted radiance of a surface material at `hit` toward `outgoingDir`. Runs the
// material's own init + emission entry points on the supplied hit; used by the
// Geometry Light sampler with a synthetic hit at the sampled point.
VISRTX_DEVICE vec3 evaluateSurfaceEmission(const FrameGPUData &fd,
    const MaterialGPUData &md,
    const SurfaceHit &hit,
    const vec3 &outgoingDir)
{
  MaterialShadingState shadingState;
  materialInitShading(&shadingState, fd, md, hit);
  return materialEvaluateEmission(shadingState, outgoingDir);
}

} // namespace visrtx
