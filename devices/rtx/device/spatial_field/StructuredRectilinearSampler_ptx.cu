/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace visrtx {

VISRTX_CALLABLE void __direct_callable__initStructuredRectilinearSampler(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  samplerState->structuredRectilinear.texObj =
      field->data.structuredRectilinear.texObj;
  samplerState->structuredRectilinear.dims =
      field->data.structuredRectilinear.dims - vec3(1);
  samplerState->structuredRectilinear.offset =
      vec3(field->data.structuredRectilinear.cellCentered ? 0.0f : 0.5f);
  samplerState->structuredRectilinear.axisLUT[0] =
      field->data.structuredRectilinear.axisLUT[0];
  samplerState->structuredRectilinear.axisLUT[1] =
      field->data.structuredRectilinear.axisLUT[1];
  samplerState->structuredRectilinear.axisLUT[2] =
      field->data.structuredRectilinear.axisLUT[2];
  samplerState->structuredRectilinear.axisBoundsMin =
      field->data.structuredRectilinear.axisBoundsMin;
  samplerState->structuredRectilinear.axisBoundsMax =
      field->data.structuredRectilinear.axisBoundsMax;
}

VISRTX_CALLABLE float __direct_callable__sampleStructuredRectilinear(
    const VolumeSamplingState *samplerState, const vec3 *location)
{
  const auto &state = samplerState->structuredRectilinear;

  // Normalize object space to [0, 1] and apply rectilinear mapping
  vec3 normalizedPos = (*location - state.axisBoundsMin)
      / (state.axisBoundsMax - state.axisBoundsMin);

  normalizedPos = vec3(tex1D<float>(state.axisLUT[0], normalizedPos.x),
      tex1D<float>(state.axisLUT[1], normalizedPos.y),
      tex1D<float>(state.axisLUT[2], normalizedPos.z));

  // Sample texture with transformed coordinates
  auto sampleCoord = normalizedPos * state.dims + state.offset;

  return tex3D<float>(
      state.texObj, sampleCoord.x, sampleCoord.y, sampleCoord.z);
}

} // namespace visrtx
