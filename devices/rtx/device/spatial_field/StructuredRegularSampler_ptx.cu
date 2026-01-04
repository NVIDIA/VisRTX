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

using namespace visrtx;

VISRTX_CALLABLE void __direct_callable__initStructuredRegularSampler(
    VolumeSamplingState *samplerState,
    const SpatialFieldGPUData *field)
{
  samplerState->structuredRegular.texObj = field->data.structuredRegular.texObj;
  samplerState->structuredRegular.origin = field->data.structuredRegular.origin;
  samplerState->structuredRegular.invDims = field->data.structuredRegular.invDims;
  samplerState->structuredRegular.invSpacing = field->data.structuredRegular.invSpacing;
  samplerState->structuredRegular.offset = vec3(field->data.structuredRegular.cellCentered ? 0.0f : 0.5f);
}

VISRTX_CALLABLE float __direct_callable__sampleStructuredRegular(
    const VolumeSamplingState *samplerState,
    const vec3 *location)
{
  const auto &state = samplerState->structuredRegular;
  const auto texelCoords = (*location - state.origin) * state.invSpacing;
  const auto coords = texelCoords + state.offset;
  return tex3D<float>(state.texObj, coords.x, coords.y, coords.z);
}
