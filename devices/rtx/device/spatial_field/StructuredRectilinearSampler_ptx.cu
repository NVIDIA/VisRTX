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
  auto &state = samplerState->structuredRectilinear;
  const auto &data = field->data.structuredRectilinear;

  state.texObj = data.texObj;
  state.dims = data.dims - vec3(1);
  state.offset = vec3(data.cellCentered ? 0.0f : 0.5f);
  state.axisLUT[0] = data.axisLUT[0];
  state.axisLUT[1] = data.axisLUT[1];
  state.axisLUT[2] = data.axisLUT[2];
  state.axisBoundsMin = data.axisBoundsMin;
  state.axisBoundsMax = data.axisBoundsMax;

  const vec3 extent = data.axisBoundsMax - data.axisBoundsMin;
  state.invAvgVoxelSpacing =
      data.cellCentered ? (state.dims + vec3(1)) / extent : state.dims / extent;
}

VISRTX_CALLABLE float __direct_callable__sampleStructuredRectilinear(
    const VolumeSamplingState *samplerState,
    const vec3 *location,
    vec3 *gradient)
{
  const auto &state = samplerState->structuredRectilinear;

  // World-to-texel coordinate transform
  vec3 normalizedPos = (*location - state.axisBoundsMin)
      / (state.axisBoundsMax - state.axisBoundsMin);
  normalizedPos = vec3(tex1D<float>(state.axisLUT[0], normalizedPos.x),
      tex1D<float>(state.axisLUT[1], normalizedPos.y),
      tex1D<float>(state.axisLUT[2], normalizedPos.z));
  const auto sampleCoord = normalizedPos * state.dims + state.offset;

  const float value =
      tex3D<float>(state.texObj, sampleCoord.x, sampleCoord.y, sampleCoord.z);

  if (gradient) {
    // Neighbor-voxel central differences at ±1 texel offset
    const auto px = sampleCoord + vec3(1, 0, 0);
    const auto nx = sampleCoord - vec3(1, 0, 0);
    const auto py = sampleCoord + vec3(0, 1, 0);
    const auto ny = sampleCoord - vec3(0, 1, 0);
    const auto pz = sampleCoord + vec3(0, 0, 1);
    const auto nz = sampleCoord - vec3(0, 0, 1);

    const float sxp = tex3D<float>(state.texObj, px.x, px.y, px.z);
    const float sxn = tex3D<float>(state.texObj, nx.x, nx.y, nx.z);
    const float syp = tex3D<float>(state.texObj, py.x, py.y, py.z);
    const float syn = tex3D<float>(state.texObj, ny.x, ny.y, ny.z);
    const float szp = tex3D<float>(state.texObj, pz.x, pz.y, pz.z);
    const float szn = tex3D<float>(state.texObj, nz.x, nz.y, nz.z);

    // Gradient in object space: scale by invAvgVoxelSpacing
    *gradient =
        vec3(sxp - sxn, syp - syn, szp - szn) * state.invAvgVoxelSpacing * 0.5f;
  }

  return value;
}

} // namespace visrtx
