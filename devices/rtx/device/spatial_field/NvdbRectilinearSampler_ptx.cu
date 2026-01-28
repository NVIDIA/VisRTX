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

#include <driver_types.h>
#include <nanovdb/math/Math.h>
#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/shadingState.h"

using namespace visrtx;

VISRTX_DEVICE nanovdb::Vec3f clamp(const nanovdb::Vec3f &v,
    const nanovdb::Vec3f &min,
    const nanovdb::Vec3f &max)
{
  return nanovdb::Vec3f(nanovdb::math::Clamp(v[0], min[0], max[0]),
      nanovdb::math::Clamp(v[1], min[1], max[1]),
      nanovdb::math::Clamp(v[2], min[2], max[2]));
}

template <typename ValueType>
VISRTX_DEVICE void initNvdbRectilinearSampler(
    NvdbRectilinearSamplerState<ValueType> &state,
    const SpatialFieldGPUData *field)
{
  using GridType = nanovdb::Grid<nanovdb::NanoTree<ValueType>>;
  const auto *grid =
      static_cast<const GridType *>(field->data.nvdbRectilinear.gridData);

  state.grid = grid;
  state.accessor = grid->getAccessor();
  // Use placement new to construct sampler in-place, as we cannot assign
  // because of deleted constructor of the nanovdb samplers
  new (&state.sampler)
      typename NvdbRectilinearSamplerState<ValueType>::SamplerType(
          nanovdb::math::createSampler<1>(state.accessor));

  const nanovdb::CoordBBox indexBBox = grid->indexBBox();
  const nanovdb::Vec3f dims = nanovdb::Vec3f(indexBBox.dim());

  // NanoVDB samplers get exact values at 0, 1, ... N, which works for
  // node centered data. For cell centered data, we need to offset by -0.5
  // and clamp to artificially create the full voxel, extrapolating the
  // outermost voxel values.
  // ScaleDown moves from index space to normalized space [0, 1]
  // ScaleUp moves from normalized space [0, 1] to index space - 1
  state.scaleDown = 1.0f / dims;
  state.scaleUp = dims - nanovdb::Vec3f(1.0f);
  state.offsetDown = -nanovdb::Vec3f(indexBBox.min());
  if (field->data.nvdbRectilinear.cellCentered) {
    state.offsetUp = nanovdb::Vec3f(-0.5f) + state.offsetDown;
  } else {
    state.offsetUp = state.offsetDown;
  }
  state.indexMin = nanovdb::Vec3f(indexBBox.min());
  state.indexMax = nanovdb::Vec3f(indexBBox.max());

  state.axisLUT[0] = field->data.nvdbRectilinear.axisLUT[0];
  state.axisLUT[1] = field->data.nvdbRectilinear.axisLUT[1];
  state.axisLUT[2] = field->data.nvdbRectilinear.axisLUT[2];
}

template <typename ValueType>
VISRTX_DEVICE float sampleNvdbRectilinear(
    const NvdbRectilinearSamplerState<ValueType> &state, const vec3 *location)
{
  const auto indexPos0 = state.grid->worldToIndexF(
      nanovdb::Vec3f(location->x, location->y, location->z));

  // Recenter and normalize
  const auto normalizedPos = (indexPos0 - state.offsetDown) * state.scaleDown;

  // Apply rectilinear mapping
  const auto normalizedPosRect =
      nanovdb::Vec3f(tex1D<float>(state.axisLUT[0], normalizedPos[0]),
          tex1D<float>(state.axisLUT[1], normalizedPos[1]),
          tex1D<float>(state.axisLUT[2], normalizedPos[2]));

  // Back to index space
  const auto indexPos = normalizedPosRect * state.scaleUp + state.offsetUp;

  return state.sampler(clamp(indexPos, state.indexMin, state.indexMax));
}

// Fp4 rectilinear sampler
VISRTX_CALLABLE void __direct_callable__initNvdbRectilinearSamplerFp4(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbRectilinearSampler(samplerState->nvdbRectilinearFp4, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbRectilinearFp4(
    const VolumeSamplingState *samplerState, const vec3 *location)
{
  return sampleNvdbRectilinear(samplerState->nvdbRectilinearFp4, location);
}

// Fp8 rectilinear sampler
VISRTX_CALLABLE void __direct_callable__initNvdbRectilinearSamplerFp8(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbRectilinearSampler(samplerState->nvdbRectilinearFp8, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbRectilinearFp8(
    const VolumeSamplingState *samplerState, const vec3 *location)
{
  return sampleNvdbRectilinear(samplerState->nvdbRectilinearFp8, location);
}

// Fp16 rectilinear sampler
VISRTX_CALLABLE void __direct_callable__initNvdbRectilinearSamplerFp16(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbRectilinearSampler(samplerState->nvdbRectilinearFp16, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbRectilinearFp16(
    const VolumeSamplingState *samplerState, const vec3 *location)
{
  return sampleNvdbRectilinear(samplerState->nvdbRectilinearFp16, location);
}

// FpN rectilinear sampler
VISRTX_CALLABLE void __direct_callable__initNvdbRectilinearSamplerFpN(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbRectilinearSampler(samplerState->nvdbRectilinearFpN, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbRectilinearFpN(
    const VolumeSamplingState *samplerState, const vec3 *location)
{
  return sampleNvdbRectilinear(samplerState->nvdbRectilinearFpN, location);
}

// Float rectilinear sampler
VISRTX_CALLABLE void __direct_callable__initNvdbRectilinearSamplerFloat(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData *field)
{
  initNvdbRectilinearSampler(samplerState->nvdbRectilinearFloat, field);
}

VISRTX_CALLABLE float __direct_callable__sampleNvdbRectilinearFloat(
    const VolumeSamplingState *samplerState, const vec3 *location)
{
  return sampleNvdbRectilinear(samplerState->nvdbRectilinearFloat, location);
}
