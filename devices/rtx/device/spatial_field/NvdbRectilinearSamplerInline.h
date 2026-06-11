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

#pragma once

// NanoVDB rectilinear-grid sampler — inline device implementations.
// See NvdbRegularSamplerInline.h for why these live in a header.

#include "NvdbRegularSamplerInline.h" // for clampNvdb

#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/shadingState.h"

#include <driver_types.h>
#include <nanovdb/math/Math.h>

namespace visrtx {

template <typename ValueType>
VISRTX_DEVICE void initNvdbRectilinearSampler(
    NvdbRectilinearSamplerState<ValueType> &state,
    const SpatialFieldGPUData *field)
{
  using GridType = nanovdb::Grid<nanovdb::NanoTree<ValueType>>;
  const auto *grid =
      static_cast<const GridType *>(field->data.nvdbRectilinear.gridData);

  state.grid = grid;
  // Leaf-only accessor — see NvdbRegularSamplerInline.h for rationale.
  state.accessor =
      typename NvdbRectilinearSamplerState<ValueType>::AccessorType(
          grid->tree().root());
  state.filter = field->data.nvdbRectilinear.filter;
  if (state.filter == SpatialFieldFilter::Nearest) {
    new (&state.nearestSampler)
        typename NvdbRectilinearSamplerState<ValueType>::NearestSamplerType(
            nanovdb::math::createSampler<0>(state.accessor));
  } else {
    new (&state.linearSampler)
        typename NvdbRectilinearSamplerState<ValueType>::LinearSamplerType(
            nanovdb::math::createSampler<1>(state.accessor));
  }

  const nanovdb::CoordBBox indexBBox = grid->indexBBox();
  const nanovdb::Vec3f dims = nanovdb::Vec3f(indexBBox.dim());

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

  const auto &iavs = field->data.nvdbRectilinear.invAvgVoxelSize;
  state.invAvgVoxelSize = nanovdb::Vec3f(iavs.x, iavs.y, iavs.z);
}

template <typename ValueType>
VISRTX_DEVICE nanovdb::Vec3f worldToIndexRectilinear(
    const NvdbRectilinearSamplerState<ValueType> &state, const vec3 *location)
{
  const auto indexPos0 = state.grid->worldToIndexF(
      nanovdb::Vec3f(location->x, location->y, location->z));

  const auto normalizedPos = (indexPos0 - state.offsetDown) * state.scaleDown;

  const auto normalizedPosRect =
      nanovdb::Vec3f(tex1D<float>(state.axisLUT[0], normalizedPos[0]),
          tex1D<float>(state.axisLUT[1], normalizedPos[1]),
          tex1D<float>(state.axisLUT[2], normalizedPos[2]));

  return normalizedPosRect * state.scaleUp + state.offsetUp;
}

template <typename ValueType>
VISRTX_DEVICE float sampleAtIndexRectilinear(
    const NvdbRectilinearSamplerState<ValueType> &state,
    const nanovdb::Vec3f &indexPos)
{
  const auto clamped = clampNvdb(indexPos, state.indexMin, state.indexMax);
  if (state.filter == SpatialFieldFilter::Nearest)
    return state.nearestSampler(clamped);
  return state.linearSampler(clamped);
}

// Shared per-sample API (see gpu/volumeIntegrationDetail.h). sampleValue
// returns the field value; sampleNormal returns the unnormalized object-space
// gradient (the raw normal direction) — callers orient and normalize. `field`
// is unused for built-in fields (present for a uniform overload set).
template <typename ValueType>
VISRTX_DEVICE float sampleValue(
    const NvdbRectilinearSamplerState<ValueType> &state,
    const SpatialFieldGPUData &,
    const vec3 &p)
{
  return sampleAtIndexRectilinear(state, worldToIndexRectilinear(state, &p));
}

template <typename ValueType>
VISRTX_DEVICE vec3 sampleNormal(
    const NvdbRectilinearSamplerState<ValueType> &state,
    const SpatialFieldGPUData &,
    const vec3 &p)
{
  const auto indexPos = worldToIndexRectilinear(state, &p);
  const float sxp =
      sampleAtIndexRectilinear(state, indexPos + nanovdb::Vec3f(1, 0, 0));
  const float sxn =
      sampleAtIndexRectilinear(state, indexPos - nanovdb::Vec3f(1, 0, 0));
  const float syp =
      sampleAtIndexRectilinear(state, indexPos + nanovdb::Vec3f(0, 1, 0));
  const float syn =
      sampleAtIndexRectilinear(state, indexPos - nanovdb::Vec3f(0, 1, 0));
  const float szp =
      sampleAtIndexRectilinear(state, indexPos + nanovdb::Vec3f(0, 0, 1));
  const float szn =
      sampleAtIndexRectilinear(state, indexPos - nanovdb::Vec3f(0, 0, 1));
  return vec3(sxp - sxn, syp - syn, szp - szn)
      * vec3(state.invAvgVoxelSize[0],
          state.invAvgVoxelSize[1],
          state.invAvgVoxelSize[2])
      * 0.5f;
}

} // namespace visrtx
