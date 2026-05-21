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

// NanoVDB regular-grid sampler — inline device implementations.
// Lives in a header so the volume integrator can call sampleNvdb directly,
// bypassing the OptiX direct-callable dispatch on hot paths. The matching
// __direct_callable__ entry points stay in NvdbRegularSampler_ptx.cu for
// renderers that go through the SBT slot.

#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/shadingState.h"

#include <nanovdb/math/Math.h>

namespace visrtx {

VISRTX_DEVICE nanovdb::Vec3f clampNvdb(const nanovdb::Vec3f &v,
    const nanovdb::Vec3f &min,
    const nanovdb::Vec3f &max)
{
  return nanovdb::Vec3f(nanovdb::math::Clamp(v[0], min[0], max[0]),
      nanovdb::math::Clamp(v[1], min[1], max[1]),
      nanovdb::math::Clamp(v[2], min[2], max[2]));
}

template <typename ValueType>
VISRTX_DEVICE void initNvdbSampler(
    NvdbRegularSamplerState<ValueType> &state, const SpatialFieldGPUData *field)
{
  using GridType = nanovdb::Grid<nanovdb::NanoTree<ValueType>>;
  const auto *grid =
      static_cast<const GridType *>(field->data.nvdbRegular.gridData);

  state.grid = grid;
  state.accessor = grid->getAccessor();
  state.filter = field->data.nvdbRegular.filter;
  // Use placement new to construct sampler in-place, as we cannot assign
  // because of deleted constructor of the nanovdb samplers
  if (state.filter == SpatialFieldFilter::Nearest) {
    new (&state.nearestSampler)
        typename NvdbRegularSamplerState<ValueType>::NearestSamplerType(
            nanovdb::math::createSampler<0>(state.accessor));
  } else {
    new (&state.linearSampler)
        typename NvdbRegularSamplerState<ValueType>::LinearSamplerType(
            nanovdb::math::createSampler<1>(state.accessor));
  }

  const nanovdb::CoordBBox indexBBox = grid->indexBBox();
  const nanovdb::Vec3f dims = nanovdb::Vec3f(indexBBox.dim());
  // NanoVDB samplers get exact values at 0, 1, ... N, which works for
  // node centered data. For cell centered data, we need to offset by -0.5
  // and clamp to artificially create the full voxel, extrapolating the
  // outermost voxel values.
  // Scale moves from index space to index space - 1
  state.offsetDown = -nanovdb::Vec3f(indexBBox.min());
  if (field->data.nvdbRegular.cellCentered) {
    state.offsetUp = nanovdb::Vec3f(-0.5f) + state.offsetDown;
    state.scale = nanovdb::Vec3f(1.0f);
  } else {
    state.offsetUp = state.offsetDown;
    state.scale = (dims - nanovdb::Vec3f(1.0f)) / dims;
  }

  state.indexMin = nanovdb::Vec3f(indexBBox.min());
  state.indexMax = nanovdb::Vec3f(indexBBox.max());
}

template <typename ValueType>
VISRTX_DEVICE float sampleNvdb(
    const NvdbRegularSamplerState<ValueType> &state,
    const vec3 *location,
    vec3 *gradient)
{
  const auto indexPos0 = state.grid->worldToIndexF(
      nanovdb::Vec3f(location->x, location->y, location->z));

  const auto indexPos =
      (indexPos0 - state.offsetDown) * state.scale + state.offsetUp;

  const auto clamped = clampNvdb(indexPos, state.indexMin, state.indexMax);

  if (state.filter == SpatialFieldFilter::Nearest) {
    const float value = state.nearestSampler(clamped);
    if (gradient) {
      // Central differences at ±1 voxel in index space
      const auto voxelSize = state.grid->voxelSize();
      const float sxp = state.nearestSampler(clampNvdb(
          indexPos + nanovdb::Vec3f(1, 0, 0), state.indexMin, state.indexMax));
      const float sxn = state.nearestSampler(clampNvdb(
          indexPos - nanovdb::Vec3f(1, 0, 0), state.indexMin, state.indexMax));
      const float syp = state.nearestSampler(clampNvdb(
          indexPos + nanovdb::Vec3f(0, 1, 0), state.indexMin, state.indexMax));
      const float syn = state.nearestSampler(clampNvdb(
          indexPos - nanovdb::Vec3f(0, 1, 0), state.indexMin, state.indexMax));
      const float szp = state.nearestSampler(clampNvdb(
          indexPos + nanovdb::Vec3f(0, 0, 1), state.indexMin, state.indexMax));
      const float szn = state.nearestSampler(clampNvdb(
          indexPos - nanovdb::Vec3f(0, 0, 1), state.indexMin, state.indexMax));
      // Convert from index space to object space
      *gradient = vec3((sxp - sxn) * state.scale[0] / (2.f * voxelSize[0]),
          (syp - syn) * state.scale[1] / (2.f * voxelSize[1]),
          (szp - szn) * state.scale[2] / (2.f * voxelSize[2]));
    }
    return value;
  }

  const float value = state.linearSampler(clamped);
  if (gradient) {
    const float sxp = state.linearSampler(clampNvdb(
        indexPos + nanovdb::Vec3f(1, 0, 0), state.indexMin, state.indexMax));
    const float sxn = state.linearSampler(clampNvdb(
        indexPos - nanovdb::Vec3f(1, 0, 0), state.indexMin, state.indexMax));
    const float syp = state.linearSampler(clampNvdb(
        indexPos + nanovdb::Vec3f(0, 1, 0), state.indexMin, state.indexMax));
    const float syn = state.linearSampler(clampNvdb(
        indexPos - nanovdb::Vec3f(0, 1, 0), state.indexMin, state.indexMax));
    const float szp = state.linearSampler(clampNvdb(
        indexPos + nanovdb::Vec3f(0, 0, 1), state.indexMin, state.indexMax));
    const float szn = state.linearSampler(clampNvdb(
        indexPos - nanovdb::Vec3f(0, 0, 1), state.indexMin, state.indexMax));
    const auto voxelSize = state.grid->voxelSize();
    *gradient = vec3((sxp - sxn) * state.scale[0] / (2.f * voxelSize[0]),
        (syp - syn) * state.scale[1] / (2.f * voxelSize[1]),
        (szp - szn) * state.scale[2] / (2.f * voxelSize[2]));
  }
  return value;
}

} // namespace visrtx
