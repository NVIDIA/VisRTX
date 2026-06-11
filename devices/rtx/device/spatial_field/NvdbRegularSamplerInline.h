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
// Lives in a header so the volume integrator and isosurface can call
// sampleValue/sampleNormal directly, bypassing the OptiX direct-callable
// dispatch on hot paths. The matching __direct_callable__ entry points stay in
// NvdbRegularSampler_ptx.cu for renderers that go through the SBT slot.

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
  // Construct the (leaf-only) accessor directly from the tree root —
  // grid->getAccessor() would return the 3-level DefaultReadAccessor and
  // mismatch our AccessorType typedef.
  state.accessor = typename NvdbRegularSamplerState<ValueType>::AccessorType(
      grid->tree().root());
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
  // NanoVDB's index space spans [0, N] over the world bbox (it treats the N
  // voxels as cells [i, i+1]), but node-centered data of N nodes only fills the
  // index range [0, N-1]. So node-centered scales the (min-relative) index by
  // (N-1)/N to map the full [0,N] domain onto the node range, giving symmetric
  // half voxels at both boundaries; the identity (scale 1) instead leaves the
  // phantom cell [N-1, N] as a full constant voxel at the high boundary. Cell-
  // centered data stores values at the cell centers, so shift by -0.5 and clamp
  // (extrapolating the outermost voxel) — no stretch. offsetDown is +min, so a
  // grid whose index bbox does not start at 0 still maps correctly.
  state.offsetDown = nanovdb::Vec3f(indexBBox.min());
  if (field->data.nvdbRegular.cellCentered) {
    state.offsetUp = nanovdb::Vec3f(indexBBox.min()) - nanovdb::Vec3f(0.5f);
    state.scale = nanovdb::Vec3f(1.0f);
  } else {
    state.offsetUp = nanovdb::Vec3f(indexBBox.min());
    state.scale = (dims - nanovdb::Vec3f(1.0f)) / dims;
  }

  state.indexMin = nanovdb::Vec3f(indexBBox.min());
  state.indexMax = nanovdb::Vec3f(indexBBox.max());

  // Use host-precomputed invTwoVoxelSize — keeps Grid::voxelSize() (Vec3d)
  // off the device init path.
  const vec3 &iv = field->data.nvdbRegular.invTwoVoxelSize;
  state.invTwoVoxelSize = nanovdb::Vec3f(iv.x, iv.y, iv.z);
}

// Object-space position -> index-space sample coordinate.
template <typename ValueType>
VISRTX_DEVICE nanovdb::Vec3f nvdbIndexPos(
    const NvdbRegularSamplerState<ValueType> &state, const vec3 &p)
{
  const auto indexPos0 =
      state.grid->worldToIndexF(nanovdb::Vec3f(p.x, p.y, p.z));
  return (indexPos0 - state.offsetDown) * state.scale + state.offsetUp;
}

// Filtered fetch at an index-space coordinate (clamped to the grid).
template <typename ValueType>
VISRTX_DEVICE float nvdbSampleAtIndex(
    const NvdbRegularSamplerState<ValueType> &state, const nanovdb::Vec3f &idx)
{
  const auto c = clampNvdb(idx, state.indexMin, state.indexMax);
  return state.filter == SpatialFieldFilter::Nearest ? state.nearestSampler(c)
                                                      : state.linearSampler(c);
}

// Shared per-sample API (see gpu/volumeIntegrationDetail.h). sampleValue
// returns the field value; sampleNormal returns the unnormalized object-space
// gradient (the raw normal direction) — callers orient and normalize. `field`
// is unused for built-in fields (present for a uniform overload set).
template <typename ValueType>
VISRTX_DEVICE float sampleValue(const NvdbRegularSamplerState<ValueType> &state,
    const SpatialFieldGPUData &,
    const vec3 &p)
{
  return nvdbSampleAtIndex(state, nvdbIndexPos(state, p));
}

template <typename ValueType>
VISRTX_DEVICE vec3 sampleNormal(
    const NvdbRegularSamplerState<ValueType> &state,
    const SpatialFieldGPUData &,
    const vec3 &p)
{
  // Central differences at ±1 voxel in index space, mapped to object space.
  const auto indexPos = nvdbIndexPos(state, p);
  const float sxp = nvdbSampleAtIndex(state, indexPos + nanovdb::Vec3f(1, 0, 0));
  const float sxn = nvdbSampleAtIndex(state, indexPos - nanovdb::Vec3f(1, 0, 0));
  const float syp = nvdbSampleAtIndex(state, indexPos + nanovdb::Vec3f(0, 1, 0));
  const float syn = nvdbSampleAtIndex(state, indexPos - nanovdb::Vec3f(0, 1, 0));
  const float szp = nvdbSampleAtIndex(state, indexPos + nanovdb::Vec3f(0, 0, 1));
  const float szn = nvdbSampleAtIndex(state, indexPos - nanovdb::Vec3f(0, 0, 1));
  return vec3((sxp - sxn) * state.scale[0] * state.invTwoVoxelSize[0],
      (syp - syn) * state.scale[1] * state.invTwoVoxelSize[1],
      (szp - szn) * state.scale[2] * state.invTwoVoxelSize[2]);
}

} // namespace visrtx
