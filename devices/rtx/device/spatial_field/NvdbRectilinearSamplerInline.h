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

  // NanoVDB's index space spans [0, N] over the world bbox (N voxels as cells),
  // mapped to the axis LUT's [0,1] domain by scaleDown then back by scaleUp.
  // offsetDown is +min (min-relative): a grid whose index bbox does not start at
  // 0 (any real index-centered NanoVDB grid) otherwise drives the normalized
  // coord negative, the LUT clamps it, and every sample collapses to a constant.
  // span = N-1, clamped to >=1 so a degenerate 1-node axis can't make scaleUp 0
  // (the DDA's boundaryPos divides by scaleUp). A 1-node rectilinear axis is
  // already rejected upstream — the axis LUT needs >= 2 coords — so this is just
  // defensive against NaN.
  const nanovdb::Vec3f span(dims[0] > 1.f ? dims[0] - 1.f : 1.f,
      dims[1] > 1.f ? dims[1] - 1.f : 1.f,
      dims[2] > 1.f ? dims[2] - 1.f : 1.f);
  state.scaleUp = span;
  state.offsetDown = nanovdb::Vec3f(indexBBox.min());
  if (field->data.nvdbRectilinear.cellCentered) {
    // Cell-centered: cells fill [0,N], no stretch. scaleDown 1/(N-1) cancels
    // scaleUp (N-1) to an identity LUT map; the -0.5 offset samples cell centers.
    state.scaleDown = nanovdb::Vec3f(1.0f) / span;
    state.offsetUp = nanovdb::Vec3f(-0.5f) + state.offsetDown;
  } else {
    // Node-centered: N nodes fill only [0, N-1] of the [0,N] domain. scaleDown
    // 1/N then scaleUp (N-1) stretches the full domain onto the node range
    // (symmetric half voxels); 1/(N-1) would leave a full constant voxel at the
    // high boundary.
    state.scaleDown = nanovdb::Vec3f(1.0f) / dims;
    state.offsetUp = state.offsetDown;
  }
  state.indexMin = nanovdb::Vec3f(indexBBox.min());
  state.indexMax = nanovdb::Vec3f(indexBBox.max());

  state.axisLUT[0] = field->data.nvdbRectilinear.axisLUT[0];
  state.axisLUT[1] = field->data.nvdbRectilinear.axisLUT[1];
  state.axisLUT[2] = field->data.nvdbRectilinear.axisLUT[2];
  state.invAxisLUT[0] = field->data.nvdbRectilinear.invAxisLUT[0];
  state.invAxisLUT[1] = field->data.nvdbRectilinear.invAxisLUT[1];
  state.invAxisLUT[2] = field->data.nvdbRectilinear.invAxisLUT[2];

  const auto &iavs = field->data.nvdbRectilinear.invAvgVoxelSize;
  state.invAvgVoxelSize = nanovdb::Vec3f(iavs.x, iavs.y, iavs.z);

  // Per-axis affine uniform-index -> world map for the isosurface DDA (assumes
  // an axis-aligned grid map; a rotated/sheared map would need the full 3x3).
  const auto w0 = grid->indexToWorldF(nanovdb::Vec3f(0.f, 0.f, 0.f));
  const auto wx = grid->indexToWorldF(nanovdb::Vec3f(1.f, 0.f, 0.f));
  const auto wy = grid->indexToWorldF(nanovdb::Vec3f(0.f, 1.f, 0.f));
  const auto wz = grid->indexToWorldF(nanovdb::Vec3f(0.f, 0.f, 1.f));
  state.worldOrigin = vec3(w0[0], w0[1], w0[2]);
  state.worldVoxelStep = vec3(wx[0] - w0[0], wy[1] - w0[1], wz[2] - w0[2]);
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

// Forward-difference gradient at a linear isosurface hit (see the regular
// sampler): reuse vHit (≈ the matched isovalue) for 3 +1-voxel taps vs 6. Per-
// axis invAvgVoxelSize weighting matches the central difference (its *0.5 and
// the 1- vs 2-voxel span both wash out under the caller's normalize).
template <typename ValueType>
VISRTX_DEVICE vec3 isosurfaceHitGradient(
    const NvdbRectilinearSamplerState<ValueType> &state,
    const SpatialFieldGPUData &,
    const vec3 &p,
    float vHit)
{
  const auto indexPos = worldToIndexRectilinear(state, &p);
  const float sx =
      sampleAtIndexRectilinear(state, indexPos + nanovdb::Vec3f(1, 0, 0));
  const float sy =
      sampleAtIndexRectilinear(state, indexPos + nanovdb::Vec3f(0, 1, 0));
  const float sz =
      sampleAtIndexRectilinear(state, indexPos + nanovdb::Vec3f(0, 0, 1));
  return vec3(sx - vHit, sy - vHit, sz - vHit)
      * vec3(state.invAvgVoxelSize[0],
          state.invAvgVoxelSize[1],
          state.invAvgVoxelSize[2]);
}

} // namespace visrtx
