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

// Structured regular-grid sampler — inline device implementations.

#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/shadingState.h"

namespace visrtx {

VISRTX_DEVICE void initStructuredRegularSampler(
    StructuredRegularSamplerState &state, const SpatialFieldGPUData *field)
{
  state.texObj = field->data.structuredRegular.texObj;
  state.origin = field->data.structuredRegular.origin;
  state.invSpacing = field->data.structuredRegular.invSpacing;
  state.offset =
      vec3(field->data.structuredRegular.cellCentered ? 0.0f : 0.5f);
  state.dims = field->data.structuredRegular.dims;
  state.filter = field->data.structuredRegular.filter;
}

// Shared per-sample API (see gpu/volumeIntegrationDetail.h). sampleValue
// returns the field value; sampleNormal returns the unnormalized object-space
// gradient (the raw normal direction) — callers orient and normalize. `field`
// is unused for built-in fields (present for a uniform overload set).
VISRTX_DEVICE float sampleValue(const StructuredRegularSamplerState &state,
    const SpatialFieldGPUData &,
    const vec3 &p)
{
  const auto coords = (p - state.origin) * state.invSpacing + state.offset;
  return tex3D<float>(state.texObj, coords.x, coords.y, coords.z);
}

VISRTX_DEVICE vec3 sampleNormal(const StructuredRegularSamplerState &state,
    const SpatialFieldGPUData &,
    const vec3 &p)
{
  const auto coords = (p - state.origin) * state.invSpacing + state.offset;
  const auto px = coords + vec3(1, 0, 0);
  const auto nx = coords - vec3(1, 0, 0);
  const auto py = coords + vec3(0, 1, 0);
  const auto ny = coords - vec3(0, 1, 0);
  const auto pz = coords + vec3(0, 0, 1);
  const auto nz = coords - vec3(0, 0, 1);

  const float sxp = tex3D<float>(state.texObj, px.x, px.y, px.z);
  const float sxn = tex3D<float>(state.texObj, nx.x, nx.y, nx.z);
  const float syp = tex3D<float>(state.texObj, py.x, py.y, py.z);
  const float syn = tex3D<float>(state.texObj, ny.x, ny.y, ny.z);
  const float szp = tex3D<float>(state.texObj, pz.x, pz.y, pz.z);
  const float szn = tex3D<float>(state.texObj, nz.x, nz.y, nz.z);

  return vec3(sxp - sxn, syp - syn, szp - szn) * state.invSpacing * 0.5f;
}

} // namespace visrtx
