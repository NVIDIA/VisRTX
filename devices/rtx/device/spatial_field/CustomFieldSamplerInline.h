// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Custom spatial-field per-sample API. Unlike the built-in inline samplers,
// the user sample() implementation is compiled into the value/normal
// direct-callables (CustomFieldSampler_ptx.cu), so sampling routes back through
// the SBT via optixDirectCall (callable-in-callable). Living in a header lets
// both the volume integrator (CustomFieldSampler_ptx.cu) and the isosurface
// intersector (Intersectors_ptx.cu) resolve sampleValue/sampleNormal on
// VolumeSamplingState by ADL.

#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/sbt.h"
#include "gpu/shadingState.h"

#include <optix_device.h>

namespace visrtx {

VISRTX_DEVICE void initSamplerState(
    VolumeSamplingState &s, const SpatialFieldGPUData &field)
{
  optixDirectCall<void>(uint32_t(field.samplerCallableIndex)
          + uint32_t(SpatialFieldSamplerEntryPoints::Init),
      &s,
      &field);
}

VISRTX_DEVICE float sampleValue(const VolumeSamplingState &s,
    const SpatialFieldGPUData &field,
    const vec3 &p)
{
  return optixDirectCall<float>(uint32_t(field.samplerCallableIndex)
          + uint32_t(SpatialFieldSamplerEntryPoints::SampleValue),
      &s,
      &field,
      &p);
}

VISRTX_DEVICE vec3 sampleNormal(const VolumeSamplingState &s,
    const SpatialFieldGPUData &field,
    const vec3 &p)
{
  return optixDirectCall<vec3>(uint32_t(field.samplerCallableIndex)
          + uint32_t(SpatialFieldSamplerEntryPoints::SampleNormal),
      &s,
      &field,
      &p);
}

} // namespace visrtx
