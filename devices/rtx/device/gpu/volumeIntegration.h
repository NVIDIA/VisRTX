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

// Renderer-side volume integration dispatch. Heavy Woodcock bodies live as
// per-sampler direct-callables in `spatial_field/*Sampler_ptx.cu`; wrappers
// here are `optixDirectCall` shims (one call per ray segment).
//
// RAY_TYPE-templated outer loops stay here — they invoke
// `intersectVolume<RAY_TYPE>` then dispatch per-segment via the same shim.

#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"
#include "gpu/intersectRay.h"
#include "gpu/sbt.h"
#include "gpu/shadingState.h"

// optix
#include <optix_device.h>
#include <limits>

namespace visrtx {

// Helpers //
VISRTX_DEVICE const SpatialFieldGPUData &getSpatialFieldData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.registry.fields[idx];
}

// ---------------------------------------------------------------------------
// Per-segment dispatch wrappers. Resolve sampler callable base offset,
// invoke per-variant Woodcock body. One direct-call amortised over the
// 10–100 candidates running inside the callable.
// ---------------------------------------------------------------------------

VISRTX_DEVICE float sampleDistanceVolume(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 &albedo,
    float &extinction,
    bool &didScatter,
    vec3 *normal = nullptr)
{
  const auto &field =
      getSpatialFieldData(*ss.frameData, hit.volume->data.tf1d.field);
  return optixDirectCall<float>(uint32_t(field.samplerCallableIndex)
          + uint32_t(SpatialFieldSamplerEntryPoints::SampleDistance),
      &ss,
      &hit,
      &albedo,
      &extinction,
      &didScatter,
      normal);
}

VISRTX_DEVICE void ratioTrackTransmittanceVolume(
    ScreenSample &ss, const VolumeHit &hit, vec3 &attenuation)
{
  const auto &field =
      getSpatialFieldData(*ss.frameData, hit.volume->data.tf1d.field);
  optixDirectCall<void>(uint32_t(field.samplerCallableIndex)
          + uint32_t(SpatialFieldSamplerEntryPoints::RatioTrackTransmittance),
      &ss,
      &hit,
      &attenuation);
}

VISRTX_DEVICE float rayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    float &opacity,
    float invSamplingRate)
{
  const auto &field =
      getSpatialFieldData(*ss.frameData, hit.volume->data.tf1d.field);
  return optixDirectCall<float>(uint32_t(field.samplerCallableIndex)
          + uint32_t(SpatialFieldSamplerEntryPoints::RayMarchVolume),
      &ss,
      &hit,
      (vec3 *)nullptr,
      (vec3 *)nullptr,
      &opacity,
      invSamplingRate);
}

VISRTX_DEVICE float rayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 &color,
    float &opacity,
    float invSamplingRate)
{
  const auto &field =
      getSpatialFieldData(*ss.frameData, hit.volume->data.tf1d.field);
  return optixDirectCall<float>(uint32_t(field.samplerCallableIndex)
          + uint32_t(SpatialFieldSamplerEntryPoints::RayMarchVolume),
      &ss,
      &hit,
      &color,
      (vec3 *)nullptr,
      &opacity,
      invSamplingRate);
}

// ---------------------------------------------------------------------------
// RAY_TYPE-templated outer loops. Surface / shadow / AOV pass each picks
// up its own `intersectVolume<RAY_TYPE>` overload; per-segment dispatch
// goes through the wrappers above.
// ---------------------------------------------------------------------------

template <typename RAY_TYPE>
VISRTX_DEVICE float rayMarchAllVolumes(ScreenSample &ss,
    Ray ray,
    RAY_TYPE type,
    float tfar,
    float invSamplingRate,
    vec3 &color,
    float &opacity,
    uint32_t &objID,
    uint32_t &instID,
    vec3 *normal = nullptr)
{
  VolumeHit hit;
  ray.t.upper = tfar;
  float depth = std::numeric_limits<float>::max();
  if (normal)
    *normal = vec3(0.f);

  constexpr float OPACITY_THRESHOLD = 0.99f;

  do {
    hit.foundHit = false;
    intersectVolume(ss, ray, type, &hit);
    if (!hit.foundHit)
      break;

    // Save where this volume ends, so
    hit.localRay.t.upper = glm::min(tfar, hit.localRay.t.upper);

    // Ray march through this volume segment
    vec3 thisNormal(0.f);
    const auto &field =
        getSpatialFieldData(*ss.frameData, hit.volume->data.tf1d.field);
    float thisDepth =
        optixDirectCall<float>(uint32_t(field.samplerCallableIndex)
                + uint32_t(SpatialFieldSamplerEntryPoints::RayMarchVolume),
            &ss,
            &hit,
            &color,
            normal ? &thisNormal : (vec3 *)nullptr,
            &opacity,
            invSamplingRate);

    // Track closest intersection depth
    if (thisDepth < depth) {
      depth = thisDepth;
      objID = hit.volume->id;
      instID = hit.instance->id;
      if (normal)
        *normal = thisNormal;
    }

    if (ray.t.lower < hit.localRay.t.upper)
      ray.t.lower = hit.localRay.t.upper;
    else
      break;

  } while (opacity < OPACITY_THRESHOLD);

  return depth;
}

// Samples the first accepted Woodcock event across intersected volume segments.
template <typename RAY_TYPE>
VISRTX_DEVICE float sampleDistanceAllVolumes(ScreenSample &ss,
    Ray ray,
    RAY_TYPE type,
    float tfar,
    vec3 &albedo,
    float &extinction,
    bool &didScatter,
    uint32_t &objID,
    uint32_t &instID,
    vec3 *normal = nullptr)
{
  VolumeHit hit;
  ray.t.upper = tfar;
  float depth = tfar;
  albedo = vec3(0.f);
  extinction = 0.f;
  didScatter = false;
  objID = ~0u;
  instID = ~0u;
  if (normal)
    *normal = vec3(0.f);

  while (true) {
    hit.foundHit = false;
    intersectVolume(ss, ray, type, &hit);
    if (!hit.foundHit)
      break;
    hit.localRay.t.upper = glm::min(tfar, hit.localRay.t.upper);
    vec3 alb(0.f);
    vec3 norm(0.f);
    float ext = 0.f;
    bool segmentDidScatter = false;
    const auto &fld =
        getSpatialFieldData(*ss.frameData, hit.volume->data.tf1d.field);
    float d = optixDirectCall<float>(uint32_t(fld.samplerCallableIndex)
            + uint32_t(SpatialFieldSamplerEntryPoints::SampleDistance),
        &ss,
        &hit,
        &alb,
        &ext,
        &segmentDidScatter,
        normal ? &norm : (vec3 *)nullptr);
    if (segmentDidScatter) {
      depth = d;
      albedo = alb;
      extinction = ext;
      didScatter = true;
      objID = hit.volume->id;
      instID = hit.instance->id;
      if (normal)
        *normal = norm;
      break;
    }

    if (ray.t.lower < hit.localRay.t.upper)
      ray.t.lower = hit.localRay.t.upper;
    else
      break;
  }

  return depth;
}

} // namespace visrtx
