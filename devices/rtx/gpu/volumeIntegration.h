/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <texture_types.h>
#include "gpu/dda.h"
#include "gpu/gpu_debug.h"
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"
#include "gpu/sampleSpatialField.h"
#include "nanovdb/NanoVDB.h"

namespace visrtx {

namespace detail {

VISRTX_DEVICE vec4 classifySample(const VolumeGPUData &v, float s)
{
  vec4 retval(0.f);
  switch (v.type) {
  case VolumeType::TF1D: {
    float coord = position(s, v.data.tf1d.valueRange);
    retval = make_vec4(tex1D<::float4>(v.data.tf1d.tfTex, coord));
    retval.w *= v.data.tf1d.densityScale;
    break;
  }
  default:
    break;
  }
  return retval;
}

template <typename Sampler>
VISRTX_DEVICE void _rayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    box1 interval,
    vec3 *color,
    float &opacity,
    float invSamplingRate)
{
  const auto &volume = *hit.volumeData;
  /////////////////////////////////////////////////////////////////////////////
  // TODO: need to generalize
  auto &svv = volume.data.tf1d;
  auto &field = getSpatialFieldData(*ss.frameData, svv.field);
  /////////////////////////////////////////////////////////////////////////////

  Sampler sampler(field);

  const float stepSize = volume.stepSize * invSamplingRate;
  interval.lower += stepSize * curand_uniform(&ss.rs); // jitter

  while (opacity < 0.99f && size(interval) >= 0.f) {
    const vec3 p = hit.localRay.org + hit.localRay.dir * interval.lower;

    const float s = sampler(p);
    if (!glm::isnan(s)) {
      const vec4 co = detail::classifySample(volume, s);
      if (color)
        accumulateValue(*color, vec3(co) * co.w * invSamplingRate, opacity);
      accumulateValue(opacity, co.w * invSamplingRate, opacity);
    }

    interval.lower += stepSize;
  }
}

VISRTX_DEVICE float rayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 *color,
    float &opacity,
    float invSamplingRate)
{
  const auto &volume = *hit.volumeData;
  /////////////////////////////////////////////////////////////////////////////
  // TODO: need to generalize
  auto &svv = volume.data.tf1d;
  auto &field = getSpatialFieldData(*ss.frameData, svv.field);
  /////////////////////////////////////////////////////////////////////////////
  const float stepSize = volume.stepSize;
  box1 interval = hit.localRay.t;
  const float depth = interval.lower;
  interval.lower += stepSize * curand_uniform(&ss.rs); // jitter

  switch (field.type) {
  case SpatialFieldType::STRUCTURED_REGULAR: {
    _rayMarchVolume<SpatialFieldSampler<cudaTextureObject_t>>(
        ss, hit, interval, color, opacity, invSamplingRate);
    break;
  }
  case SpatialFieldType::NANOVDB_REGULAR: {
    switch (field.data.nvdbRegular.gridType) {
    case nanovdb::GridType::Fp4: {
      _rayMarchVolume<NvdbSpatialFieldSampler<nanovdb::Fp4>>(
          ss, hit, interval, color, opacity, invSamplingRate);
      break;
    }
    case nanovdb::GridType::Fp8: {
      _rayMarchVolume<NvdbSpatialFieldSampler<nanovdb::Fp8>>(
          ss, hit, interval, color, opacity, invSamplingRate);
      break;
    }
    case nanovdb::GridType::Fp16: {
      _rayMarchVolume<NvdbSpatialFieldSampler<nanovdb::Fp16>>(
          ss, hit, interval, color, opacity, invSamplingRate);
      break;
    }
    case nanovdb::GridType::FpN: {
      _rayMarchVolume<NvdbSpatialFieldSampler<nanovdb::FpN>>(
          ss, hit, interval, color, opacity, invSamplingRate);
      break;
    }
    case nanovdb::GridType::Float: {
      _rayMarchVolume<NvdbSpatialFieldSampler<float>>(
          ss, hit, interval, color, opacity, invSamplingRate);
      break;
    }
    default:
      break;
    }
    break;
  }
  default:
    break;
  }

  return depth;
}

template <typename Sampler>
VISRTX_DEVICE float _sampleDistance(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 *albedo,
    float &extinction,
    float &tr)
{
  const auto &volume = *hit.volumeData;
  /////////////////////////////////////////////////////////////////////////////
  // TODO: need to generalize
  auto &svv = volume.data.tf1d;
  auto &field = getSpatialFieldData(*ss.frameData, svv.field);
  /////////////////////////////////////////////////////////////////////////////

  Sampler sampler(field);

  const float stepSize = volume.stepSize;
  float t_out = hit.localRay.t.upper;
  tr = 1.f;

  Ray objRay = hit.localRay;
  objRay.org += hit.localRay.dir * hit.localRay.t.lower;
  objRay.t.lower -= hit.localRay.t.lower;
  objRay.t.upper -= hit.localRay.t.lower;

  auto woodcockFunc = [&](const int leafID, float t0, float t1) {
    const float majorant = field.grid.maxOpacities[leafID];
    float t = t0;

    while (1) {
      if (majorant <= 0.f)
        break;

      t -= logf(1.f - curand_uniform(&ss.rs)) / majorant * stepSize;

      if (t >= t1)
        break;

      const vec3 p =
          hit.localRay.org + hit.localRay.dir * (t + hit.localRay.t.lower);
      const float s = sampler(p);
      if (!glm::isnan(s)) {
        const vec4 co = detail::classifySample(volume, s);
        *albedo = vec3(co);
        extinction = co.w;
        float u = curand_uniform(&ss.rs);
        if (extinction >= u * majorant) {
          tr = 0.f;
          t_out = t;
          return false; // stop traversal
        }
      }
    }

    return true; // cont. traversal to the next spat. partition
  };

  if (debug()) {
    printf("DDA with ray org: (%f,%f,%f), dir: (%f,%f,%f), [t0,t1]: %f,%f\n",
        objRay.org.x,
        objRay.org.y,
        objRay.org.z,
        objRay.dir.x,
        objRay.dir.y,
        objRay.dir.z,
        hit.localRay.t.lower,
        hit.localRay.t.upper);
  }

  dda3(objRay, field.grid.dims, field.grid.worldBounds, woodcockFunc);
  return t_out + hit.localRay.t.lower;
}

VISRTX_DEVICE float sampleDistance(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 *albedo,
    float &extinction,
    float &tr)
{
  const auto &volume = *hit.volumeData;
  /////////////////////////////////////////////////////////////////////////////
  // TODO: need to generalize
  auto &svv = volume.data.tf1d;
  auto &field = getSpatialFieldData(*ss.frameData, svv.field);
  /////////////////////////////////////////////////////////////////////////////

  switch (field.type) {
  case SpatialFieldType::STRUCTURED_REGULAR: {
    return _sampleDistance<SpatialFieldSampler<cudaTextureObject_t>>(
        ss, hit, albedo, extinction, tr);
    break;
  }
  case SpatialFieldType::NANOVDB_REGULAR: {
    switch (field.data.nvdbRegular.gridType) {
    case nanovdb::GridType::Fp4: {
      return _sampleDistance<NvdbSpatialFieldSampler<nanovdb::Fp4>>(
          ss, hit, albedo, extinction, tr);
      break;
    }
    case nanovdb::GridType::Fp8: {
      return _sampleDistance<NvdbSpatialFieldSampler<nanovdb::Fp8>>(
          ss, hit, albedo, extinction, tr);
      break;
    }
    case nanovdb::GridType::Fp16: {
      return _sampleDistance<NvdbSpatialFieldSampler<nanovdb::Fp16>>(
          ss, hit, albedo, extinction, tr);
      break;
    }
    case nanovdb::GridType::FpN: {
      return _sampleDistance<NvdbSpatialFieldSampler<nanovdb::FpN>>(
          ss, hit, albedo, extinction, tr);
      break;
    }
    case nanovdb::GridType::Float: {
      return _sampleDistance<NvdbSpatialFieldSampler<float>>(
          ss, hit, albedo, extinction, tr);
      break;
    }
    default:
      break;
    }
    break;
  }
  default:
    break;
  }

  return hit.localRay.t.upper;
}

} // namespace detail

VISRTX_DEVICE float rayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    float &opacity,
    float invSamplingRate)
{
  return detail::rayMarchVolume(ss, hit, nullptr, opacity, invSamplingRate);
}

VISRTX_DEVICE float rayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 &color,
    float &opacity,
    float invSamplingRate)
{
  return detail::rayMarchVolume(ss, hit, &color, opacity, invSamplingRate);
}

template <typename RAY_TYPE>
VISRTX_DEVICE float rayMarchAllVolumes(ScreenSample &ss,
    Ray ray,
    RAY_TYPE type,
    float tfar,
    float invSamplingRate,
    vec3 &color,
    float &opacity,
    uint32_t &objID,
    uint32_t &instID)
{
  VolumeHit hit;
  ray.t.upper = tfar;
  float depth = tfar;
  bool firstHit = true;

  do {
    hit.foundHit = false;
    intersectVolume(ss, ray, type, &hit);
    if (!hit.foundHit)
      break;
    else if (firstHit) {
      objID = hit.volumeData->id;
      instID = hit.instID;
      firstHit = false;
    }
    depth = min(depth, hit.localRay.t.lower);
    hit.localRay.t.upper = glm::min(tfar, hit.localRay.t.upper);
    detail::rayMarchVolume(ss, hit, &color, opacity, invSamplingRate);
    ray.t.lower = hit.localRay.t.upper + 1e-3f;
  } while (opacity < 0.99f);

  return depth;
}

template <typename RAY_TYPE>
VISRTX_DEVICE float sampleDistanceAllVolumes(ScreenSample &ss,
    Ray ray,
    RAY_TYPE type,
    float tfar,
    vec3 &albedo,
    float &extinction,
    float &transmittance,
    uint32_t &objID,
    uint32_t &instID)
{
  VolumeHit hit;
  ray.t.upper = tfar;
  float depth = tfar;
  transmittance = 1.f;

  while (true) {
    hit.foundHit = false;
    intersectVolume(ss, ray, type, &hit);
    if (!hit.foundHit)
      break;
    hit.localRay.t.upper = glm::min(tfar, hit.localRay.t.upper);
    vec3 alb(0.f);
    float ext = 0.f, tr = 0.f;
    float d = detail::sampleDistance(ss, hit, &alb, ext, tr);
    if (d < depth) {
      depth = d;
      albedo = alb;
      extinction = ext;
      transmittance = tr;
      objID = hit.volumeData->id;
      instID = hit.instID;
    }
    ray.t.lower = hit.localRay.t.upper + 1e-3f;
  }

  return depth;
}

} // namespace visrtx
