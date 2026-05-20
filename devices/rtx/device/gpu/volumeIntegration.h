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

#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"
#include "gpu/gridTraversal.h"
#include "gpu/shadingState.h"

// cuda
#include <texture_types.h>

// nanovdb
#include <nanovdb/NanoVDB.h>

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

VISRTX_DEVICE void volumeSamplerInit(
    VolumeSamplingState *samplerState, const SpatialFieldGPUData &field)
{
  optixDirectCall<void>(uint32_t(field.samplerCallableIndex)
          + static_cast<uint32_t>(SpatialFieldSamplerEntryPoints::Init),
      samplerState,
      &field);
}

VISRTX_DEVICE float volumeSamplerSample(const VolumeSamplingState *samplerState,
    const SpatialFieldGPUData &field,
    const vec3 &position)
{
  return optixDirectCall<float>(uint32_t(field.samplerCallableIndex)
          + static_cast<uint32_t>(SpatialFieldSamplerEntryPoints::Sample),
      samplerState,
      &position,
      (vec3 *)nullptr);
}

VISRTX_DEVICE float volumeSamplerSampleWithGradient(
    const VolumeSamplingState *samplerState,
    const SpatialFieldGPUData &field,
    const vec3 &position,
    vec3 *gradient)
{
  return optixDirectCall<float>(uint32_t(field.samplerCallableIndex)
          + static_cast<uint32_t>(SpatialFieldSamplerEntryPoints::Sample),
      samplerState,
      &position,
      gradient);
}

namespace detail {

VISRTX_DEVICE vec4 classifySample(const VolumeGPUData &v, float s)
{
  vec4 retval(0.f);
  switch (v.type) {
  case VolumeType::TF1D: {
    if (v.data.tf1d.tfTex) {
      float coord = position(s, v.data.tf1d.valueRange);
      retval = make_vec4(tex1D<::float4>(v.data.tf1d.tfTex, coord));
    } else
      retval = vec4(v.data.tf1d.uniformColor, v.data.tf1d.uniformOpacity);
    break;
  }
  default:
    break;
  }
  return retval;
}

VISRTX_DEVICE float opacityToExtinction(
    float opacity, float oneOverUnitDistance)
{
  constexpr float OPACITY_EPSILON = 1e-7f;
  // Ceiling = largest float < 1 (1 - 2⁻²⁴). Keeps 1-α representable for
  // log() and pins σ_t ≤ σ_maj (cell majorant uses same clamp) — invariant
  // for the residual ratio-tracking estimator.
  constexpr float OPACITY_CEILING = 1.f - 0x1p-24f;
  const float clampedOpacity = glm::clamp(opacity, 0.f, OPACITY_CEILING);
  if (clampedOpacity <= OPACITY_EPSILON || !(oneOverUnitDistance > 0.f))
    return 0.f;
  return -logf(1.f - clampedOpacity) * oneOverUnitDistance;
}

VISRTX_DEVICE vec3 computeWorldNormal(const VolumeSamplingState *samplerState,
    const SpatialFieldGPUData &field,
    const vec3 &localPos,
    const mat3x4 &worldToObject)
{
  vec3 localGradient(0.f);
  volumeSamplerSampleWithGradient(
      samplerState, field, localPos, &localGradient);
  constexpr float MIN_GRADIENT_LENGTH_SQ = 1e-12f;
  if (glm::dot(localGradient, localGradient) <= MIN_GRADIENT_LENGTH_SQ)
    return vec3(0.f);
  const mat3 normalXfm = glm::transpose(mat3(worldToObject));
  const vec3 worldNormal = normalXfm * (-localGradient);
  const float worldNormalLength = glm::length(worldNormal);
  constexpr float MIN_WORLD_NORMAL_LENGTH = 1e-6f;
  if (worldNormalLength <= MIN_WORLD_NORMAL_LENGTH)
    return vec3(0.f);
  return worldNormal * (1.f / worldNormalLength);
}

VISRTX_DEVICE float rayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 *color,
    vec3 *normal,
    float &opacity,
    float invSamplingRate)
{
  const auto &volume = *hit.volume;
  auto &svv = volume.data.tf1d;
  auto &field = getSpatialFieldData(*ss.frameData, svv.field);

  VolumeSamplingState samplerState;
  volumeSamplerInit(&samplerState, field);

  // The local ray direction is actually accounting for instance scaling
  // transformation, meaning it's not a unit vector.
  // We need to use that to compensate our step size.
  const float localDirLen = glm::length(hit.localRay.dir);
  const float localStep = volume.stepSize * invSamplingRate;
  const float dt = localStep / localDirLen;
  const float exponent = dt * svv.oneOverUnitDistance;
  if (localStep <= 0.f)
    return std::numeric_limits<float>::max();

  box1 interval = hit.localRay.t;
  interval.lower +=
      curand_uniform(&ss.rs) * min(dt, interval.upper - interval.lower);

  float depth = std::numeric_limits<float>::max();

  constexpr float MIN_OPACITY_THRESHOLD = 1e-2f;
  constexpr float MAX_OPACITY_THRESHOLD = 0.99f;
  while (opacity < MAX_OPACITY_THRESHOLD && size(interval) >= 0.f) {
    const vec3 p = hit.localRay.org + hit.localRay.dir * interval.lower;

    const float s = volumeSamplerSample(&samplerState, field, p);
    if (!glm::isnan(s)) {
      const vec4 co = classifySample(volume, s);

      const float stepAlpha = 1.0f - glm::pow(1.0f - co.w, exponent);
      if (stepAlpha > 0.0f) {
        const float weight = (1.0f - opacity);
        if (color)
          *color += weight * stepAlpha * vec3(co);
        opacity += weight * stepAlpha;

        if (opacity > MIN_OPACITY_THRESHOLD
            && depth == std::numeric_limits<float>::max())
          depth = interval.lower;
      }
    }

    interval.lower += dt;
  }

  if (normal) {
    *normal = vec3(0.f);
    if (depth < std::numeric_limits<float>::max()) {
      const vec3 p = hit.localRay.org + hit.localRay.dir * depth;
      *normal = computeWorldNormal(&samplerState, field, p, hit.worldToObject);
    }
  }

  return depth;
}

VISRTX_DEVICE float sampleDistance(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 &albedo,
    float &extinction,
    bool &didScatter,
    vec3 *normal = nullptr)
{
  const auto &volume = *hit.volume;
  auto &svv = volume.data.tf1d;
  auto &field = getSpatialFieldData(*ss.frameData, svv.field);

  VolumeSamplingState samplerState;
  volumeSamplerInit(&samplerState, field);

  albedo = vec3(0.f);
  extinction = 0.f;
  didScatter = false;
  if (normal)
    *normal = vec3(0.f);

  float scatterT = hit.localRay.t.upper;
  vec3 scatterPos(0.f);
  const Ray objRay = hit.localRay;
  if (!(objRay.t.lower < objRay.t.upper))
    return scatterT;

  GridTraversal trav(objRay, field.grid.dims, field.grid.worldBounds);
  while (trav.valid()) {
    const float maxOpacity = field.grid.maxOpacities[trav.cellIndex];
    const float majorantExtinction =
        opacityToExtinction(maxOpacity, svv.oneOverUnitDistance);

    if (majorantExtinction > 0.f) {
      constexpr int MAX_WOODCOCK_STEPS_PER_CELL = 128;
      int steps = 0;
      float t = trav.tEntry;
      while (t < trav.tExit) {
        if (++steps > MAX_WOODCOCK_STEPS_PER_CELL)
          break;
        t += -logf(fmaxf(1e-10f, 1.f - curand_uniform(&ss.rs)))
            / majorantExtinction;

        if (t >= trav.tExit)
          break;

        const vec3 p = objRay.org + objRay.dir * t;
        const float s = volumeSamplerSample(&samplerState, field, p);

        if (!glm::isnan(s)) {
          const vec4 co = detail::classifySample(volume, s);
          const float actualExtinction =
              opacityToExtinction(co.w, svv.oneOverUnitDistance);
          if (actualExtinction > 0.f
              && actualExtinction
                  >= curand_uniform(&ss.rs) * majorantExtinction) {
            albedo = vec3(co);
            extinction = actualExtinction;
            didScatter = true;
            scatterPos = p;
            scatterT = t;
            break;
          }
        }
      }
    }

    if (didScatter)
      break;
    trav.next();
  }

  if (normal && didScatter) {
    *normal =
        computeWorldNormal(&samplerState, field, scatterPos, hit.worldToObject);
  }

  return scatterT;
}

// Unbiased ratio-tracking transmittance estimator over one volume segment.
// At each Woodcock candidate, multiply attenuation by (1 - sigma_t / sigma_maj)
// rather than terminating on a "real" event. Across multiple any-hit
// invocations on adjacent volume AABBs, OptiX composes the per-segment
// estimates multiplicatively via the shared payload, so this is the right
// primitive to call from __anyhit__shadow.
VISRTX_DEVICE void ratioTrackTransmittance(
    ScreenSample &ss, const VolumeHit &hit, vec3 &attenuation)
{
  const auto &volume = *hit.volume;
  auto &svv = volume.data.tf1d;
  auto &field = getSpatialFieldData(*ss.frameData, svv.field);

  const Ray objRay = hit.localRay;
  if (!(objRay.t.lower < objRay.t.upper))
    return;

  // Match Quality_ptx.cu's ATTENUATION_EPSILON.
  constexpr float TRANSMITTANCE_EPSILON =
      std::numeric_limits<float>::epsilon();

  VolumeSamplingState samplerState;
  volumeSamplerInit(&samplerState, field);

  GridTraversal trav(objRay, field.grid.dims, field.grid.worldBounds);
  while (trav.valid()) {
    const float maxOpacity = field.grid.maxOpacities[trav.cellIndex];
    const float majorantExtinction =
        opacityToExtinction(maxOpacity, svv.oneOverUnitDistance);

    if (majorantExtinction > 0.f) {
      float t = trav.tEntry;
      while (true) {
        t += -logf(fmaxf(1e-10f, 1.f - curand_uniform(&ss.rs)))
            / majorantExtinction;
        if (t >= trav.tExit)
          break;

        const vec3 p = objRay.org + objRay.dir * t;
        const float s = volumeSamplerSample(&samplerState, field, p);
        if (glm::isnan(s))
          continue;

        const vec4 co = classifySample(volume, s);
        const float actualExtinction =
            opacityToExtinction(co.w, svv.oneOverUnitDistance);
        const float ratio =
            glm::clamp(actualExtinction / majorantExtinction, 0.f, 1.f);
        attenuation *= (1.0f - ratio);

        if (glm::all(
                glm::lessThanEqual(attenuation, vec3(TRANSMITTANCE_EPSILON))))
          return;
      }
    }
    trav.next();
  }
}

} // namespace detail

VISRTX_DEVICE float sampleDistanceVolume(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 &albedo,
    float &extinction,
    bool &didScatter,
    vec3 *normal = nullptr)
{
  return detail::sampleDistance(
      ss, hit, albedo, extinction, didScatter, normal);
}

VISRTX_DEVICE void ratioTrackTransmittanceVolume(
    ScreenSample &ss, const VolumeHit &hit, vec3 &attenuation)
{
  detail::ratioTrackTransmittance(ss, hit, attenuation);
}

VISRTX_DEVICE float rayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    float &opacity,
    float invSamplingRate)
{
  return detail::rayMarchVolume(
      ss, hit, nullptr, nullptr, opacity, invSamplingRate);
}

VISRTX_DEVICE float rayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    vec3 &color,
    float &opacity,
    float invSamplingRate)
{
  return detail::rayMarchVolume(
      ss, hit, &color, nullptr, opacity, invSamplingRate);
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
    float thisDepth = detail::rayMarchVolume(ss,
        hit,
        &color,
        normal ? &thisNormal : nullptr,
        opacity,
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
    float d = detail::sampleDistance(
        ss, hit, alb, ext, segmentDidScatter, normal ? &norm : nullptr);
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
