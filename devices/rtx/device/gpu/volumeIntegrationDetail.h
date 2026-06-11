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

// Shared helpers + Woodcock-body templates for the per-sampler callables.
// Templates are parameterised on the sampler state type only; they call the
// shared sampleValue / sampleNormal overloads, resolved by ADL on the concrete
// state type, so codegen stays monomorphic per variant.

#include "gpu/gpu_decl.h"
#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"
#include "gpu/gpu_util.h"
#include "gpu/gridTraversal.h"
#include "gpu/sbt.h"
#include "gpu/shadingState.h"

// PCG RNG, see gpu/pcg.h

#include <limits>
#include <type_traits>

namespace visrtx {

VISRTX_DEVICE const SpatialFieldGPUData &getSpatialFieldData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.registry.fields[idx];
}

// Suppresses default ctor for `State` — some sampler states embed members
// with deleted default ctors (nanovdb::ReadAccessor, SampleFromVoxels —
// they hold references). Variant init writes into the raw storage via
// assignment (trivially-copyable members) or placement-new (inner union).
template <typename State>
union SamplerStateBox
{
  // Empty dtor skips State's destruction — fine while sampler states hold
  // only POD / non-owning refs. static_assert breaks the build if anyone
  // adds a non-trivial destructor.
  static_assert(std::is_trivially_destructible_v<State>,
      "SamplerStateBox skips destruction; State must be trivially "
      "destructible.");

  State state;
  VISRTX_DEVICE SamplerStateBox() {}
  VISRTX_DEVICE ~SamplerStateBox() {}
};

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
  // Clamp ceiling at 1 - 2^-24 (largest float strictly less than 1) so
  // 1 - α stays representable for log(). The cell majorant is computed with
  // the same clamp; any non-saturating candidate's σ_t therefore stays ≤
  // σ_maj — load-bearing invariant for the residual ratio-tracking estimator.
  constexpr float OPACITY_CEILING = 1.f - 0x1p-24f;
  const float clampedOpacity = glm::clamp(opacity, 0.f, OPACITY_CEILING);
  if (clampedOpacity <= OPACITY_EPSILON || !(oneOverUnitDistance > 0.f))
    return 0.f;
  return -logf(1.f - clampedOpacity) * oneOverUnitDistance;
}

// Shadow ratio-track RR. Returns true on kill. Survive with
// p = maxAttn / threshold, rescale by 1/p — unbiased.
//
// `ss.shadowContribWeight ∈ (0, 1]` (raygen-set, = maxContrib / RR_BASE)
// raises the threshold for dim rays so RR engages sooner. Per-ray variance
// climbs; per-ray expected work drops faster.
//
// Amplification 1/pSurvive = rrThreshold/maxAttn. RR_MAX_THRESHOLD caps the
// numerator; maxAttn floats with transmittance and can land near
// TRANSMITTANCE_EPSILON → worst-case amplification ~6.7e7 with these
// constants. Unbiased in expectation; downstream firefly filter handles
// the tail.
VISRTX_DEVICE bool applyShadowRussianRoulette(
    vec3 &attenuation, ScreenSample &ss)
{
  constexpr float RR_BASE = 0.5f;
  // Numerator cap. Does NOT bound amplification — see header comment.
  constexpr float RR_MAX_THRESHOLD = 8.0f;
  constexpr float MIN_CONTRIB_WEIGHT = 1.0e-3f;
  constexpr float TRANSMITTANCE_EPSILON =
      std::numeric_limits<float>::epsilon();

  const float w = glm::max(ss.shadowContribWeight, MIN_CONTRIB_WEIGHT);
  const float rrThreshold = glm::min(RR_BASE / w, RR_MAX_THRESHOLD);

  const float maxAttn =
      glm::max(attenuation.x, glm::max(attenuation.y, attenuation.z));
  if (maxAttn <= TRANSMITTANCE_EPSILON) {
    attenuation = vec3(0.0f);
    return true;
  }
  if (maxAttn >= rrThreshold)
    return false;

  const float pSurvive = maxAttn / rrThreshold;
  if (pcg_uniform(&ss.rs) >= pSurvive) {
    attenuation = vec3(0.0f);
    return true;
  }
  attenuation *= (1.0f / pSurvive);
  return false;
}

template <typename State>
VISRTX_DEVICE  vec3 computeWorldNormal(const State &samplerState,
    const SpatialFieldGPUData &field,
    const vec3 &localPos,
    const mat3x4 &worldToObject)
{
  const vec3 localGradient = sampleNormal(samplerState, field, localPos);
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

// ---------------------------------------------------------------------------
// Woodcock-loop body templates. Each sampler family's per-variant callable
// passes an already-inited state; the body calls the shared sampleValue /
// sampleNormal overloads for that state type, so each variant's hot path is
// monomorphic.

// Distance sampling via decomposition tracking (Kutz/Thiery/Novák/Iwasaki 2017):
// inside each macrocell, split σ_t = σ_c + σ_r where σ_c = per-cell min
// extinction (constant lower bound) and σ_r ∈ [0, σ_maj - σ_c]. Race a
// closed-form Exp(σ_c) free-flight against a Woodcock walk on the residual.
// Whichever fires first is the next event; both branches use the local σ_t at
// the event point for albedo/extinction. Degenerates to pure delta tracking
// when σ_c = 0 (sharp TF) and to closed-form analytic flight when σ_r = 0.
template <typename State>
VISRTX_DEVICE  float woodcockSampleDistance(ScreenSample &ss,
    const VolumeHit &hit,
    State &samplerState,
    const SpatialFieldGPUData &field,
    vec3 &albedo,
    float &extinction,
    bool &didScatter,
    vec3 *normal)
{
  const auto &volume = *hit.volume;
  auto &svv = volume.data.tf1d;

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

  GridTraversal trav(objRay, field.grid.dims, field.grid.objectBounds);
  while (trav.valid()) {
    const float2 bounds = __ldg(&field.grid.opacityBounds[trav.cellIndex]);
    const float σ_min =
        opacityToExtinction(bounds.x, svv.oneOverUnitDistance);
    const float σ_maj =
        opacityToExtinction(bounds.y, svv.oneOverUnitDistance);
    const float σ_residual = fmaxf(0.f, σ_maj - σ_min);

    if (σ_maj <= 0.f) {
      trav.next();
      continue;
    }

    // Closed-form control free-flight starting at trav.tEntry.
    const float t_control = σ_min > 0.f
        ? trav.tEntry
            - logf(fmaxf(1e-10f, pcg_uniform(&ss.rs))) / σ_min
        : std::numeric_limits<float>::infinity();
    const float residualCutoff = fminf(t_control, trav.tExit);

    float t = trav.tEntry;
    if (σ_residual > 0.f) {
      // Hoist the reciprocal out of the inner candidate loop — divisions
      // by σ_residual are otherwise on the critical path for every
      // candidate's t-advance and acceptance test.
      const float invSigmaResidual = 1.0f / σ_residual;
      while (t < residualCutoff) {
        t += -logf(fmaxf(1e-10f, pcg_uniform(&ss.rs)))
            * invSigmaResidual;
        if (t >= residualCutoff)
          break;

        const vec3 p = objRay.org + objRay.dir * t;
        const float s = sampleValue(samplerState, field, p);
        if (glm::isnan(s))
          continue;

        const vec4 co = classifySample(volume, s);
        const float σ_t_p =
            opacityToExtinction(co.w, svv.oneOverUnitDistance);
        // σ_t_p ≥ σ_min by construction; residual at p is σ_t_p - σ_min.
        const float σ_r_p = fmaxf(0.f, σ_t_p - σ_min);
        if (σ_r_p * invSigmaResidual > pcg_uniform(&ss.rs)) {
          albedo = vec3(co);
          extinction = σ_t_p;
          didScatter = true;
          scatterPos = p;
          scatterT = t;
          break;
        }
      }
    }

    // If the residual walk didn't fire, accept the control flight if it
    // landed inside the cell. Every control event is real (σ_min is a lower
    // bound on σ_t).
    if (!didScatter && t_control < trav.tExit) {
      const vec3 p = objRay.org + objRay.dir * t_control;
      const float s = sampleValue(samplerState, field, p);
      if (!glm::isnan(s)) {
        const vec4 co = classifySample(volume, s);
        const float σ_t_p =
            opacityToExtinction(co.w, svv.oneOverUnitDistance);
        albedo = vec3(co);
        extinction = σ_t_p;
        didScatter = true;
        scatterPos = p;
        scatterT = t_control;
      }
    }

    if (didScatter)
      break;
    trav.next();
  }

  if (normal && didScatter) {
    *normal = computeWorldNormal(
        samplerState, field, scatterPos, hit.instance->worldToObject);
  }

  return scatterT;
}

// Residual ratio tracking (Novák et al. 2014) over one volume segment.
// Splits σ_t = σ_c + σ_r per cell where σ_c is the per-cell min extinction
// (constant lower bound). The control factor exp(-σ_c · L) is evaluated in
// closed form per cell; the residual factor is ratio-tracked at majorant
// σ_r_maj = σ_maj - σ_c, accumulating attenuation *= (1 - σ_r / σ_r_maj) per
// candidate. Tighter than plain ratio tracking when σ_c > 0; degenerates to
// plain ratio tracking when σ_c = 0.
template <typename State>
VISRTX_DEVICE  void woodcockRatioTrackTransmittance(ScreenSample &ss,
    const VolumeHit &hit,
    State &samplerState,
    const SpatialFieldGPUData &field,
    vec3 &attenuation)
{
  const auto &volume = *hit.volume;
  auto &svv = volume.data.tf1d;

  const Ray objRay = hit.localRay;
  if (!(objRay.t.lower < objRay.t.upper))
    return;

  GridTraversal trav(objRay, field.grid.dims, field.grid.objectBounds);
  while (trav.valid()) {
    const float2 bounds = __ldg(&field.grid.opacityBounds[trav.cellIndex]);
    const float σ_min =
        opacityToExtinction(bounds.x, svv.oneOverUnitDistance);
    const float σ_maj =
        opacityToExtinction(bounds.y, svv.oneOverUnitDistance);
    const float σ_residual = fmaxf(0.f, σ_maj - σ_min);

    if (σ_maj <= 0.f) {
      trav.next();
      continue;
    }

    // Closed-form control factor exp(-σ_min · cell-length) folded into
    // attenuation. No ratio walk needed for this component.
    const float cellLength = fmaxf(0.f, trav.tExit - trav.tEntry);
    if (σ_min > 0.f) {
      attenuation *= __expf(-σ_min * cellLength);
      if (applyShadowRussianRoulette(attenuation, ss))
        return;
    }

    // Ratio-track the residual. Note σ_residual can be zero (uniform cell),
    // in which case the control factor is the complete answer.
    if (σ_residual > 0.f) {
      // Hoist the reciprocal — same rationale as sampleDistance.
      const float invSigmaResidual = 1.0f / σ_residual;
      float t = trav.tEntry;
      while (true) {
        t += -logf(fmaxf(1e-10f, pcg_uniform(&ss.rs)))
            * invSigmaResidual;
        if (t >= trav.tExit)
          break;

        const vec3 p = objRay.org + objRay.dir * t;
        const float s = sampleValue(samplerState, field, p);
        if (glm::isnan(s))
          continue;

        const vec4 co = classifySample(volume, s);
        const float σ_t_p =
            opacityToExtinction(co.w, svv.oneOverUnitDistance);
        const float σ_r_p = fmaxf(0.f, σ_t_p - σ_min);
        const float ratio = glm::clamp(σ_r_p * invSigmaResidual, 0.f, 1.f);
        attenuation *= (1.0f - ratio);

        if (applyShadowRussianRoulette(attenuation, ss))
          return;
      }
    }
    trav.next();
  }
}

// Front-to-back lattice integration (the non-distance-sampling renderers'
// shadow / AOV path). Sample positions across the whole interval lie on a
// fixed dt-spaced lattice; each macrocell continues marching where the
// previous one left off — empty cells are skipped without disturbing the
// lattice phase. Despite living next to the Woodcock-family bodies above,
// this is deterministic emission-absorption ray marching: the macrocell
// `maxOpacity` is used only as an empty-cell flag, never as a majorant for
// null-collision sampling.
template <typename State>
VISRTX_DEVICE  float latticeRayMarchVolume(ScreenSample &ss,
    const VolumeHit &hit,
    State &samplerState,
    const SpatialFieldGPUData &field,
    vec3 *color,
    vec3 *normal,
    float &opacity,
    float invSamplingRate)
{
  const auto &volume = *hit.volume;
  auto &svv = volume.data.tf1d;

  // The local ray direction accounts for instance scaling transformation,
  // meaning it's not a unit vector. We use that to compensate step size.
  const float localDirLen = glm::length(hit.localRay.dir);
  const float localStep = volume.stepSize * invSamplingRate;
  if (localStep <= 0.f || localDirLen <= 0.f)
    return std::numeric_limits<float>::max();
  const float dt = localStep / localDirLen;
  const float exponent = dt * svv.oneOverUnitDistance;

  const Ray objRay = hit.localRay;
  if (!(objRay.t.lower < objRay.t.upper))
    return std::numeric_limits<float>::max();

  float depth = std::numeric_limits<float>::max();

  constexpr float MIN_OPACITY_THRESHOLD = 1e-2f;
  // Early-out once the segment is effectively opaque. Kept moderately high so
  // the residual transmittance zeroed below is genuinely negligible.
  constexpr float MAX_OPACITY_THRESHOLD = 0.999f;

  // Single stratified jitter at the segment start.
  const float jitter =
      pcg_uniform(&ss.rs) * fminf(dt, objRay.t.upper - objRay.t.lower);
  float nextSampleT = objRay.t.lower + jitter;

  GridTraversal trav(objRay, field.grid.dims, field.grid.objectBounds);
  while (trav.valid()) {
    if (opacity >= MAX_OPACITY_THRESHOLD)
      break;

    const float2 bounds = __ldg(&field.grid.opacityBounds[trav.cellIndex]);
    const float maxOpacity = bounds.y;
    if (maxOpacity <= 0.f) {
      trav.next();
      continue;
    }

    // Skip forward on the global sample lattice to the first sample inside
    // this cell.
    if (nextSampleT < trav.tEntry) {
      const float advance = ceilf((trav.tEntry - nextSampleT) / dt);
      nextSampleT += advance * dt;
    }

    while (nextSampleT < trav.tExit) {
      if (opacity >= MAX_OPACITY_THRESHOLD)
        break;
      const vec3 p = objRay.org + objRay.dir * nextSampleT;

      const float s = sampleValue(samplerState, field, p);
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
            depth = nextSampleT;
        }
      }

      nextSampleT += dt;
    }
    trav.next();
  }

  // The early-out treats the segment as opaque, but front-to-back compositing
  // leaves a small residual transmittance (1 - opacity). Against a low-dynamic-
  // range background it's invisible; against an HDR background (bright sky, sun)
  // even ~0.1% leaks visibly and makes the volume look more transparent than it
  // should be.
  if (opacity >= MAX_OPACITY_THRESHOLD)
    opacity = 1.0f;

  if (normal) {
    *normal = vec3(0.f);
    if (depth < std::numeric_limits<float>::max()) {
      const vec3 p = objRay.org + objRay.dir * depth;
      *normal = computeWorldNormal(
          samplerState, field, p, hit.instance->worldToObject);
    }
  }

  return depth;
}

} // namespace detail
} // namespace visrtx
