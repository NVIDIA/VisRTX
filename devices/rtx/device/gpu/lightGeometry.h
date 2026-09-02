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

// Shared geometry and radiometry leaves for the analytic AREA lights (rect and
// ring). Every quantity that both next-event estimation and the hit-side
// deposit need lives here exactly once.
//
// Why a separate header (ADR 0009): the hit-side pdf and the NEE pdf must be
// the SAME function, or the MIS balance heuristic is weighted against a density
// nothing actually sampled and the image is silently biased. Two copies that
// "look equivalent" is the failure mode this file exists to make impossible.
//
// This header is deliberately CUDA-FREE — glm and the GPU POD structs only, no
// CUB, no device atomics, no OptiX device intrinsics. sampleLight.h pulls all
// of those, which is why the math it used to own could not be unit tested.
// Keeping these leaves compilable on the host is what lets a unit test assert
// pdf(hit) == pdf(NEE) exactly instead of hoping a rendered image reveals a
// mismatch. Do not add a CUDA-only dependency here.
//
// KNOWN DEFECT, pre-existing and deliberately reproduced: the area terms
// measure the light in OBJECT space while distances are world-space, applying
// no transform Jacobian. Under an instance scale s the reported density is s^2
// times the true one (measured: s=2 gives ratio 4, s=4 gives 16), so a SCALED
// area light's next-event contribution is off by 1/s^2.
//
// Sharing one function across NEE and the hit side does NOT make that
// unbiased -- it only makes the two MIS weights sum to 1 for the wrong
// integral. A diffuse surface reached only by NEE has no competing technique to
// rebalance against and is biased outright.
//
// It is reproduced rather than fixed here because fixing it means changing the
// SAMPLER, which is out of scope for the proxy work and needs its own
// equivalence test against a scaled emissive-surface oracle. What this header
// does guarantee is that the hit side cannot drift from whatever the sampler
// does. Unscaled and rigidly-transformed lights -- the overwhelmingly common
// case, and every case in the test suite -- are exact.
//
// See also the same approximation, documented from the Pick Power side, in
// lightPickPower.h.

#include "gpu/gpu_math.h"
#include "gpu/gpu_objects.h"

namespace visrtx {

// Rect ///////////////////////////////////////////////////////////////////////

// The two derived quantities both sides need, from one cross product: the
// world-space unit normal and the object-space area.
struct RectFrame
{
  // Unit normal of the transformed rect, or exactly zero for a degenerate one
  // (see rectFrame).
  vec3 worldNormal;
  float area; // object-space |edge2 x edge1|, zero for a degenerate rect
};

// The front side is edge2 x edge1, as the ANARI spec defines it for quad
// lights.
//
// The world normal is taken as cross(M*e2, M*e1) -- the normal of the
// TRANSFORMED rectangle -- not as xfmVec(M, cross(e2,e1)).
//
// The two differ by det(M): cross(M*e2, M*e1) = det(M) * M^-T * cross(e2,e1).
// For det(M) > 0 they point the same way, but under a MIRRORING instance
// transform (negative determinant, e.g. a scale with one negative component)
// the forward-transformed normal points into the opposite hemisphere. That
// would make the shared side predicate call the front face the back one, so a
// single-sided light would illuminate the wrong half-space -- and because
// shadow rays never test proxies, nothing downstream would catch it.
//
// Deriving it from the transformed edges also makes this agree with the
// intersector by construction, since the intersector works with the
// world-space rectangle directly.
//
// A degenerate rect -- a zero-length edge, parallel edges, or an instance
// transform that collapses the transformed ones -- yields a zero cross product
// and gets a ZERO normal rather than the NaN normalize() would produce. Zero is
// the answer that stays correct downstream: rectEmissionCosTheta then reports
// cosTheta == 0 for every direction, so every `cosTheta > 0` emission gate
// fails and a light with no area contributes nothing. Relying on NaN to fail
// those same comparisons would work only as long as nothing is built with fast
// math, and would still leak a NaN normal to anything reading the frame.
VISRTX_HOST_DEVICE RectFrame rectFrame(
    const RectLightGPUData &rect, const mat4 &xfm)
{
  RectFrame f;
  f.area = length(cross(rect.edge2, rect.edge1));
  const vec3 worldCross =
      cross(xfmVec(xfm, rect.edge2), xfmVec(xfm, rect.edge1));
  const float worldCrossLength = length(worldCross);
  f.worldNormal =
      worldCrossLength > 0.0f ? worldCross / worldCrossLength : vec3(0.0f);
  return f;
}

// The single `side` predicate. Resolves the light's front/back/both
// configuration against a direction and returns the SIGNED cosine: positive
// means the light emits toward `dirToLight`'s origin, non-positive means it
// does not. Both the NEE radiance/pdf gate and the hit-side cull consume this
// one function, so "which side is lit" cannot be answered two ways.
//
// dirToLight points FROM the shaded point TO the light, matching
// LightSample::dir.
VISRTX_HOST_DEVICE float rectEmissionCosTheta(const RectLightGPUData &rect,
    const vec3 &worldNormal,
    const vec3 &dirToLight)
{
  auto cosTheta = dot(worldNormal, -dirToLight);

  if (rect.side.back) {
    if (rect.side.front)
      cosTheta = fabsf(cosTheta); // Both sides: always positive
    else
      cosTheta = -cosTheta; // Back only: flip to back face
  }
  // Front only: use cosTheta as-is (positive for front face)

  return cosTheta;
}

// Lambertian: radiance is independent of distance and of viewing angle. The
// cosine is carried by the pdf, not by the radiance.
VISRTX_HOST_DEVICE vec3 rectRadiance(
    const RectLightGPUData &rect, const vec3 &color)
{
  return color * rect.intensity;
}

// Uniform-area sampling converted to solid angle: (1/area) * dist^2 / cosTheta.
// Caller must have established cosTheta > 0 via rectEmissionCosTheta.
//
// The operation order here is load-bearing for the "no behavior change"
// requirement of the extraction: it reproduces sampleRectLight's original
// `areaPdf * pow2(dist) / cosTheta` exactly, rather than the algebraically
// equal `pow2(dist) / (area * cosTheta)`, which rounds differently.
VISRTX_HOST_DEVICE float rectSolidAnglePdf(
    float area, float dist, float cosTheta)
{
  const float areaPdf = 1.0f / area;
  return areaPdf * pow2(dist) / cosTheta;
}

// Ring ///////////////////////////////////////////////////////////////////////

// Smoothstep cone falloff, shared so the visible disk shows the same
// attenuation the illumination has. cosTheta is measured against the ring's
// world-space axis.
VISRTX_HOST_DEVICE float ringSpotAttenuation(
    const RingLightGPUData &ring, float cosTheta)
{
  if (cosTheta < ring.cosOuterAngle) {
    // Outside cone: no illumination
    return 0.0f;
  } else if (cosTheta > ring.cosInnerAngle) {
    // Inside inner cone: full illumination
    return 1.0f;
  }
  // Falloff region: smooth interpolation using smoothstep function
  // smoothstep(t) = 3t^2 - 2t^3 provides C1 continuity
  float spot = (cosTheta - ring.cosOuterAngle)
      / (ring.cosInnerAngle - ring.cosOuterAngle);
  return spot * spot * (3.0f - 2.0f * spot);
}

VISRTX_HOST_DEVICE vec3 ringRadiance(
    const RingLightGPUData &ring, const vec3 &color, float spot)
{
  return color * ring.intensity * spot;
}

// Ring area is pi*(R^2 - r^2), precomputed host-side as oneOverArea.
// Same operation order as the original sampleRingLight, per rectSolidAnglePdf.
VISRTX_HOST_DEVICE float ringSolidAnglePdf(
    const RingLightGPUData &ring, float dist, float cosTheta)
{
  const float areaPdf = ring.oneOverArea; // This is 1 / ring_area
  return areaPdf * pow2(dist) / cosTheta;
}

} // namespace visrtx
