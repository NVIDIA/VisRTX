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

#include <glm/gtc/matrix_inverse.hpp>

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

// The geometric terms relating a shading point to a point on a rect light.
struct RectPointRelation
{
  vec3 dir; // unit, from the shading point TO the light
  float dist; // world-space distance
  float cosTheta; // signed by the side predicate; <= 0 means not emitting
  float solidAnglePdf; // 0 when cosTheta <= 0
};

// THE shared density function (ADR 0009).
//
// Given a shading point and a point on the light, produce the direction,
// distance, emission cosine and solid-angle density. NEE calls this with the
// point it sampled; the hit-side deposit calls it with the point the ray hit.
// One function, so the two densities cannot drift and the MIS balance heuristic
// stays honest.
//
// The frame travels as one value because its two halves are only ever produced
// together by rectFrame() and only ever consumed together here -- splitting
// them at the call site invited passing a normal from one rect with the area of
// another.
//
// RectFrame::area is the light's OBJECT-space area while worldPoint and origin
// are world-space. That asymmetry is the sampler's pre-existing behavior,
// reproduced rather than corrected -- see the header note.
VISRTX_HOST_DEVICE RectPointRelation rectRelateToPoint(
    const RectLightGPUData &rect,
    const RectFrame &frame,
    const vec3 &origin,
    const vec3 &worldPoint)
{
  RectPointRelation r;
  r.dir = worldPoint - origin;
  r.dist = length(r.dir);
  r.dir /= r.dist;
  r.cosTheta = rectEmissionCosTheta(rect, frame.worldNormal, r.dir);
  r.solidAnglePdf = r.cosTheta > 0.0f
      ? rectSolidAnglePdf(frame.area, r.dist, r.cosTheta)
      : 0.0f;
  return r;
}

// Analytic ray/rect intersection //////////////////////////////////////////////

struct RectIntersection
{
  bool hit;
  float t; // distance along the (not necessarily unit) ray direction
  vec2 uv; // parametric position within the rect, both in [0,1] on a hit
};

// Ray against the light's rectangle, in the rect's own coordinate space.
//
// Reports the parametric uv directly so the pdf leaf can consume it with no
// reconstruction step: recomputing the hit position or the rect's area on the
// deposit side would be a second chance to disagree with the sampler, which is
// exactly what ADR 0009 exists to prevent.
//
// The uv solve uses the Gram matrix rather than the dot(d,e1)/|e1|^2 shortcut,
// because edge1 and edge2 need not be perpendicular. ANARI places no such
// constraint on a quad light, and the shortcut silently mis-bounds a sheared
// parallelogram (accepting points outside it and rejecting points inside).
VISRTX_HOST_DEVICE RectIntersection intersectRect(
    const RectLightGPUData &rect, const vec3 &org, const vec3 &dir)
{
  RectIntersection out;
  out.hit = false;
  out.t = 0.0f;
  out.uv = vec2(0.0f);

  const vec3 e1 = rect.edge1;
  const vec3 e2 = rect.edge2;
  const vec3 normal = cross(e1, e2);

  const float denom = dot(normal, dir);
  // Ray parallel to (or lying in) the plane: no well-defined single crossing.
  // Exactly zero rather than an epsilon — a near-parallel ray still has a
  // genuine, if distant, intersection, and t is checked by the caller.
  if (denom == 0.0f)
    return out;

  const float t = dot(normal, rect.position - org) / denom;
  if (!(t > 0.0f))
    return out; // behind the origin, or exactly at it

  const vec3 d = (org + t * dir) - rect.position;

  const float e11 = dot(e1, e1);
  const float e12 = dot(e1, e2);
  const float e22 = dot(e2, e2);
  const float det = e11 * e22 - e12 * e12;
  // Degenerate rect (zero-length or parallel edges): no area, never hit. This
  // also guards the divide below.
  if (!(det > 0.0f))
    return out;

  const float d1 = dot(d, e1);
  const float d2 = dot(d, e2);
  const float invDet = 1.0f / det;
  const float u = (d1 * e22 - d2 * e12) * invDet;
  const float v = (d2 * e11 - d1 * e12) * invDet;

  if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f)
    return out;

  out.hit = true;
  out.t = t;
  out.uv = vec2(u, v);
  return out;
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

// The ring's world-space axis: the UNIT normal of the TRANSFORMED disk.
//
// A plane normal transforms by the inverse transpose, not by the matrix
// itself. Under a rotation or a uniform scale the two agree; under a
// non-uniform scale a ring whose axis is not aligned with a scale axis gets a
// forward-transformed direction that is NOT perpendicular to the transformed
// disk. The intersector would then solve against a plane tilted away from the
// surface NEE samples, and the cone falloff would be measured against the wrong
// axis on both sides. (rectFrame sidesteps the same trap by taking the cross
// product of the transformed edges.)
//
// Orientation comes for free: (M^-T d) . (M d) = d . d > 0, so the emitting
// hemisphere keeps following the forward-transformed direction, mirroring
// included.
//
// Normalizing AFTER the transform is load-bearing: cosTheta is computed as
// dot(axis, -dir) with dir already unit, so a non-unit axis scales the cosine
// and silently shifts the cone thresholds (cosInnerAngle/cosOuterAngle) for any
// scaled instance.
VISRTX_HOST_DEVICE vec3 ringWorldAxis(
    const RingLightGPUData &ring, const mat4 &xfm)
{
  return normalize(
      glm::inverseTranspose(mat3(xfm)) * normalize(ring.direction));
}

// Ring counterpart of rectRelateToPoint: THE shared density function for ring
// lights, called by NEE with the point it sampled and by the hit-side deposit
// with the point the ray hit.
//
// Unlike rect, a ring carries a cone falloff, so this also returns the spot
// attenuation -- the visible disk must show the same falloff the illumination
// has, which it gets by calling the same leaf rather than a second copy.
struct RingPointRelation
{
  vec3 dir; // unit, from the shading point TO the light
  float dist;
  float cosTheta; // against the ring axis; <= 0 means not emitting
  float spot; // cone attenuation in [0,1]
  float solidAnglePdf; // 0 unless both spot > 0 and cosTheta > 0
};

VISRTX_HOST_DEVICE RingPointRelation ringRelateToPoint(
    const RingLightGPUData &ring,
    const vec3 &worldAxis,
    const vec3 &origin,
    const vec3 &worldPoint)
{
  RingPointRelation r;
  r.dir = worldPoint - origin;
  r.dist = length(r.dir);
  r.dir /= r.dist;
  r.cosTheta = dot(worldAxis, -r.dir);
  r.spot = ringSpotAttenuation(ring, r.cosTheta);
  r.solidAnglePdf = (r.spot > 0.0f && r.cosTheta > 0.0f)
      ? ringSolidAnglePdf(ring, r.dist, r.cosTheta)
      : 0.0f;
  return r;
}

// Sphere
// ///////////////////////////////////////////////////////////////////////

// An ANARI `point` light with radius > 0 becomes LightType::SPHERE: a real area
// light, sampled uniformly over the whole sphere surface by sampleSphereLight.

struct SpherePointRelation
{
  vec3 dir; // unit, from the shading point TO the light
  float dist;
  float cosTheta; // against the OUTWARD surface normal at the point
  float solidAnglePdf; // 0 when the sampled element faces away
};

VISRTX_HOST_DEVICE vec3 sphereRadiance(
    const SphereLightGPUData &sphere, const vec3 &color)
{
  return color * sphere.intensity;
}

// Uniform-area sampling over the sphere's whole surface (4*pi*r^2) converted to
// solid angle. Matches sampleSphereLight's operation order exactly.
//
// Like rect/ring this uses the OBJECT-space radius with world-space distances
// and no transform Jacobian -- the same documented approximation, reproduced so
// the hit side cannot drift from the sampler.
VISRTX_HOST_DEVICE SpherePointRelation sphereRelateToPoint(
    const SphereLightGPUData &sphere,
    const vec3 &worldCenter,
    const vec3 &origin,
    const vec3 &worldPoint)
{
  SpherePointRelation r;
  r.dir = worldPoint - origin;
  r.dist = length(r.dir);
  r.dir /= r.dist;

  const vec3 surfaceNormal = normalize(worldPoint - worldCenter);
  r.cosTheta = dot(surfaceNormal, -r.dir);

  if (r.cosTheta > 0.0f) {
    const float areaPdf = 1.f / (4.f * kPi * sphere.radius * sphere.radius);
    r.solidAnglePdf = areaPdf * pow2(r.dist) / r.cosTheta;
  } else
    r.solidAnglePdf = 0.0f;

  return r;
}

struct SphereIntersection
{
  bool hit;
  float t;
};

// Ray/sphere, nearest positive root. The proxy only ever needs the visible
// (front) surface: a sphere light's interior is not a place a deposit can come
// from, and the exit crossing would be behind the emitting surface anyway.
//
// Solved in the light's OBJECT frame: the world ray is pulled back through
// worldToObject and tested against the object-space sphere. sampleSphereLight
// pushes its object-space points forward through the same transform, so the
// hittable set is exactly the samplable set under ANY affine instance
// transform -- a non-uniformly scaled sphere is an ellipsoid, and no
// world-space sphere of any single radius matches it. The parametric t is
// invariant under the affine map (origin and direction transform together), so
// the returned t is directly the world-ray t.
VISRTX_HOST_DEVICE SphereIntersection intersectSphereLight(
    const SphereLightGPUData &sphere,
    const mat4 &worldToObject,
    const vec3 &worldOrg,
    const vec3 &worldDir)
{
  SphereIntersection out;
  out.hit = false;
  out.t = 0.0f;

  const vec3 org = xfmPoint(worldToObject, worldOrg);
  const vec3 dir = xfmVec(worldToObject, worldDir);

  const vec3 oc = org - sphere.position;
  const float a = dot(dir, dir);
  if (!(a > 0.0f))
    return out;
  const float b = dot(oc, dir);
  const float c = dot(oc, oc) - sphere.radius * sphere.radius;
  const float disc = b * b - a * c;
  if (disc < 0.0f)
    return out;

  const float sq = sqrtf(disc);
  // Nearest positive root; fall through to the far one when the origin is
  // inside.
  float t = (-b - sq) / a;
  if (!(t > 0.0f))
    t = (-b + sq) / a;
  if (!(t > 0.0f))
    return out;

  out.hit = true;
  out.t = t;
  return out;
}

// Analytic ray/ring intersection //////////////////////////////////////////////

struct RingIntersection
{
  bool hit;
  float t;
  float radius; // distance from the ring centre, in [innerRadius, radius]
};

// Ray against the ring's annulus: plane intersection, then a radial band test.
// The inner hole must MISS -- it is the ring's analogue of the rectangle's edge
// bounds, and a ring rendered as a full disk is the obvious failure.
//
// Solved in the light's OBJECT frame, like intersectSphereLight: the world ray
// is pulled back through worldToObject and tested against the object-space
// annulus with the object radii and the object axis. sampleRingLight pushes
// its object-space points forward through the same transform, so the hittable
// set is exactly the samplable set under any affine instance transform. A
// non-uniformly scaled disk is an ellipse, and neither a world-space circle of
// averaged radius (the previous approach) nor a forward-transformed normal
// describes it; in object space there is nothing to approximate. `radius` is
// reported in object units.
VISRTX_HOST_DEVICE RingIntersection intersectRing(const RingLightGPUData &ring,
    const mat4 &worldToObject,
    const vec3 &worldOrg,
    const vec3 &worldDir)
{
  RingIntersection out;
  out.hit = false;
  out.t = 0.0f;
  out.radius = 0.0f;

  // An annulus with no area never hits. Without this the radial band test
  // below, which is inclusive at both edges so a normal ring's rim is not a
  // seam, would accept the single radius where a collapsed ring's outer edge
  // and inner hole coincide -- reporting a hit on a light that emits nothing.
  // Mirrors Ring::hasAreaProxy on the host, which keeps the same rings out of
  // the proxy BLAS.
  if (!(ring.radius > ring.innerRadius))
    return out;

  const vec3 org = xfmPoint(worldToObject, worldOrg);
  const vec3 dir = xfmVec(worldToObject, worldDir);
  const vec3 axis = normalize(ring.direction);

  const float denom = dot(axis, dir);
  if (denom == 0.0f)
    return out; // parallel to (or lying in) the ring's plane

  const float t = dot(axis, ring.position - org) / denom;
  if (!(t > 0.0f))
    return out;

  const vec3 p = org + t * dir;
  const vec3 radial = p - ring.position;
  // Distance from the axis, not from the centre: the hit lies in the plane, so
  // these agree, but subtracting the axial component keeps it exact under fp
  // error near grazing incidence.
  const vec3 inPlane = radial - axis * dot(radial, axis);
  const float r = length(inPlane);

  if (r > ring.radius || r < ring.innerRadius)
    return out; // outside the outer edge, or through the inner hole

  out.hit = true;
  out.t = t;
  out.radius = r;
  return out;
}

} // namespace visrtx
