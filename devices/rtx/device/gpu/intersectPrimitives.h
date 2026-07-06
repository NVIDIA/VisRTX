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

// Pure analytic primitive intersectors (sphere, cylinder, cone), independent of
// OptiX/CUDA so they compile & run on the host for unit testing. Each
// forEach*Crossing enumerates a solid's boundary crossings (entry AND exit) and
// emits them straight into `report`, unclipped by tmin/tmax and in no
// particular order (no per-thread hit buffer to spill in the IS hot path). The
// OptiX IS wrapper reports every crossing — entry and exit — tagging each with
// its facing (see reportCrossing); OptiX keeps the nearest in [tmin, tmax], so
// interiors and transmission exits resolve like triangle back faces. The
// solve* array overloads collect all crossings for host unit tests. Normals
// are outward and unit. Cap enablement is a per-endpoint bitmask (bit0 = the
// p0 end / "first", bit1 = the p1 end / "second").
//
// Capped cylinders/cones are convex solids: a ray has at most 2 true boundary
// crossings. Up to 4 *candidates* exist before span/disk clipping and
// coincident-t de-dup (2 quadratic roots + 2 cap-plane hits), which sizes the
// solve* output arrays.

#pragma once

#include "gpu/gpu_decl.h"

#include <glm/glm.hpp>

namespace visrtx {

using glm::vec3;

struct PrimHit
{
  float t;
  vec3 Ng; // outward, unit
  float u; // attribute0 parameter
};

enum CapBit : uint8_t
{
  CAP_FIRST = 1u << 0, // p0 end
  CAP_SECOND = 1u << 1, // p1 end
};

namespace detail {

constexpr float kFltMax = 3.402823466e38f;

// Relative slack for degeneracy gates (dimensionless; compared against a
// quantity's natural squared scale).
constexpr float kRelEps = 1e-6f;

// Coincident-t de-dup: ~32 float ulps, relative to the larger |t| of the
// compared pair. Tight enough that genuine entry/exit crossings of thin
// primitives stay distinct, loose enough to merge a tangent/rim corner counted
// by two surface patches.
constexpr float kDedupRelEps = 4e-6f;

// True for finite floats only: the comparison is false for both inf and NaN
// (NaN compares false to everything). Works identically on host and device
// without pulling in <cmath>/isfinite intrinsics.
VISRTX_HOST_DEVICE bool isFiniteF(float x)
{
  return glm::abs(x) <= kFltMax;
}

VISRTX_HOST_DEVICE bool isFiniteVec(const vec3 &v)
{
  return isFiniteF(v.x) && isFiniteF(v.y) && isFiniteF(v.z);
}

// Coincident-t filter held in scalar slots — no dynamically indexed array, so
// it stays in registers inside an OptiX IS program. Three slots cover the
// worst case of four candidate crossings.
struct DedupT
{
  float ta{kFltMax}, tb{kFltMax}, tc{kFltMax};

  VISRTX_HOST_DEVICE static bool coincides(float t, float prev)
  {
    return glm::abs(t - prev)
        <= kDedupRelEps * glm::max(glm::abs(t), glm::abs(prev));
  }

  VISRTX_HOST_DEVICE bool insert(float t)
  {
    if (coincides(t, ta) || coincides(t, tb) || coincides(t, tc))
      return false;
    tc = tb;
    tb = ta;
    ta = t;
    return true;
  }
};

// Emit one crossing: drop non-finite results and any that coincide (in t) with
// one already emitted — a tangent/graze must count once, not twice (duplicate
// anyhit reports would double-count opacity).
template <typename ReportFn>
VISRTX_HOST_DEVICE void emitCrossing(
    DedupT &dedup, ReportFn &&report, const PrimHit &h)
{
  if (!isFiniteF(h.t) || !isFiniteVec(h.Ng))
    return;
  if (!dedup.insert(h.t))
    return;
  report(h);
}

// Ray vs a flat cap disk: plane through pc with normal `axis` (need not be
// unit), disk radius `radius`. axis2 = dot(axis,axis) and dir2 = dot(rd,rd)
// are passed in (both callers already have them). `outwardN` is the unit
// outward normal for this end. Divides by sd=dot(rd,axis) — guarded against a
// perpendicular ray.
VISRTX_HOST_DEVICE bool capDiskHit(const vec3 &ro,
    const vec3 &rd,
    const vec3 &pc,
    const vec3 &axis,
    float axis2,
    float dir2,
    float radius,
    const vec3 &outwardN,
    float u,
    PrimHit &h)
{
  const float sd = glm::dot(rd, axis);
  if (sd * sd <= kRelEps * axis2 * dir2) // ray parallel to the cap plane
    return false;
  const float t = glm::dot(pc - ro, axis) / sd;
  const vec3 rel = (ro + t * rd) - pc; // lies in the cap plane
  if (glm::dot(rel, rel) > radius * radius)
    return false;
  h = PrimHit{t, outwardN, u};
  return true;
}

} // namespace detail

// Sphere: emits both roots (the IS wrapper keeps the front one). u is unused.
// Roots are the closest-approach parameter tmid = -b/a plus/minus a half-chord
// thalf = sqrt((r^2 - |perp|^2)/a), with perp = oc + tmid*rd the ray-to-center
// rejection. This never forms c = |oc|^2 - r^2 (which cancels catastrophically
// when the origin is near a large sphere's surface, |oc| ~ r — the source of
// the concentric shading rings on large ground spheres): the near-root normal
// there routes through c/q and its lost precision bands the normal. The perp
// form of the discriminant also avoids the b^2 - a*c cancellation of small
// distant spheres, and tmid -/+ thalf is exact to the inputs off grazing rays
// (thalf << |tmid|), cancelling only benignly at grazing (Sterbenz).
template <typename ReportFn>
VISRTX_HOST_DEVICE void forEachSphereCrossing(const vec3 &ro,
    const vec3 &rd,
    const vec3 &center,
    float radius,
    ReportFn &&report)
{
  using namespace detail;
  radius = glm::abs(radius);
  const vec3 oc = ro - center;
  const float a = glm::dot(rd, rd);
  if (a <= 0.f)
    return;
  const float b = glm::dot(oc, rd); // half-b of a t^2 + 2b t + c = 0
  const float tmid = -b / a; // parameter of closest approach to center
  const vec3 perp = oc + tmid * rd; // ray-to-center rejection at tmid
  const float halfChord2 = radius * radius - glm::dot(perp, perp);
  if (halfChord2 < 0.f)
    return;
  const float thalf = glm::sqrt(halfChord2 / a);
  const float t0 = tmid - thalf;
  const float t1 = tmid + thalf;
  DedupT dedup;
  emitCrossing(
      dedup, report, PrimHit{t0, (ro + t0 * rd - center) / radius, 0.f});
  emitCrossing(
      dedup, report, PrimHit{t1, (ro + t1 * rd - center) / radius, 0.f});
}

// Cylinder: curved wall (within the axis span) + enabled flat caps. radius is
// treated as a magnitude. u along the wall is the axial fraction; caps use
// 0/1. The wall quadratic is recentered at the ray's closest approach to the
// axis (fp) before forming the discriminant — forming it at the ray origin
// cancels catastrophically for distant origins.
template <typename ReportFn>
VISRTX_HOST_DEVICE void forEachCylinderCrossing(const vec3 &ro,
    const vec3 &rd,
    const vec3 &p0,
    const vec3 &p1,
    float radius,
    uint8_t capBits,
    ReportFn &&report)
{
  using namespace detail;
  const vec3 s = p1 - p0; // axis
  const float s2 = glm::dot(s, s);
  const float d2 = glm::dot(rd, rd);
  radius = glm::abs(radius);
  if (s2 <= 0.f || d2 <= 0.f) // degenerate (p0==p1) or null ray
    return;

  DedupT dedup;

  const vec3 f = p0 - ro;
  const vec3 sxd = glm::cross(s, rd);
  const float a = glm::dot(sxd, sxd); // s2*d2*sin^2(axis,rd)
  if (a > kRelEps * s2 * d2) { // skip when ray ~parallel to axis (a->0)
    const float ra = 1.f / a;
    const float ts = glm::dot(sxd, glm::cross(s, f)) * ra; // closest approach
    const vec3 fp = f - ts * rd; // p0 relative to the closest-approach point
    const vec3 perp = glm::cross(s, fp);
    const float cc = radius * radius * s2 - glm::dot(perp, perp);
    if (cc >= 0.f) {
      const float td = glm::sqrt(cc * ra);
      const float tw[2] = {ts - td, ts + td};
      for (int i = 0; i < 2; ++i) {
        const vec3 h = ro + tw[i] * rd;
        const float u = glm::dot(h - p0, s) / s2;
        if (u >= 0.f && u <= 1.f)
          emitCrossing(
              dedup, report, PrimHit{tw[i], (h - (p0 + u * s)) / radius, u});
      }
    }
  }

  if (capBits) {
    const vec3 axisN = s * glm::inversesqrt(s2);
    PrimHit h;
    if ((capBits & CAP_FIRST)
        && capDiskHit(ro, rd, p0, s, s2, d2, radius, -axisN, 0.f, h))
      emitCrossing(dedup, report, h);
    if ((capBits & CAP_SECOND)
        && capDiskHit(ro, rd, p1, s, s2, d2, radius, axisN, 1.f, h))
      emitCrossing(dedup, report, h);
  }
}

// Cone (truncated): tapered body (both roots, within the axis span) + enabled
// flat caps sized to each endpoint radius. Radii treated as magnitudes. The
// body solve normalizes rd so it is correct under non-unit (instance-scaled)
// directions; t is rescaled back on output.
template <typename ReportFn>
VISRTX_HOST_DEVICE void forEachConeCrossing(const vec3 &ro,
    const vec3 &rd,
    const vec3 &p0,
    const vec3 &p1,
    float r0,
    float r1,
    uint8_t capBits,
    ReportFn &&report)
{
  using namespace detail;
  const float ra = glm::abs(r0);
  const float rb = glm::abs(r1);
  const vec3 ba = p1 - p0;
  const float m0 = glm::dot(ba, ba);
  const float d2 = glm::dot(rd, rd);
  if (m0 <= 0.f || d2 <= 0.f) // degenerate axis or null ray
    return;
  const float rlen = glm::sqrt(d2);

  const vec3 rdn = rd / rlen; // unit direction; scale t back by 1/rlen
  const vec3 oa = ro - p0;
  const float m1 = glm::dot(oa, ba);
  const float m3 = glm::dot(rdn, ba);
  const float m4 = glm::dot(rdn, oa);
  const float m5 = glm::dot(oa, oa);
  const float rr = ra - rb;
  const float hy = m0 + rr * rr;

  // k2 t^2 + 2 k1 t + k0 = 0 (rd unit here).
  const float k2 = m0 * m0 - m3 * m3 * hy;
  const float k1 = m0 * m0 * m4 - m1 * m3 * hy + m0 * ra * rr * m3;
  const float k0 =
      m0 * m0 * m5 - m1 * m1 * hy + m0 * ra * (rr * m1 * 2.0f - m0 * ra);

  float tn0 = 0.f, tn1 = 0.f;
  int nroots = 0;
  const float k2scale = glm::max(m0 * m0, glm::abs(m3 * m3 * hy));
  if (glm::abs(k2) <= kRelEps * k2scale) {
    // Near-parallel to the slant generator: 2 k1 t + k0 = 0. Gate |k1| against
    // the magnitudes of its own terms (same units, any scene scale) — the root
    // is only unreliable when k1 itself is a catastrophic cancellation.
    const float k1scale = glm::max(glm::abs(m0 * m0 * m4),
        glm::max(glm::abs(m1 * m3 * hy), glm::abs(m0 * ra * rr * m3)));
    if (glm::abs(k1) > kRelEps * k1scale) {
      tn0 = -k0 / (2.0f * k1);
      nroots = 1;
    }
  } else {
    const float h = k1 * k1 - k2 * k0;
    if (h >= 0.f) {
      // Stable pairing (qq/k2, k0/qq): avoids the -k1 + sqrt cancellation.
      const float qq = -(k1 + (k1 >= 0.f ? 1.f : -1.f) * glm::sqrt(h));
      tn0 = qq / k2;
      tn1 = k0 / qq; // qq==0 -> non-finite -> dropped by emitCrossing
      nroots = 2;
    }
  }

  DedupT dedup;

  const auto emitBody = [&](float tn) {
    const float y = m1 + tn * m3;
    if (y > 0.f && y < m0) {
      const vec3 nrm = glm::normalize(
          m0 * (m0 * (oa + tn * rdn) + rr * ba * ra) - ba * hy * y);
      emitCrossing(dedup, report, PrimHit{tn / rlen, nrm, y / m0});
    }
  };
  if (nroots > 0)
    emitBody(tn0);
  if (nroots > 1)
    emitBody(tn1);

  // Flat caps sized per endpoint (magnitude radii).
  if (capBits) {
    const vec3 axisN = ba * glm::inversesqrt(m0);
    PrimHit hh;
    if ((capBits & CAP_FIRST) && ra > 0.f
        && capDiskHit(ro, rd, p0, ba, m0, d2, ra, -axisN, 0.f, hh))
      emitCrossing(dedup, report, hh);
    if ((capBits & CAP_SECOND) && rb > 0.f
        && capDiskHit(ro, rd, p1, ba, m0, d2, rb, axisN, 1.f, hh))
      emitCrossing(dedup, report, hh);
  }
}

// Array-collecting overloads for host unit tests (and any caller that wants
// the crossings materialized). Emission counts are bounded by the candidate
// counts documented above, so the fixed-size outputs cannot overflow.

VISRTX_HOST_DEVICE int solveSphere(const vec3 &ro,
    const vec3 &rd,
    const vec3 &center,
    float radius,
    PrimHit out[2])
{
  int n = 0;
  forEachSphereCrossing(
      ro, rd, center, radius, [&](const PrimHit &h) { out[n++] = h; });
  return n;
}

VISRTX_HOST_DEVICE int solveCylinder(const vec3 &ro,
    const vec3 &rd,
    const vec3 &p0,
    const vec3 &p1,
    float radius,
    uint8_t capBits,
    PrimHit out[4])
{
  int n = 0;
  forEachCylinderCrossing(
      ro, rd, p0, p1, radius, capBits, [&](const PrimHit &h) { out[n++] = h; });
  return n;
}

VISRTX_HOST_DEVICE int solveCone(const vec3 &ro,
    const vec3 &rd,
    const vec3 &p0,
    const vec3 &p1,
    float r0,
    float r1,
    uint8_t capBits,
    PrimHit out[4])
{
  int n = 0;
  forEachConeCrossing(
      ro, rd, p0, p1, r0, r1, capBits, [&](const PrimHit &h) { out[n++] = h; });
  return n;
}

} // namespace visrtx
