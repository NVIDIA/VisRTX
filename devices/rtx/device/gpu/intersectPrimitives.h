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

#include <cmath>

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

// Noise floor for the quadratic discriminant, relative to its operands' scale
// (~32 fp32 ulps). Near tangency the discriminant is a catastrophic
// cancellation of ~equal squared terms, so below this floor its SIGN is fp
// noise. That matters for rays grazing their own origin surface: secondary
// rays are lifted only ~1e-6*|P| off the surface (epsilonFrom), which puts the
// true discriminant (~ -2*r*lift for a sphere of radius r) inside the noise
// band, and a false-positive root pair yields a phantom back-facing EXIT
// crossing at t up to ~1e-3*r — far beyond any tmin, self-shadowing the
// primitive (concentric acne rings on large ground spheres). Exit crossings
// are therefore only emitted when the discriminant clears this floor; entry
// crossings stay ungated (a phantom entry lies behind the origin and is culled
// by tmin, a genuine graze entry shades correctly). The cost is real exits
// within ~sqrt(kGrazeRelEps) ≈ 0.1 degree of tangency — visually nil.
constexpr float kGrazeRelEps = 4e-6f;

// True for finite floats only: the comparison is false for both inf and NaN
// (NaN compares false to everything). Works identically on host and device
// without pulling in <cmath>/isfinite intrinsics.
VISRTX_HOST_DEVICE bool isFiniteF(float x)
{
  return glm::abs(x) <= kFltMax;
}

// Difference of products a*b - c*d, accurate to ~1.5 ulp even when a*b ~= c*d
// (Kahan's FMA algorithm; pbrt-v4 DifferenceOfProducts). fma is correctly
// rounded on both host and device and is NOT degraded by --use_fast_math, so
// this bound survives fast-math. This is the fix for catastrophic cancellation
// in the quadratic discriminant (k1^2 - k2*k0), which near tangency / a sharp
// cone taper is a subtraction of nearly-equal squared terms — the root of the
// cone-apex speckle. `err` is the exact rounding error of the product c*d.
VISRTX_HOST_DEVICE float differenceOfProducts(float a, float b, float c, float d)
{
  const float cd = c * d;
  const float err = fmaf(-c, d, cd);
  return fmaf(a, b, -cd) + err;
}

// Correctly-rounded sqrt / divide, immune to --use_fast_math's .approx lowering
// (fast-math implies -prec-sqrt=false/-prec-div=false, degrading sqrtf/`/` to
// ~1-2 ulp; these intrinsics stay 0 ulp). Used only on the few
// cancellation-critical operations of the quadratic solve; host maps to the
// default round-to-nearest ops.
VISRTX_HOST_DEVICE float sqrtExact(float x)
{
#if defined(__CUDA_ARCH__)
  return __fsqrt_rn(x);
#else
  return glm::sqrt(x);
#endif
}

VISRTX_HOST_DEVICE float divExact(float a, float b)
{
#if defined(__CUDA_ARCH__)
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
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
  // t1 is always the exit; gate it on the discriminant noise floor (see
  // kGrazeRelEps) so a graze of the ray's own origin sphere can't emit a
  // phantom self-shadowing exit.
  if (halfChord2 > kGrazeRelEps * radius * radius)
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
      // tw[1] is always the exit; gate it on the discriminant noise floor (see
      // kGrazeRelEps) so a graze of the ray's own origin wall can't emit a
      // phantom self-shadowing exit.
      const int nw = cc > kGrazeRelEps * radius * radius * s2 ? 2 : 1;
      for (int i = 0; i < nw; ++i) {
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
// directions; t is rescaled back on output. The quadratic is apex-free (radius
// varies linearly along the axis, so r0==r1 reduces continuously to a cylinder
// with no apex construction) and formulated for fp32: the ray origin is
// re-centered at its closest approach to the segment before the coefficients
// are built (so they scale with local geometry, not the ray's distance from the
// primitive), the primitive is then prescaled to ~unit by an exact power of two
// (so the m0^2-scaled coefficients stay in fp32's precise range at any absolute
// size), the discriminant uses the cancellation-free difference-of-products,
// and the cancellation-critical sqrt/divides use correctly-rounded intrinsics.
// See .scratch/intersection-grounds/derivations.md.
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

  // Re-center the origin at the ray's closest approach to the segment midpoint
  // (unit-space offset `s`): makes every coefficient O(primitive extent) rather
  // than O(|ro|), removing distant-origin cancellation. The shift is
  // self-correcting, so it need not be exact.
  const vec3 mid = 0.5f * (p0 + p1);
  const float s = glm::dot(mid - ro, rdn);
  const vec3 oaU = (ro + s * rdn) - p0;

  // Exact power-of-two prescale so the primitive is ~unit-sized. The
  // m0^2-scaled coefficients below reach ~|extent|^6, which overflows fp32's
  // precise range for large primitives (and underflows into the ftz-flush zone
  // for tiny ones); scaling every length by an exact power of two keeps them in
  // range, and the offset root is un-scaled by the same factor at t conversion.
  // log2f need only be approximate — it only picks the integer exponent.
  const float refLen = glm::max(glm::sqrt(m0), glm::max(ra, rb));
  const float sigma = exp2f(-roundf(log2f(refLen)));
  const vec3 oa = oaU * sigma;
  const vec3 baS = ba * sigma;
  const float raS = ra * sigma;
  const float rbS = rb * sigma;

  const float m0S = glm::dot(baS, baS);
  const float m1 = glm::dot(oa, baS);
  const float m3 = glm::dot(rdn, baS);
  const float m4 = glm::dot(rdn, oa);
  const float m5 = glm::dot(oa, oa);
  const float rr = raS - rbS;
  const float hy = m0S + rr * rr;

  // k2 t^2 + 2 k1 t + k0 = 0 (rd unit, primitive prescaled to ~unit).
  const float k2 = differenceOfProducts(m0S, m0S, m3 * m3, hy);
  const float k1 = m0S * m0S * m4 - m1 * m3 * hy + m0S * raS * rr * m3;
  const float k0 = m0S * m0S * m5 - m1 * m1 * hy
      + m0S * raS * (rr * m1 * 2.0f - m0S * raS);

  // Scaled offset roots (distance along rdn in the prescaled frame). The total
  // unit-direction param is s + to/sigma; ray t divides that by rlen, i.e.
  // t = (sigma*s + to) / (sigma*rlen) (see emitBody).
  float to0 = 0.f, to1 = 0.f;
  int nroots = 0;
  // Exit-side reliability of the quadratic solve; see kGrazeRelEps. The exit
  // root isn't positionally fixed for a cone (k2's sign orders the roots), so
  // the gate is applied by facing in emitBody below. The linear (k2~0) path
  // has no discriminant and keeps its single root.
  bool exitReliable = true;
  const float k2scale = glm::max(m0S * m0S, glm::abs(m3 * m3 * hy));
  if (glm::abs(k2) <= kRelEps * k2scale) {
    // Near-parallel to the slant generator: 2 k1 t + k0 = 0. Gate |k1| against
    // the magnitudes of its own terms (same units, any scene scale) — the root
    // is only unreliable when k1 itself is a catastrophic cancellation.
    const float k1scale = glm::max(glm::abs(m0S * m0S * m4),
        glm::max(glm::abs(m1 * m3 * hy), glm::abs(m0S * raS * rr * m3)));
    if (glm::abs(k1) > kRelEps * k1scale) {
      to0 = divExact(-k0, 2.0f * k1);
      nroots = 1;
    }
  } else {
    // Cancellation-free discriminant (see differenceOfProducts): near tangency
    // and near a sharp taper h = k1^2 - k2*k0 is a subtraction of nearly-equal
    // squared terms — the cone-apex speckle lives here.
    const float h = differenceOfProducts(k1, k1, k2, k0);
    if (h >= 0.f) {
      // Stable pairing (qq/k2, k0/qq): avoids the -k1 + sqrt cancellation.
      const float qq = -(k1 + (k1 >= 0.f ? 1.f : -1.f) * sqrtExact(h));
      to0 = divExact(qq, k2);
      to1 = divExact(k0, qq); // qq==0 -> non-finite -> dropped downstream
      nroots = 2;
      exitReliable = h > kGrazeRelEps * glm::max(k1 * k1, glm::abs(k2 * k0));
    }
  }

  DedupT dedup;

  const auto emitBody = [&](float to) {
    const float y = m1 + to * m3; // scaled axial coord, span [0, m0S]
    if (y > 0.f && y < m0S) {
      const vec3 nrm = glm::normalize(
          m0S * (m0S * (oa + to * rdn) + rr * baS * raS) - baS * hy * y);
      if (!exitReliable && glm::dot(nrm, rdn) > 0.f)
        return; // graze-noise exit: phantom self-shadow (see kGrazeRelEps)
      emitCrossing(dedup,
          report,
          PrimHit{
              divExact(fmaf(sigma, s, to), sigma * rlen), nrm, y / m0S});
    }
  };
  if (nroots > 0)
    emitBody(to0);
  if (nroots > 1)
    emitBody(to1);

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
