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

// Oracle / property / metamorphic host tests for the analytic primitive
// solvers. These are the "computationally correct grounds" gates: the fp32
// solvers in intersectPrimitives.h are checked against an INDEPENDENT fp64
// reference across the full threat model (non-unit sheared directions, far-from-
// origin coordinates, extreme radius/length aspect ratios, grazing/axis-parallel
// rays), plus metamorphic invariants (exact power-of-two scaling, cone->cylinder
// continuity). The reference uses a different algebra from the fp32 code (a
// canonical linear-radius quadratic, c = |w_perp|^2 - R(w_z)^2) so a formulation
// bug cannot hide inside a shared blind spot; the fp32 code uses iquilezles'
// m0^2-scaled coefficients. See .scratch/intersection-grounds/derivations.md.
//
// No CUDA/OptiX/GPU. Returns nonzero on any failure (assert() is compiled out in
// Release, the repo default).

#include "gpu/intersectPrimitives.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace visrtx;
using glm::dvec3;
using glm::vec3;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);             \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

// ------------------------------------------------------------------ oracle ---
// Independent fp64 reference. Enumerates every boundary crossing (body entry +
// exit + enabled caps), unordered, matching the solve* contract.

struct DHit
{
  double t;
  dvec3 n; // outward, unit
  bool cap;
};

bool finiteD(double x)
{
  return std::isfinite(x);
}

// Ray vs flat cap disk (plane through pc, outward unit normal on, radius R).
bool oracleCap(const dvec3 &ro,
    const dvec3 &rd,
    const dvec3 &pc,
    const dvec3 &axis, // need not be unit
    const dvec3 &on,
    double R,
    DHit &h)
{
  const double sd = glm::dot(rd, axis);
  const double axis2 = glm::dot(axis, axis);
  const double d2 = glm::dot(rd, rd);
  if (sd * sd <= 1e-24 * axis2 * d2)
    return false;
  const double t = glm::dot(pc - ro, axis) / sd;
  const dvec3 rel = (ro + t * rd) - pc;
  if (glm::dot(rel, rel) > R * R)
    return false;
  h = DHit{t, on, true};
  return true;
}

// Truncated-cone crossings via the canonical linear-radius quadratic (see file
// header). Radius R(z) = r0 + k z along the axis, k = (r1-r0)/L, z in [0,L].
// Cylinder is r0 == r1 (k == 0). caps: bit0 = p0 end, bit1 = p1 end.
std::vector<DHit> oracleCone(const dvec3 &ro,
    const dvec3 &rd,
    const dvec3 &p0,
    const dvec3 &p1,
    double r0,
    double r1,
    uint8_t caps)
{
  std::vector<DHit> out;
  r0 = std::fabs(r0);
  r1 = std::fabs(r1);
  const dvec3 a = p1 - p0;
  const double L2 = glm::dot(a, a);
  const double A2 = glm::dot(rd, rd);
  if (L2 <= 0.0 || A2 <= 0.0)
    return out;
  const double L = std::sqrt(L2);
  const dvec3 ah = a / L;
  const double k = (r1 - r0) / L;

  const dvec3 w = ro - p0;
  const double wz = glm::dot(w, ah);
  const double dz = glm::dot(rd, ah);
  const double WD = glm::dot(w, rd);
  const double W2 = glm::dot(w, w);
  const double R0 = r0 + k * wz; // radius at the origin's axial foot

  // a t^2 + b t + c = 0
  const double qa = A2 - dz * dz * (1.0 + k * k);
  const double qb = 2.0 * (WD - dz * (wz * (1.0 + k * k) + r0 * k));
  const double qc = (W2 - wz * wz) - R0 * R0;

  auto emitBody = [&](double t) {
    const dvec3 P = ro + t * rd;
    const double z = glm::dot(P - p0, ah);
    if (z < 0.0 || z > L)
      return;
    const dvec3 perp = (P - p0) - z * ah;
    const double R = r0 + k * z;
    const dvec3 n = glm::normalize(perp - k * R * ah);
    out.push_back(DHit{t, n, false});
  };

  if (std::fabs(qa) <= 1e-14 * std::max(A2, dz * dz)) {
    if (std::fabs(qb) > 0.0)
      emitBody(-qc / qb);
  } else {
    const double disc = qb * qb - 4.0 * qa * qc;
    if (disc >= 0.0) {
      const double sq = std::sqrt(disc);
      const double q = -0.5 * (qb + (qb >= 0.0 ? sq : -sq));
      emitBody(q / qa);
      emitBody(qc / q);
    }
  }

  DHit h;
  if ((caps & CAP_FIRST) && r0 > 0.0
      && oracleCap(ro, rd, p0, a, -ah, r0, h))
    out.push_back(h);
  if ((caps & CAP_SECOND) && r1 > 0.0
      && oracleCap(ro, rd, p1, a, ah, r1, h))
    out.push_back(h);
  return out;
}

std::vector<DHit> oracleCylinder(const dvec3 &ro,
    const dvec3 &rd,
    const dvec3 &p0,
    const dvec3 &p1,
    double r,
    uint8_t caps)
{
  return oracleCone(ro, rd, p0, p1, r, r, caps);
}

std::vector<DHit> oracleSphere(
    const dvec3 &ro, const dvec3 &rd, const dvec3 &c, double r)
{
  std::vector<DHit> out;
  r = std::fabs(r);
  const dvec3 oc = ro - c;
  const double a = glm::dot(rd, rd);
  if (a <= 0.0)
    return out;
  const double b = 2.0 * glm::dot(oc, rd);
  const double cc = glm::dot(oc, oc) - r * r;
  const double disc = b * b - 4.0 * a * cc;
  if (disc < 0.0)
    return out;
  const double sq = std::sqrt(disc);
  const double q = -0.5 * (b + (b >= 0.0 ? sq : -sq));
  const double t0 = q / a;
  const double t1 = cc / q;
  for (double t : {t0, t1})
    out.push_back(DHit{t, glm::normalize((ro + t * rd) - c), false});
  return out;
}

// --------------------------------------------------------------- utilities ---

template <typename Hit>
const Hit *nearestForward(const std::vector<Hit> &h, double tmin)
{
  const Hit *best = nullptr;
  for (const auto &x : h)
    if (finiteD(double(x.t)) && double(x.t) >= tmin
        && (!best || x.t < best->t))
      best = &x;
  return best;
}

const PrimHit *nearestForward(const PrimHit *h, int n, float tmin)
{
  const PrimHit *best = nullptr;
  for (int i = 0; i < n; ++i)
    if (std::isfinite(h[i].t) && h[i].t >= tmin && (!best || h[i].t < best->t))
      best = &h[i];
  return best;
}

double angleBetween(const dvec3 &a, const vec3 &bf)
{
  const dvec3 b(bf.x, bf.y, bf.z);
  const double d = glm::clamp(glm::dot(glm::normalize(a), glm::normalize(b)),
      -1.0,
      1.0);
  return std::acos(d);
}

bool anyNonFinite(const PrimHit *h, int n)
{
  for (int i = 0; i < n; ++i)
    if (!std::isfinite(h[i].t) || !std::isfinite(h[i].Ng.x)
        || !std::isfinite(h[i].Ng.y) || !std::isfinite(h[i].Ng.z))
      return true;
  return false;
}

enum Prim
{
  P_SPHERE,
  P_CYL,
  P_CONE
};

struct Case
{
  Prim prim;
  vec3 ro, rd, p0, p1;
  float r0, r1; // sphere: r0 = radius, center = p0
  uint8_t caps;
};

double primScale(const Case &c);
double maxAbsCoord(const Case &c);
std::vector<DHit> runOracle(const Case &c);

// True when the case sits in a numerically ambiguous band where fp32 and fp64
// legitimately disagree on hit existence or exact position/normal, and the
// disagreement is sub-pixel / visually irrelevant. Strict comparisons are
// skipped there; only NaN-freedom is still required. The bands are:
//   (a) grazing incidence — |cos(ray, normal)| tiny: the normal and the
//       entry/exit split are ill-conditioned by nature (silhouette edge);
//   (b) near-coincident crossings — two surfaces (tangent, or wall/cap rim)
//       within a hair in t: which one is "nearest" is fp-noise;
//   (c) silhouette/thinness conditioning — perturbing the radii by the fp32
//       noise level (relative primitive tol OR the absolute coordinate-ulp
//       floor, whichever is larger) changes hit existence or the normal
//       materially. This one predicate captures both a grazing silhouette and
//       a primitive thinner than the ulp of its own coordinates: in either
//       case the answer is set by fp noise no reformulation can undo.
bool ambiguousBand(const Case &c, const std::vector<DHit> &oh, double tmin)
{
  const dvec3 rd(c.rd);
  const DHit *n0 = nearestForward(oh, tmin);

  // (a) grazing.
  if (n0 && finiteD(n0->n.x)
      && std::fabs(glm::dot(glm::normalize(rd), n0->n)) < 0.08)
    return true;

  // (b) near-coincident crossings.
  if (n0) {
    const double band =
        std::max(3e-3 * primScale(c), 3e-3 * std::fabs(n0->t));
    int nearCount = 0;
    for (const auto &x : oh)
      if (finiteD(x.t) && std::fabs(double(x.t) - n0->t) < band)
        ++nearCount;
    if (nearCount > 1)
      return true;
  }

  // (c) silhouette/thinness perturbation at the fp32 noise level. eps scales
  // with the primitive extent (axis length included), deliberately conservative:
  // for a NEEDLE-thin cone (r << L) both hit/miss and the surface normal are
  // inherently position-noise-sensitive (normal sensitivity ~ 1/local_radius),
  // so a radius-only eps would under-band them and surface fp32-ambiguous — not
  // defective — needle cases as failures. The coordinate-ulp floor additionally
  // bands primitives thinner than fp32 can resolve at their own coordinates.
  const double resFloor = 128.0 * std::max(maxAbsCoord(c), 1.0) * 0x1p-24;
  const float eps = float(std::max(3e-3 * primScale(c), resFloor));
  Case cp = c, cm = c;
  cp.r0 = c.r0 + eps;
  cp.r1 = c.r1 + eps;
  cm.r0 = std::max(0.f, c.r0 - eps);
  cm.r1 = std::max(0.f, c.r1 - eps);
  const std::vector<DHit> hp = runOracle(cp);
  const std::vector<DHit> hm = runOracle(cm);
  const DHit *np = nearestForward(hp, tmin);
  const DHit *nm = nearestForward(hm, tmin);
  if (bool(np) != bool(nm))
    return true; // visibility set by fp noise
  if (np && nm && finiteD(np->n.x) && finiteD(nm->n.x)
      && angleBetween(np->n, vec3(nm->n)) > 0.02)
    return true; // normal set by fp noise
  return false;
}

// ---------------------------------------------------- comparison of a case ---

std::vector<DHit> runOracle(const Case &c)
{
  const dvec3 ro(c.ro), rd(c.rd), p0(c.p0), p1(c.p1);
  switch (c.prim) {
  case P_SPHERE:
    return oracleSphere(ro, rd, p0, double(c.r0));
  case P_CYL:
    return oracleCylinder(ro, rd, p0, p1, double(c.r0), c.caps);
  case P_CONE:
    return oracleCone(ro, rd, p0, p1, double(c.r0), double(c.r1), c.caps);
  }
  return {};
}

int runFp32(const Case &c, PrimHit out[4])
{
  switch (c.prim) {
  case P_SPHERE:
    return solveSphere(c.ro, c.rd, c.p0, c.r0, out);
  case P_CYL:
    return solveCylinder(c.ro, c.rd, c.p0, c.p1, c.r0, c.caps, out);
  case P_CONE:
    return solveCone(c.ro, c.rd, c.p0, c.p1, c.r0, c.r1, c.caps, out);
  }
  return 0;
}

// Characteristic length of the primitive (for relative tolerances).
double primScale(const Case &c)
{
  const double L = glm::length(dvec3(c.p1) - dvec3(c.p0));
  return std::max({double(std::fabs(c.r0)),
      double(std::fabs(c.r1)),
      c.prim == P_SPHERE ? 0.0 : L,
      1e-30});
}

// Largest absolute coordinate the solver differences (bounds input snapping:
// a coordinate of magnitude M carries ~M*2^-24 absolute fp32 error before any
// arithmetic, which no reformulation can remove).
double maxAbsCoord(const Case &c)
{
  auto m = [](const vec3 &v) {
    return double(std::max({std::fabs(v.x), std::fabs(v.y), std::fabs(v.z)}));
  };
  return std::max({m(c.ro), m(c.p0), m(c.p1), double(std::fabs(c.r0)), 1.0});
}

struct Stats
{
  int trials = 0, compared = 0, banded = 0, visMismatch = 0, nan = 0;
  int setCompared = 0; // cases whose FULL crossing set was verified
  double maxTRel = 0, maxAngle = 0;
};

// Full detail (ray/primitive/crossings) for the first few failing cases, so a
// regression is diagnosable from the test log alone. Inert when everything
// passes.
int g_dumpBudget = 24;

void dumpCase(const char *why, const Case &c, const std::vector<DHit> &oh)
{
  if (g_dumpBudget <= 0)
    return;
  --g_dumpBudget;
  PrimHit fp[4];
  const int nf = runFp32(c, fp);
  std::printf("---- DUMP %s prim=%d caps=%u\n", why, c.prim, c.caps);
  std::printf("  ro=(%.6g %.6g %.6g) rd=(%.6g %.6g %.6g) |rd|=%.6g\n",
      c.ro.x, c.ro.y, c.ro.z, c.rd.x, c.rd.y, c.rd.z, glm::length(c.rd));
  std::printf("  p0=(%.6g %.6g %.6g) p1=(%.6g %.6g %.6g) r0=%.6g r1=%.6g\n",
      c.p0.x, c.p0.y, c.p0.z, c.p1.x, c.p1.y, c.p1.z, c.r0, c.r1);
  std::printf("  fp32 %d crossing(s):\n", nf);
  for (int i = 0; i < nf; ++i)
    std::printf("    t=%.8g Ng=(%.5g %.5g %.5g) u=%.5g\n", fp[i].t, fp[i].Ng.x,
        fp[i].Ng.y, fp[i].Ng.z, fp[i].u);
  std::printf("  oracle %zu crossing(s):\n", oh.size());
  for (const auto &h : oh)
    std::printf("    t=%.8g n=(%.5g %.5g %.5g) cap=%d\n", h.t, h.n.x, h.n.y,
        h.n.z, int(h.cap));
}

// Verify the FULL set of crossings — entry AND exit AND caps — against the
// oracle, not just the nearest-forward hit. This is the coverage the sweeps
// otherwise miss: exit crossings, cap-vs-body, and the reported crossing COUNT
// (the two-sided reporting contract). Runs only for "strongly clear" cases —
// every oracle crossing well away from grazing/tangency and pairwise separated
// — so the solver's legitimate dedup/graze gates cannot drop one and cause a
// false failure. Returns true if it performed a comparison.
bool compareCrossingSet(
    const Case &c, const PrimHit *fp, int nf, const std::vector<DHit> &oh,
    Stats &st)
{
  const dvec3 rdHat = glm::normalize(dvec3(c.rd));
  const double scale = primScale(c);
  const double resFloor = 128.0 * std::max(maxAbsCoord(c), 1.0) * 0x1p-24;
  if (scale < 8.0 * resFloor)
    return false; // thinner than fp32 resolves

  std::vector<const DHit *> ov;
  for (const auto &h : oh)
    if (finiteD(h.t))
      ov.push_back(&h);

  // Strongly-clear gate: no grazing crossing, no body crossing on the axial
  // span rim (where the solver's strict y in (0,m0S) and the oracle's inclusive
  // z in [0,L] can legitimately disagree by an fp ulp), and no near-coincident
  // pair. Near-coincidence is judged at the fp32 t-RESOLUTION (input snapping /
  // |rd| plus t ulps), not a fixed fraction of |t| — the latter over-bands both
  // far (large |t|) and non-unit-direction (large |rd|) cases and silently
  // skips them.
  const double rdLen = glm::length(dvec3(c.rd));
  for (auto *h : ov) {
    if (!finiteD(h->n.x) || std::fabs(glm::dot(rdHat, h->n)) < 0.12)
      return false;
    if (!h->cap && c.prim != P_SPHERE) { // body crossing near an axial endpoint
      const dvec3 ba = dvec3(c.p1) - dvec3(c.p0);
      const double u =
          glm::dot((dvec3(c.ro) + h->t * dvec3(c.rd)) - dvec3(c.p0), ba)
          / glm::dot(ba, ba);
      if (u < 2e-3 || u > 1.0 - 2e-3)
        return false;
    }
  }
  const double tRes =
      256.0 * 0x1p-24 * (maxAbsCoord(c) / std::max(rdLen, 1e-30));
  for (size_t i = 0; i < ov.size(); ++i)
    for (size_t j = i + 1; j < ov.size(); ++j) {
      const double band = std::max(tRes,
          256.0 * 0x1p-24 * std::max(std::fabs(ov[i]->t), std::fabs(ov[j]->t)));
      if (std::fabs(ov[i]->t - ov[j]->t) < band)
        return false;
    }

  // Crossing COUNT must match (a dropped exit/cap or a spurious crossing here
  // is a real two-sided-reporting defect).
  if (int(ov.size()) != nf) {
    std::printf("FAIL crossing-count: prim=%d fp32=%d oracle=%zu scale=%g\n",
        c.prim, nf, ov.size(), scale);
    dumpCase("crossing-count", c, oh);
    ++g_failures;
    return true;
  }
  if (nf == 0)
    return false;

  // Sort both by t; matched pairwise (all separated, so order aligns).
  std::vector<int> fi(nf);
  for (int i = 0; i < nf; ++i)
    fi[i] = i;
  std::sort(fi.begin(), fi.end(),
      [&](int a, int b) { return fp[a].t < fp[b].t; });
  std::sort(ov.begin(), ov.end(),
      [](const DHit *a, const DHit *b) { return a->t < b->t; });

  ++st.setCompared;
  for (int i = 0; i < nf; ++i) {
    const PrimHit &f = fp[fi[i]];
    const DHit &o = *ov[i];
    const dvec3 Pf = dvec3(c.ro) + double(f.t) * dvec3(c.rd);
    const dvec3 Po = dvec3(c.ro) + o.t * dvec3(c.rd);
    const double posTol = 3e-3 * std::max(scale, glm::length(Po - dvec3(c.p0)))
        + 8.0 * maxAbsCoord(c) * 0x1p-24;
    if (glm::length(Pf - Po) > posTol) {
      std::printf("FAIL crossing[%d] pos: prim=%d t_fp=%g t_or=%g scale=%g\n",
          i, c.prim, double(f.t), o.t, scale);
      dumpCase("crossing-pos", c, oh);
      ++g_failures;
      break;
    }
    const double cosInc = std::fabs(glm::dot(rdHat, o.n));
    if (angleBetween(o.n, f.Ng)
        > 3e-3 + 5.0 * posTol / std::max(scale, 1e-30) + 0.02 / cosInc) {
      std::printf("FAIL crossing[%d] normal: prim=%d t=%g scale=%g\n", i,
          c.prim, o.t, scale);
      dumpCase("crossing-normal", c, oh);
      ++g_failures;
      break;
    }
  }
  return true;
}

// Compare one case; record failures. tmin models the render epsilon.
void compareCase(const Case &c, Stats &st, double tmin = 0.0)
{
  ++st.trials;
  PrimHit fp[4];
  const int nf = runFp32(c, fp);

  if (anyNonFinite(fp, nf)) {
    std::printf("FAIL NaN/Inf: prim=%d ro=(%g %g %g) rd=(%g %g %g)\n",
        c.prim, c.ro.x, c.ro.y, c.ro.z, c.rd.x, c.rd.y, c.rd.z);
    ++st.nan;
    ++g_failures;
    return;
  }

  const std::vector<DHit> oh = runOracle(c);
  const double scale = primScale(c);

  const PrimHit *fn = nearestForward(fp, nf, float(tmin));
  const DHit *on = nearestForward(oh, tmin);

  const bool banded = ambiguousBand(c, oh, tmin);
  if (banded) {
    ++st.banded;
    return; // NaN-freedom already checked; skip strict compare
  }

  // Verify the whole crossing set (count + exit + caps), not just the nearest
  // forward hit — the two-sided reporting contract. Self-gated to clean cases.
  compareCrossingSet(c, fp, nf, oh, st);

  // Visibility agreement outside the band.
  if (bool(fn) != bool(on)) {
    // Allow a one-sided miss only if the oracle hit is itself within a few
    // ulp of tmin (a crossing straddling the epsilon plane).
    const bool nearTmin = on && std::fabs(on->t - tmin) < 3e-3 * scale;
    const bool nearTmin2 = fn && std::fabs(double(fn->t) - tmin) < 3e-3 * scale;
    if (!nearTmin && !nearTmin2) {
      std::printf(
          "FAIL visibility: prim=%d fp32=%s oracle=%s scale=%g maxc=%g\n",
          c.prim,
          fn ? "hit" : "miss",
          on ? "hit" : "miss",
          scale,
          maxAbsCoord(c));
      ++st.visMismatch;
      ++g_failures;
    }
    return;
  }
  if (!fn || !on)
    return;

  ++st.compared;

  // Hit-point agreement (position error absorbs both t error and input
  // snapping). tol = relative primitive tol + coordinate-snapping floor.
  const dvec3 Pf = dvec3(c.ro) + double(fn->t) * dvec3(c.rd);
  const dvec3 Po = dvec3(c.ro) + on->t * dvec3(c.rd);
  const double posErr = glm::length(Pf - Po);
  const double posTol =
      3e-3 * std::max(scale, glm::length(Po - dvec3(c.p0)))
      + 8.0 * maxAbsCoord(c) * 0x1p-24;
  if (posErr > posTol) {
    std::printf("FAIL hitpos: prim=%d err=%g tol=%g t_fp=%g t_or=%g "
                "scale=%g maxc=%g |rd|=%g\n",
        c.prim, posErr, posTol, double(fn->t), on->t, scale, maxAbsCoord(c),
        glm::length(dvec3(c.rd)));
    dumpCase("hitpos", c, oh);
    ++g_failures;
  }
  const double tRel = std::fabs(double(fn->t) - on->t)
      / std::max(std::fabs(on->t), 1e-30);
  st.maxTRel = std::max(st.maxTRel, tRel);

  // Normal agreement (skip if the oracle normal is degenerate). The surface
  // normal's sensitivity to the hit-point error scales as 1/cos(incidence):
  // at grazing incidence the normal is nearly perpendicular to the ray, so a
  // small along-ray position error rotates it a lot. That is inherent
  // conditioning, not a solver defect, so the tolerance carries a 1/cosInc term
  // (grazing incidence below ~0.08 is banded outright in ambiguousBand).
  if (finiteD(on->n.x)) {
    const double ang = angleBetween(on->n, fn->Ng);
    const double cosInc =
        std::fabs(glm::dot(glm::normalize(dvec3(c.rd)), on->n));
    st.maxAngle = std::max(st.maxAngle, ang);
    const double normalTol = 3e-3 + 5.0 * posTol / std::max(scale, 1e-30)
        + 0.02 / std::max(cosInc, 0.05);
    if (ang > normalTol) {
      std::printf("FAIL normal: prim=%d angle=%g t=%g scale=%g cosInc=%g\n",
          c.prim, ang, on->t, scale, cosInc);
      dumpCase("normal", c, oh);
      ++g_failures;
    }
  }
}

// --------------------------------------------------------- property sweeps ---

struct Rng
{
  std::mt19937 gen;
  explicit Rng(uint32_t seed) : gen(seed) {}
  double uni(double a, double b)
  {
    return a + (b - a) * (double(gen()) / double(0xFFFFFFFFu));
  }
  dvec3 dir()
  {
    // Uniform-ish on the sphere; magnitude applied by caller.
    for (;;) {
      dvec3 v(uni(-1, 1), uni(-1, 1), uni(-1, 1));
      const double l2 = glm::dot(v, v);
      if (l2 > 1e-6 && l2 <= 1.0)
        return v / std::sqrt(l2);
    }
  }
  // log-uniform magnitude in [10^lo, 10^hi]
  double logMag(double lo, double hi)
  {
    return std::pow(10.0, uni(lo, hi));
  }
};

// Build a random case that aims the ray at the primitive so a large fraction
// hit, across the threat-model ranges. `farExp` sets the world-offset decade
// (coordinates ~ 10^farExp), `dirLo/dirHi` the direction-magnitude decades
// (non-unit = instance scale/shear analogue), `rLo/rHi` the log10 radius/length
// aspect bounds.
Case randomCase(Rng &rng, Prim prim, double farExp, double dirLo, double dirHi,
    double rLo, double rHi)
{
  Case c;
  c.prim = prim;
  c.caps = uint8_t(rng.uni(0, 3.999));

  // Far world offset (same for prim and ray so the object-space difference is
  // well-scaled but the absolute coordinates are large).
  const dvec3 F = farExp > 0 ? rng.dir() * rng.logMag(farExp - 0.5, farExp)
                             : dvec3(0.0);

  const double L = rng.logMag(-1.0, 1.0); // axis length ~ [0.1, 10]
  // Radius/length aspect drawn log-uniformly in [10^rLo, 10^rHi]. The important
  // sci-vis direction is THIN (fibers, streamlines, molecular bonds: r << L),
  // swept down to 1e-6. The fat/pancake direction (r >> L) is a near-flat disk;
  // beyond ~1e2 the apex-free quadratic's hy = m0 + rr^2 is swamped by rr^2 and
  // fp32 conditioning degrades (still safe — finite, clean — but not fp64-
  // accurate), so the gate caps it rather than asserting precision the
  // formulation does not provide.
  const double rBase = L * rng.logMag(rLo, rHi);
  const dvec3 axisDir = rng.dir();
  const dvec3 p0 = F;
  const dvec3 p1 = F + axisDir * L;

  double r0 = rBase, r1 = rBase;
  if (prim == P_CONE) {
    r0 = L * rng.logMag(rLo, rHi);
    r1 = L * rng.logMag(rLo, rHi);
    if (rng.uni(0, 1) < 0.15)
      r1 = 0.0; // apex end (issue #333)
  }
  const dvec3 center = prim == P_SPHERE ? F : 0.5 * (p0 + p1);
  const double rad = prim == P_SPHERE ? rBase : std::max(r0, r1);

  // Aim: a target point near the primitive, and a ray origin at a random
  // distance back along a random direction.
  const dvec3 target = center
      + rng.dir() * rad * rng.uni(-1.2, 1.2)
      + axisDir * (prim == P_SPHERE ? 0.0 : L * rng.uni(-0.2, 1.2));
  const dvec3 rdHat = rng.dir();
  const double dist = rng.logMag(-1.0, std::max(farExp, 1.0)); // reach
  const dvec3 ro = target - rdHat * dist;
  const double dmag = rng.logMag(dirLo, dirHi);
  const dvec3 rd = rdHat * dmag;

  c.ro = vec3(ro);
  c.rd = vec3(rd);
  c.p0 = vec3(prim == P_SPHERE ? center : p0);
  c.p1 = vec3(p1);
  c.r0 = float(prim == P_SPHERE ? rad : r0);
  c.r1 = float(r1);
  return c;
}

void sweep(const char *name, Prim prim, double farExp, double dirLo,
    double dirHi, double rLo, double rHi, int n, uint32_t seed)
{
  Rng rng(seed);
  Stats st;
  const int before = g_failures;
  for (int i = 0; i < n; ++i) {
    const Case c = randomCase(rng, prim, farExp, dirLo, dirHi, rLo, rHi);
    // tmin = 0: the nearest-FORWARD comparison uses it, while compareCrossingSet
    // verifies every crossing (incl. negative-t exits) regardless.
    compareCase(c, st, 0.0);
  }
  // Guard against a vacuous pass: if a future change pushed nearly every case
  // into the ambiguous band, the sweep would compare almost nothing. The floor
  // is well below the legitimate minimum (thin-fiber sweeps band ~88%, so
  // ~11.5% is compared) but still trips if strict comparison collapses.
  if (st.compared < st.trials / 20) {
    std::printf("FAIL %s: only %d/%d cases strictly compared (band too wide?)\n",
        name, st.compared, st.trials);
    ++g_failures;
  }
  // Same guard for the full-crossing-set path: its strongly-clear gate is
  // stricter, but must not silently collapse to nothing either.
  if (st.setCompared < st.trials / 50) {
    std::printf("FAIL %s: only %d/%d full crossing sets verified\n", name,
        st.setCompared, st.trials);
    ++g_failures;
  }
  std::printf("[%-22s] trials=%d compared=%d set=%d banded=%d  maxTRel=%.2e "
              "maxAng=%.2e  %s\n",
      name, st.trials, st.compared, st.setCompared, st.banded, st.maxTRel,
      st.maxAngle, g_failures == before ? "ok" : "FAILURES");
}

// ------------------------------------------------------------ metamorphic ---

// Scaling all inputs by a power of two must reproduce bit-identical t and
// (scale-invariant) normal: every op scales by an exact power of two, so no
// rounding differs. Guards against any hidden absolute constant.
void testPow2Invariance()
{
  Rng rng(12345);
  int bad = 0;
  for (int i = 0; i < 20000; ++i) {
    const int which = int(rng.uni(0, 2.999));
    Case c = randomCase(rng, Prim(which), 0.0, -1.0, 1.0, -1.5, 1.5);
    PrimHit a[4], b[4];
    const int na = runFp32(c, a);
    // Scale by 2^s (exact); s kept moderate so nothing over/underflows.
    const float s = std::ldexp(1.0f, int(rng.uni(-8, 8)));
    Case cs = c;
    cs.ro = c.ro * s;
    cs.rd = c.rd * s;
    cs.p0 = c.p0 * s;
    cs.p1 = c.p1 * s;
    cs.r0 = c.r0 * s;
    cs.r1 = c.r1 * s;
    const int nb = runFp32(cs, b);
    if (na != nb) {
      ++bad;
      continue;
    }
    for (int j = 0; j < na; ++j) {
      // t is scale-invariant (positions and direction scale together); normal
      // is a normalized direction, also invariant.
      if (a[j].t != b[j].t || a[j].Ng != b[j].Ng)
        ++bad;
    }
  }
  if (bad) {
    std::printf("FAIL pow2 invariance: %d mismatch(es)\n", bad);
    g_failures += 1;
  } else
    std::printf("[pow2-invariance       ] 20000 cases bit-identical  ok\n");
}

// Scaling all inputs by a large/small NON-power-of-two factor must reproduce
// the same intersection: hit t scales with the factor, the normal is invariant.
// This gates HUGE and TINY absolute primitive sizes (axis and radius ~1e4 or
// ~1e-4) — the regime the offset sweeps miss (they place a small primitive at a
// far offset, never a large primitive) and where the cone's m0^2-scaled
// coefficients need the exact power-of-two prescale to stay conditioned.
void testScaleInvariance()
{
  Rng rng(24680);
  const double factors[] = {1e-4, 1e-2, 1e2, 1e4, 1e5};
  int bad = 0, tested = 0;
  for (int i = 0; i < 60000; ++i) {
    const int which = int(rng.uni(0, 2.999));
    Case c = randomCase(rng, Prim(which), 0.0, -1.0, 1.0, -1.5, 1.5);
    const PrimHit *r0 = nullptr;
    PrimHit a[4];
    const int na = runFp32(c, a);
    r0 = nearestForward(a, na, 0.f);
    if (!r0)
      continue;
    const double K = factors[i % 5];
    Case cs = c;
    cs.ro = c.ro * float(K);
    cs.rd = c.rd; // keep direction; then t scales by K
    cs.p0 = c.p0 * float(K);
    cs.p1 = c.p1 * float(K);
    cs.r0 = float(c.r0 * K);
    cs.r1 = float(c.r1 * K);
    PrimHit b[4];
    const int nb = runFp32(cs, b);
    const PrimHit *r1 = nearestForward(b, nb, 0.f);
    // Skip the numerically ambiguous band (grazing/thin/silhouette) at the
    // scaled size, matching the sweep policy.
    if (ambiguousBand(cs, runOracle(cs), 0.0))
      continue;
    ++tested;
    if (!r1) {
      ++bad;
      continue;
    }
    const double tExpected = double(r0->t) * K; // ro,p scaled by K, rd fixed
    if (std::fabs(double(r1->t) - tExpected)
            > 3e-3 * std::max(std::fabs(tExpected), 1e-30)
        || angleBetween(dvec3(r1->Ng), r0->Ng) > 5e-3)
      ++bad;
  }
  if (bad) {
    std::printf("FAIL scale invariance: %d/%d mismatch(es)\n", bad, tested);
    g_failures += 1;
  } else
    std::printf(
        "[scale-invariance      ] %d cases (1e-4..1e5) match  ok\n", tested);
}

// A cone with r0 == r1 must intersect identically to a cylinder of that radius
// (the apex-free formulation's continuous cylinder limit — no special case).
void testConeCylinderContinuity()
{
  Rng rng(777);
  int bad = 0;
  for (int i = 0; i < 20000; ++i) {
    Case c = randomCase(rng, P_CYL, 0.0, -0.5, 0.5, -1.5, 1.5);
    PrimHit cyl[4], cone[4];
    const int ncyl = solveCylinder(
        c.ro, c.rd, c.p0, c.p1, c.r0, c.caps, cyl);
    const int ncone = solveCone(
        c.ro, c.rd, c.p0, c.p1, c.r0, c.r0, c.caps, cone);
    const PrimHit *a = nearestForward(cyl, ncyl, 0.f);
    const PrimHit *b = nearestForward(cone, ncone, 0.f);
    if (bool(a) != bool(b)) {
      ++bad;
      continue;
    }
    if (a && b) {
      if (std::fabs(a->t - b->t)
              > 1e-3 * std::max(std::fabs(double(a->t)), 1.0)
          || angleBetween(dvec3(b->Ng), a->Ng) > 5e-3) {
        // Tolerate disagreement only in the numerically ambiguous band.
        Case cc = c;
        cc.prim = P_CONE;
        cc.r1 = c.r0;
        if (!ambiguousBand(cc, runOracle(cc), 0.0))
          ++bad;
      }
    }
  }
  if (bad) {
    std::printf("FAIL cone==cylinder continuity: %d mismatch(es)\n", bad);
    g_failures += 1;
  } else
    std::printf("[cone==cylinder        ] 20000 cases match  ok\n");
}

// ------------------------------------------------------- targeted regressions

// Coincident crossings must count ONCE, not twice, or an accumulating anyhit
// double-counts opacity. Deterministic and AA-free — a render test cannot gate
// this because the coincidence sits on the measure-zero, anti-aliased
// silhouette. Two distinct mechanisms:
//   - a TANGENT ray: the two body roots coincide; the grazing gate
//     (kGrazeRelEps) / dedup must not emit both. A tangent may legitimately
//     round to a miss, so the gate is "<= 1", never 2.
//   - a WALL/CAP RIM: a wall crossing and a cap crossing land at the SAME t
//     where the wall meets a cap. Here the two roots are NON-grazing and
//     distinct-surface, so ONLY DedupT (coincident-t merge) prevents the double
//     count. A ray through both rim circles yields 4 raw candidates (2 wall +
//     2 cap) that must dedup to exactly 2 — this is the path the render
//     transparency test defers to, and it fails (reports 4) if DedupT regresses.
void testCoincidentDedup()
{
  PrimHit h[4];
  int bad = 0;
  const auto tangentOnce = [&](const char *name, int n) {
    if (n > 1) {
      std::printf("FAIL dedup tangent: %s reported %d (double count)\n", name,
          n);
      ++bad;
    }
  };
  const auto rimCount = [&](const char *name, int n) {
    if (n != 2) {
      std::printf("FAIL dedup rim: %s reported %d (expected 2; 4 = wall/cap "
                  "double count)\n",
          name, n);
      ++bad;
    }
  };
  const auto rimAtMost2 = [&](const char *name, int n) {
    if (n > 2) {
      std::printf("FAIL dedup rim: %s reported %d (>2; a convex solid has at "
                  "most 2 crossings — wall/cap double count)\n",
          name, n);
      ++bad;
    }
  };

  // Tangents (grazing gate / dedup).
  tangentOnce(
      "sphere", solveSphere(vec3(0, 1, -5), vec3(0, 0, 1), vec3(0), 1.f, h));
  tangentOnce("cylinder",
      solveCylinder(vec3(1, 0, -5), vec3(0, 0, 1), vec3(0, -1, 0),
          vec3(0, 1, 0), 1.f, 0, h));
  // Cone slant tangent: ray at radius r(z=1)=0.5, moving circumferentially
  // (+y), touches the slant circle once at (0.5,0,1). |perp| is minimized to
  // exactly the radius there -> coincident roots.
  tangentOnce("cone",
      solveCone(vec3(0.5f, -5, 1), vec3(0, 1, 0), vec3(0, 0, 0), vec3(0, 0, 2),
          1.f, 0.f, 0, h));

  // Wall/cap rim coincidence — the DedupT path (distinct-surface, NON-grazing,
  // so only DedupT can merge them). A ray through the axis of a capped cylinder
  // crosses the entry-cap rim (wall u=0 + cap) and the exit-cap rim (wall u=1 +
  // cap): 4 raw candidates that must dedup to exactly 2. This is robustly
  // non-vacuous — the cylinder span check is INCLUSIVE (u in [0,1]), so both
  // rim walls are always emitted; removing DedupT makes this report 4.
  //
  // The cone shares the same emitCrossing/DedupT path, so the cylinder-rim case
  // is the primary (robustly non-vacuous) gate. The cone's own wall/cap
  // coincidence sits exactly on its STRICT (exclusive) axial span boundary,
  // where whether the coincident wall root is emitted is an fp-sign coin toss —
  // so it is checked below with the weaker but never-flaky convex upper bound
  // (n <= 2) rather than an exact == 2.
  rimCount("cylinder-rim",
      solveCylinder(vec3(2, 0, -1), vec3(-1, 0, 1), vec3(0, 0, 0),
          vec3(0, 0, 2), 1.f, CAP_FIRST | CAP_SECOND, h));

  // Cone through both cap rims: ro=(1.5,0,-2) dir=(-0.5,0,2) meets the base-cap
  // rim (r=1 at z=0) at t=1 and the top-cap rim (r=0.5 at z=2) at t=2 — up to 4
  // raw candidates (2 wall on the strict boundary + 2 cap) that DedupT must hold
  // at the convex max of 2. A dedup regression that fails to merge an emitted rim
  // wall against its cap reports 3 or 4 and trips this bound.
  rimAtMost2("cone-rim",
      solveCone(vec3(1.5f, 0, -2), vec3(-0.5f, 0, 2), vec3(0, 0, 0),
          vec3(0, 0, 2), 1.f, 0.5f, CAP_FIRST | CAP_SECOND, h));

  g_failures += bad;
  if (!bad)
    std::printf("[coincident-dedup      ] tangent<=1, wall/cap rim counts once "
                " ok\n");
}

// Two-sided reporting: a ray starting INSIDE a primitive must report a forward
// EXIT crossing whose outward normal faces along the ray (back face). Verifies
// exit crossings survive (not just the nearest external entry).
void testTwoSidedInterior()
{
  PrimHit h[4];
  int bad = 0;
  const auto backFacingExit = [&](int n) {
    const PrimHit *fwd = nearestForward(h, n, 0.f);
    return fwd && glm::dot(fwd->Ng, vec3(0, 0, 1)) > 0.f;
  };

  // Camera inside the sphere.
  if (!backFacingExit(solveSphere(vec3(0, 0, 0), vec3(0, 0, 1), vec3(0), 1.f, h)))
    ++bad;
  // Inside the cylinder (axis +z, radius 1), off-axis.
  if (!backFacingExit(solveCylinder(vec3(0.2f, 0, 1), vec3(0, 0, 1),
          vec3(0, 0, 0), vec3(0, 0, 4), 1.f, CAP_FIRST | CAP_SECOND, h)))
    ++bad;
  // Inside the cone.
  if (!backFacingExit(solveCone(vec3(0.1f, 0, 1), vec3(0, 0, 1), vec3(0, 0, 0),
          vec3(0, 0, 4), 1.f, 0.6f, CAP_FIRST | CAP_SECOND, h)))
    ++bad;

  if (bad) {
    std::printf("FAIL two-sided interior: %d prim(s) lost the exit crossing\n",
        bad);
    g_failures += 1;
  } else
    std::printf("[two-sided-interior    ] inside view sees back face  ok\n");
}

// #333: zero-radius cone apex. Rays through and near the apex must hit cleanly
// (no fray/miss). Mirrors embree cone.h verify() apex rays: apex at p1, rays
// aimed at the exact apex point should hit at the apex.
void testConeApex()
{
  const vec3 p0(0, 0, 0), apex(0, 0, 2);
  const float r0 = 1.f;
  PrimHit h[4];
  int bad = 0;

  // Ray straight down the axis toward the apex from beyond it: with caps off it
  // grazes the apex point (single tangent-like hit or clean miss, never NaN).
  int n = solveCone(vec3(0, 0, 5), vec3(0, 0, -1), p0, apex, r0, 0.f, 0, h);
  CHECK(!anyNonFinite(h, n));

  // Side rays crossing the taper near the apex at many heights: fp32 must agree
  // with the oracle on the near hit (away from the exact apex singularity).
  for (int i = 1; i <= 40; ++i) {
    const float z = 2.f * float(i) / 41.f; // (0,2), radius shrinks to 0
    const vec3 ro(-5, 0, z), rd(1, 0, 0);
    n = solveCone(ro, rd, p0, apex, r0, 0.f, 0, h);
    const auto oh = oracleCone(dvec3(ro), dvec3(rd), dvec3(p0), dvec3(apex),
        double(r0), 0.0, 0);
    const PrimHit *fn = nearestForward(h, n, 0.f);
    const DHit *on = nearestForward(oh, 0.0);
    if (anyNonFinite(h, n)) {
      ++bad;
      continue;
    }
    if (bool(fn) != bool(on)) {
      // near the very tip the radius is ~0: allow disagreement in the last few.
      if (z < 1.9f) {
        std::printf("FAIL apex vis z=%g fp=%s or=%s\n", z,
            fn ? "hit" : "miss", on ? "hit" : "miss");
        ++bad;
      }
      continue;
    }
    if (fn && on && z < 1.85f) {
      if (std::fabs(double(fn->t) - on->t) > 1e-2 * std::max(on->t, 1.0)) {
        std::printf("FAIL apex t z=%g fp=%g or=%g\n", z, double(fn->t), on->t);
        ++bad;
      }
    }
  }
  if (bad)
    g_failures += bad;
  else
    std::printf("[cone-apex #333        ] clean near zero-radius apex  ok\n");
}

// #73: cylinder viewed exactly along its axis. caps=both must give 2 cap
// crossings (the black-bar regression); caps=none gives 0 (see-through).
void testCylinderAxisView()
{
  const vec3 p0(0, 0, 0), p1(0, 0, 2);
  PrimHit h[4];
  int n = solveCylinder(
      vec3(0, 0, -5), vec3(0, 0, 1), p0, p1, 1.f, CAP_FIRST | CAP_SECOND, h);
  CHECK(n == 2);
  CHECK(solveCylinder(vec3(0, 0, -5), vec3(0, 0, 1), p0, p1, 1.f, 0, h) == 0);
  // Far-from-origin axis view (conditioning): same result at 1e4.
  const vec3 F(1e4f, -1e4f, 1e4f);
  n = solveCylinder(F + vec3(0, 0, -5),
      vec3(0, 0, 1),
      F + p0,
      F + p1,
      1.f,
      CAP_FIRST | CAP_SECOND,
      h);
  CHECK(n == 2);
  if (g_failures == 0)
    std::printf("[cylinder-axis #73     ] axis view caps resolve  ok\n");
}

} // namespace

int main()
{
  // Property sweeps across the threat model. Aspect bounds are log10(r/L): the
  // thin (fiber) direction runs to -6 (r/L = 1e-6); the fat direction is capped
  // at +2 (a near-flat disk beyond that is outside what the apex-free fp32 form
  // guarantees — see randomCase).
  sweep("sphere near", P_SPHERE, 0.0, -0.5, 0.5, -1.0, 1.0, 60000, 1);
  sweep("sphere far-1e4", P_SPHERE, 4.0, -0.5, 0.5, -1.0, 1.0, 60000, 2);
  sweep("sphere nonunit-dir", P_SPHERE, 2.0, -3.0, 3.0, -1.0, 1.0, 60000, 3);

  sweep("cylinder near", P_CYL, 0.0, -0.5, 0.5, -2.0, 2.0, 60000, 11);
  sweep("cylinder far-1e4", P_CYL, 4.0, -0.5, 0.5, -2.0, 2.0, 60000, 12);
  sweep("cylinder nonunit-dir", P_CYL, 2.0, -3.0, 3.0, -2.0, 2.0, 60000, 13);
  sweep("cylinder thin-1e6", P_CYL, 1.0, -1.0, 1.0, -6.0, 0.0, 60000, 14);

  sweep("cone near", P_CONE, 0.0, -0.5, 0.5, -2.0, 2.0, 80000, 21);
  sweep("cone far-1e4", P_CONE, 4.0, -0.5, 0.5, -2.0, 2.0, 80000, 22);
  sweep("cone nonunit-dir", P_CONE, 2.0, -3.0, 3.0, -2.0, 2.0, 80000, 23);
  sweep("cone thin-1e6", P_CONE, 1.0, -1.0, 1.0, -6.0, 0.0, 80000, 24);

  testPow2Invariance();
  testScaleInvariance();
  testConeCylinderContinuity();
  testCoincidentDedup();
  testTwoSidedInterior();
  testConeApex();
  testCylinderAxisView();

  if (g_failures) {
    std::printf("%d oracle check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all intersector oracle/property tests passed\n");
  return 0;
}
