/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Oracle and property tests for the analytic ray/rect solver (ADR 0009).
//
// The oracle is an INDEPENDENT fp64 two-triangle formulation — Moller-Trumbore
// over the rect's two triangles. That is precisely the "triangulated proxy" the
// ADR rejected as a shipping representation, because a second area/geometry
// representation can drift from sampleRectLight's own. Used as a *test oracle*
// it buys the confidence without shipping the divergence: different algebra, so
// a formulation bug cannot hide in a shared blind spot.
//
// The threat model is deliberately unkind: sheared (non-perpendicular)
// parallelograms, non-unit directions, far-from-origin coordinates, extreme
// aspect ratios, and grazing rays.
//
// No CUDA/OptiX/GPU. Returns nonzero on any failure.

#include "gpu/lightGeometry.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdio>
#include <random>

using namespace visrtx;
using glm::dvec3;

static int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);              \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

// ------------------------------------------------------------------ oracle ---

struct OracleHit
{
  bool hit;
  double t;
};

// Moller-Trumbore in fp64 against one triangle.
static bool triHit(const dvec3 &org,
    const dvec3 &dir,
    const dvec3 &v0,
    const dvec3 &v1,
    const dvec3 &v2,
    double &tOut)
{
  const dvec3 e1 = v1 - v0;
  const dvec3 e2 = v2 - v0;
  const dvec3 p = glm::cross(dir, e2);
  const double det = glm::dot(e1, p);
  if (std::fabs(det) < 1e-300)
    return false;
  const double invDet = 1.0 / det;
  const dvec3 tv = org - v0;
  const double u = glm::dot(tv, p) * invDet;
  if (u < 0.0 || u > 1.0)
    return false;
  const dvec3 q = glm::cross(tv, e1);
  const double v = glm::dot(dir, q) * invDet;
  if (v < 0.0 || u + v > 1.0)
    return false;
  const double t = glm::dot(e2, q) * invDet;
  if (!(t > 0.0))
    return false;
  tOut = t;
  return true;
}

// The rect as two triangles: (p, p+e1, p+e1+e2) and (p, p+e1+e2, p+e2).
static OracleHit oracleRect(
    const RectLightGPUData &rect, const vec3 &orgf, const vec3 &dirf)
{
  const dvec3 org(orgf), dir(dirf);
  const dvec3 p(rect.position), e1(rect.edge1), e2(rect.edge2);
  const dvec3 a = p, b = p + e1, c = p + e1 + e2, d = p + e2;

  OracleHit out{false, 0.0};
  double t;
  if (triHit(org, dir, a, b, c, t)) {
    out.hit = true;
    out.t = t;
  }
  if (triHit(org, dir, a, c, d, t)) {
    if (!out.hit || t < out.t) {
      out.hit = true;
      out.t = t;
    }
  }
  return out;
}

static RectLightGPUData makeRect(
    const vec3 &position, const vec3 &e1, const vec3 &e2)
{
  RectLightGPUData r{};
  r.position = position;
  r.edge1 = e1;
  r.edge2 = e2;
  r.intensity = 1.0f;
  r.side.front = 1;
  r.side.back = 0;
  const float area = length(cross(e1, e2));
  r.oneOverArea = area > 0.0f ? 1.0f / area : 1.0f;
  return r;
}

int main()
{
  // A unit quad in the XZ plane at the origin, spanning [0,1]^2.
  const RectLightGPUData unit =
      makeRect(vec3(0.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f));
  const vec3 down(0.0f, -1.0f, 0.0f);
  const vec3 up(0.0f, 1.0f, 0.0f);

  // --- targeted: hits strictly inside ---------------------------------------
  {
    // Straight down through the centre from 2 units up.
    const RectIntersection h =
        intersectRect(unit, vec3(0.5f, 2.0f, 0.5f), down);
    CHECK(h.hit);
    CHECK(std::fabs(h.t - 2.0f) < 1e-5f);
    CHECK(std::fabs(h.uv.x - 0.5f) < 1e-5f);
    CHECK(std::fabs(h.uv.y - 0.5f) < 1e-5f);

    // uv maps to the edges in the expected orientation: u along edge1 (X),
    // v along edge2 (Z). Swapping them would pass the centre test above.
    const RectIntersection q =
        intersectRect(unit, vec3(0.25f, 1.0f, 0.75f), down);
    CHECK(q.hit);
    CHECK(std::fabs(q.uv.x - 0.25f) < 1e-5f);
    CHECK(std::fabs(q.uv.y - 0.75f) < 1e-5f);
  }

  // --- targeted: misses beyond each of the four edges independently ---------
  {
    CHECK(!intersectRect(unit, vec3(-0.01f, 1.0f, 0.5f), down).hit); // u < 0
    CHECK(!intersectRect(unit, vec3(1.01f, 1.0f, 0.5f), down).hit); // u > 1
    CHECK(!intersectRect(unit, vec3(0.5f, 1.0f, -0.01f), down).hit); // v < 0
    CHECK(!intersectRect(unit, vec3(0.5f, 1.0f, 1.01f), down).hit); // v > 1
  }

  // --- targeted: edges and corners are consistent and finite ----------------
  // No claim about inclusive vs exclusive; only that the result is a clean
  // decision with in-range uv, never a NaN or a half-populated hit.
  {
    const vec3 probes[] = {vec3(0.0f, 1.0f, 0.5f),
        vec3(1.0f, 1.0f, 0.5f),
        vec3(0.5f, 1.0f, 0.0f),
        vec3(0.5f, 1.0f, 1.0f),
        vec3(0.0f, 1.0f, 0.0f),
        vec3(1.0f, 1.0f, 1.0f),
        vec3(0.0f, 1.0f, 1.0f),
        vec3(1.0f, 1.0f, 0.0f)};
    for (const vec3 &o : probes) {
      const RectIntersection h = intersectRect(unit, o, down);
      if (h.hit) {
        CHECK(std::isfinite(h.t));
        CHECK(h.t > 0.0f);
        CHECK(h.uv.x >= 0.0f && h.uv.x <= 1.0f);
        CHECK(h.uv.y >= 0.0f && h.uv.y <= 1.0f);
      }
    }
  }

  // --- targeted: parallel, in-plane, behind, and backfacing rays ------------
  {
    // Parallel to the plane, above it.
    CHECK(!intersectRect(unit, vec3(0.5f, 1.0f, 0.5f), vec3(1.0f, 0.0f, 0.0f))
            .hit);
    // Lying exactly in the plane.
    CHECK(!intersectRect(unit, vec3(-1.0f, 0.0f, 0.5f), vec3(1.0f, 0.0f, 0.0f))
            .hit);
    // Pointing away: the rect is behind the origin.
    CHECK(!intersectRect(unit, vec3(0.5f, 1.0f, 0.5f), up).hit);
    // Origin exactly on the rect, pointing away.
    CHECK(!intersectRect(unit, vec3(0.5f, 0.0f, 0.5f), up).hit);
    // Backfacing (from below, pointing up) still hits: culling is the caller's
    // job via the side predicate, not the solver's.
    const RectIntersection back =
        intersectRect(unit, vec3(0.5f, -2.0f, 0.5f), up);
    CHECK(back.hit);
    CHECK(std::fabs(back.t - 2.0f) < 1e-5f);
  }

  // --- targeted: degenerate rects never hit ---------------------------------
  {
    const RectLightGPUData zeroEdge =
        makeRect(vec3(0.0f), vec3(0.0f), vec3(0.0f, 0.0f, 1.0f));
    CHECK(!intersectRect(zeroEdge, vec3(0.5f, 1.0f, 0.5f), down).hit);

    const RectLightGPUData parallelEdges =
        makeRect(vec3(0.0f), vec3(1.0f, 0.0f, 0.0f), vec3(2.0f, 0.0f, 0.0f));
    CHECK(!intersectRect(parallelEdges, vec3(0.5f, 1.0f, 0.0f), down).hit);

    const RectLightGPUData bothZero =
        makeRect(vec3(0.0f), vec3(0.0f), vec3(0.0f));
    CHECK(!intersectRect(bothZero, vec3(0.0f, 1.0f, 0.0f), down).hit);
  }

  // --- targeted: sheared parallelogram --------------------------------------
  // The case the Gram-matrix solve exists for. With edges (1,0,0) and (1,0,1)
  // the rect covers x in [v, 1+v] at height z=v. A dot/|e|^2 formulation
  // mis-bounds this; both probes below discriminate.
  {
    const RectLightGPUData sheared =
        makeRect(vec3(0.0f), vec3(1.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 1.0f));
    // Inside: at z=0.5 the span is x in [0.5, 1.5].
    CHECK(intersectRect(sheared, vec3(1.0f, 1.0f, 0.5f), down).hit);
    // Outside, but inside the axis-aligned bounding box.
    CHECK(!intersectRect(sheared, vec3(0.2f, 1.0f, 0.5f), down).hit);
    // Cross-check both against the oracle.
    CHECK(oracleRect(sheared, vec3(1.0f, 1.0f, 0.5f), down).hit);
    CHECK(!oracleRect(sheared, vec3(0.2f, 1.0f, 0.5f), down).hit);
  }

  // --- property: uv round-trips to the hit position -------------------------
  // position + u*e1 + v*e2 must reproduce the hit point. This is the property
  // the deposit relies on when it feeds uv to the pdf leaf with no
  // reconstruction.
  {
    std::mt19937 g(4242);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (int i = 0; i < 20000; ++i) {
      const RectLightGPUData r = makeRect(vec3(uni(g), uni(g), uni(g)),
          vec3(uni(g), uni(g), uni(g)),
          vec3(uni(g), uni(g), uni(g)));
      const vec3 org(uni(g) * 3.0f, uni(g) * 3.0f, uni(g) * 3.0f);
      const vec3 dir(uni(g), uni(g), uni(g));
      if (length(dir) < 1e-3f)
        continue;
      const RectIntersection h = intersectRect(r, org, dir);
      if (!h.hit)
        continue;
      const vec3 fromUv = r.position + h.uv.x * r.edge1 + h.uv.y * r.edge2;
      const vec3 fromT = org + h.t * dir;
      CHECK(length(fromUv - fromT) < 1e-3f * glm::max(1.0f, length(fromT)));
    }
  }

  // --- oracle: randomized agreement, unit-scale ------------------------------
  {
    std::mt19937 g(777);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    int compared = 0, hits = 0, disagree = 0, tMismatch = 0;
    for (int i = 0; i < 200000; ++i) {
      const RectLightGPUData r = makeRect(vec3(uni(g), uni(g), uni(g)),
          vec3(uni(g), uni(g), uni(g)),
          vec3(uni(g), uni(g), uni(g)));
      const vec3 org(uni(g) * 2.0f, uni(g) * 2.0f, uni(g) * 2.0f);
      const vec3 dir(uni(g), uni(g), uni(g));
      if (length(dir) < 1e-3f || length(cross(r.edge1, r.edge2)) < 1e-3f)
        continue;

      const RectIntersection mine = intersectRect(r, org, dir);
      const OracleHit ref = oracleRect(r, org, dir);
      ++compared;
      if (ref.hit)
        ++hits;

      // Classification may legitimately differ within a hair of the boundary;
      // count only decisive disagreements, where the oracle's uv is clear of
      // the edge.
      if (mine.hit != ref.hit) {
        if (ref.hit) {
          // Oracle says hit: only a fault if comfortably interior.
          const vec3 p = org + float(ref.t) * dir;
          const vec3 d = p - r.position;
          const float e11 = dot(r.edge1, r.edge1);
          const float e12 = dot(r.edge1, r.edge2);
          const float e22 = dot(r.edge2, r.edge2);
          const float det = e11 * e22 - e12 * e12;
          const float u = (dot(d, r.edge1) * e22 - dot(d, r.edge2) * e12) / det;
          const float v = (dot(d, r.edge2) * e11 - dot(d, r.edge1) * e12) / det;
          if (u > 1e-3f && u < 1.0f - 1e-3f && v > 1e-3f && v < 1.0f - 1e-3f)
            ++disagree;
        } else {
          if (mine.uv.x > 1e-3f && mine.uv.x < 1.0f - 1e-3f && mine.uv.y > 1e-3f
              && mine.uv.y < 1.0f - 1e-3f)
            ++disagree;
        }
      } else if (mine.hit && ref.hit) {
        const double rel =
            std::fabs(double(mine.t) - ref.t) / std::fmax(1.0, ref.t);
        if (rel > 1e-4)
          ++tMismatch;
      }
    }
    std::printf(
        "  oracle(unit): %d compared, %d oracle-hits, %d classification, %d t\n",
        compared,
        hits,
        disagree,
        tMismatch);
    CHECK(hits > 1000); // the sampler must actually be exercising hits
    CHECK(disagree == 0);
    CHECK(tMismatch == 0);
  }

  // --- oracle: far from origin and extreme aspect ratios --------------------
  // fp32 cancellation is worst when the rect sits far from the origin or is a
  // long thin slat; the oracle stays in fp64 throughout.
  {
    std::mt19937 g(31337);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    int compared = 0, hits = 0, disagree = 0, tMismatch = 0;
    for (int i = 0; i < 100000; ++i) {
      const vec3 centre(uni(g) * 500.0f, uni(g) * 500.0f, uni(g) * 500.0f);
      const float aspect = std::pow(10.0f, uni(g) * 2.0f); // 0.01 .. 100
      const RectLightGPUData r = makeRect(
          centre, vec3(aspect, 0.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f / aspect));
      // Aim at a point known to be on the rect so hits are common.
      const float tu = 0.5f + uni(g) * 0.45f;
      const float tv = 0.5f + uni(g) * 0.45f;
      const vec3 target = r.position + tu * r.edge1 + tv * r.edge2;
      const vec3 org = target + vec3(uni(g), 1.0f + std::fabs(uni(g)), uni(g));
      const vec3 dir = target - org;
      if (length(dir) < 1e-3f)
        continue;

      const RectIntersection mine = intersectRect(r, org, dir);
      const OracleHit ref = oracleRect(r, org, dir);
      ++compared;
      if (ref.hit)
        ++hits;
      if (mine.hit != ref.hit)
        ++disagree;
      else if (mine.hit && ref.hit) {
        const double rel =
            std::fabs(double(mine.t) - ref.t) / std::fmax(1.0, ref.t);
        if (rel > 1e-3)
          ++tMismatch;
      }
    }
    std::printf(
        "  oracle(far/aspect): %d compared, %d oracle-hits, %d classification, %d t\n",
        compared,
        hits,
        disagree,
        tMismatch);
    CHECK(hits > 1000);
    CHECK(disagree == 0);
    CHECK(tMismatch == 0);
  }

  // --- metamorphic: exact power-of-two scaling ------------------------------
  // Scaling the whole configuration by a power of two is exact in fp32, so the
  // hit decision must be identical and t must scale exactly.
  {
    std::mt19937 g(9001);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (int i = 0; i < 20000; ++i) {
      const RectLightGPUData r = makeRect(vec3(uni(g), uni(g), uni(g)),
          vec3(uni(g), uni(g), uni(g)),
          vec3(uni(g), uni(g), uni(g)));
      const vec3 org(uni(g) * 2.0f, uni(g) * 2.0f, uni(g) * 2.0f);
      const vec3 dir(uni(g), uni(g), uni(g));
      if (length(dir) < 1e-3f || length(cross(r.edge1, r.edge2)) < 1e-3f)
        continue;

      const RectIntersection a = intersectRect(r, org, dir);

      const float k = 4.0f;
      const RectLightGPUData rs =
          makeRect(r.position * k, r.edge1 * k, r.edge2 * k);
      const RectIntersection b = intersectRect(rs, org * k, dir);

      CHECK(a.hit == b.hit);
      if (a.hit && b.hit) {
        CHECK(std::fabs(a.t * k - b.t) <= 1e-4f * std::fmax(1.0f, b.t));
        CHECK(std::fabs(a.uv.x - b.uv.x) < 1e-4f);
        CHECK(std::fabs(a.uv.y - b.uv.y) < 1e-4f);
      }
    }
  }

  // --- metamorphic: direction scaling reparameterizes t ----------------------
  // intersectRect takes a non-unit direction; doubling it must halve t and
  // leave the hit decision and uv untouched.
  {
    std::mt19937 g(2718);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (int i = 0; i < 20000; ++i) {
      const RectLightGPUData r = makeRect(vec3(uni(g), uni(g), uni(g)),
          vec3(uni(g), uni(g), uni(g)),
          vec3(uni(g), uni(g), uni(g)));
      const vec3 org(uni(g) * 2.0f, uni(g) * 2.0f, uni(g) * 2.0f);
      const vec3 dir(uni(g), uni(g), uni(g));
      if (length(dir) < 1e-3f || length(cross(r.edge1, r.edge2)) < 1e-3f)
        continue;

      const RectIntersection a = intersectRect(r, org, dir);
      const RectIntersection b = intersectRect(r, org, dir * 2.0f);
      CHECK(a.hit == b.hit);
      if (a.hit && b.hit) {
        CHECK(std::fabs(a.t * 0.5f - b.t) <= 1e-4f * std::fmax(1.0f, b.t));
        CHECK(std::fabs(a.uv.x - b.uv.x) < 1e-4f);
        CHECK(std::fabs(a.uv.y - b.uv.y) < 1e-4f);
      }
    }
  }

  // --- THE MIS IDENTITY -----------------------------------------------------
  // The single most important property in ADR 0009.
  //
  // NEE samples a point on the light and reports a density. A BSDF continuation
  // toward that same point hits the proxy and must reconstruct the SAME
  // density, or the balance heuristic weights the deposit against a density
  // nothing sampled and the render is biased.
  //
  // This exercises the full round trip -- sample, shoot, intersect, reconstruct
  // -- rather than just calling the shared leaf twice, so a drift introduced by
  // the solver or by the uv handoff is caught too.
  {
    std::mt19937 g(20260009);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    // Two effects are excluded, both measured rather than assumed:
    //
    // 1. Grazing incidence. The density carries dist^2/cosTheta, so as cosTheta
    //    goes to 0 it is ill-conditioned BY CONSTRUCTION: a 1-ulp difference in
    //    the hit point is amplified without bound. Empirically every deviation
    //    above 1e-3 has cosTheta < 1.2e-4, while the 133k cases with
    //    cosTheta > 1e-3 agree to 1.3e-4. Both facts are asserted below, so the
    //    exclusion cannot quietly grow to hide a real drift.
    //
    // 2. Points sampled on the rect's boundary to within an ulp. NEE can sample
    //    v = 1e-6 and the round trip lands at v = -9e-8, just outside. That is
    //    a boundary classification, not a density disagreement.
    constexpr float kWellConditionedCos = 1e-3f;
    constexpr float kBoundaryMargin = 1e-4f;

    int tested = 0, mismatches = 0;
    int grazingSkipped = 0, boundarySkipped = 0;
    double worstRel = 0.0;
    float worstMismatchCos = 0.0f;

    for (int i = 0; i < 200000; ++i) {
      RectLightGPUData r = makeRect(vec3(uni(g), uni(g), uni(g)),
          vec3(uni(g), uni(g), uni(g)),
          vec3(uni(g), uni(g), uni(g)));
      // Exercise every side configuration.
      const int sideSel = int(g() % 3);
      r.side.front = (sideSel == 0 || sideSel == 2) ? 1 : 0;
      r.side.back = (sideSel == 1 || sideSel == 2) ? 1 : 0;

      const float area = length(cross(r.edge1, r.edge2));
      if (area < 1e-2f)
        continue;

      const vec3 origin(uni(g) * 3.0f, uni(g) * 3.0f, uni(g) * 3.0f);

      // --- NEE side: sample a point uniformly on the rect.
      const vec2 uv(unit(g), unit(g));
      const vec3 sampled = r.position + r.edge1 * uv.x + r.edge2 * uv.y;

      const mat4 identity(1.0f);
      const RectFrame frame = rectFrame(r, identity);
      const RectPointRelation nee =
          rectRelateToPoint(r, frame, origin, sampled);
      if (!(nee.solidAnglePdf > 0.0f))
        continue; // the light does not emit toward this point; NEE reports 0

      const bool onBoundary = uv.x < kBoundaryMargin
          || uv.x > 1.0f - kBoundaryMargin || uv.y < kBoundaryMargin
          || uv.y > 1.0f - kBoundaryMargin;

      // --- Hit side: shoot toward that point and reconstruct from the hit.
      const RectIntersection isect = intersectRect(r, origin, nee.dir);
      if (!isect.hit) {
        // The sampled point is on the rect and the light emits toward the
        // origin, so a ray aimed at it must hit -- UNLESS the sample sat on the
        // boundary to within an ulp, where the round trip can land a hair
        // outside. Anywhere else a miss is a real fault: NEE would report a
        // positive density for a direction the hit side says cannot reach the
        // light, and MIS could never rebalance it.
        if (onBoundary)
          ++boundarySkipped;
        else
          ++mismatches;
        continue;
      }

      const vec3 hitPoint = origin + isect.t * nee.dir;
      const RectPointRelation hit =
          rectRelateToPoint(r, frame, origin, hitPoint);

      const double rel =
          std::fabs(double(hit.solidAnglePdf) - double(nee.solidAnglePdf))
          / std::fmax(1e-30, double(nee.solidAnglePdf));

      if (nee.cosTheta < kWellConditionedCos) {
        ++grazingSkipped;
        // Still track how far the excluded region strays, so the exclusion is
        // characterized rather than a blind spot.
        if (rel > 1e-3)
          worstMismatchCos = std::fmax(worstMismatchCos, nee.cosTheta);
        continue;
      }

      ++tested;
      worstRel = std::fmax(worstRel, rel);
      // fp32 round trip through the solver; beyond this is drift, not rounding.
      if (rel > 1e-3) {
        ++mismatches;
        worstMismatchCos = std::fmax(worstMismatchCos, nee.cosTheta);
      }
    }

    std::printf(
        "  MIS identity: %d tested, %d mismatches, worst rel %.3e"
        " (skipped %d grazing, %d boundary; worst mismatch cos %.3e)\n",
        tested,
        mismatches,
        worstRel,
        grazingSkipped,
        boundarySkipped,
        worstMismatchCos);

    CHECK(tested > 10000);
    // The identity holds exactly where the density is well conditioned.
    CHECK(mismatches == 0);
    // And it holds TIGHTLY there -- not merely inside the 1e-3 gate.
    CHECK(worstRel < 1e-3);
    // The grazing exclusion must stay a thin sliver. If a future change starts
    // disagreeing at moderate angles, this catches it even though those cases
    // are skipped above.
    CHECK(worstMismatchCos < kWellConditionedCos);
    // Neither exclusion may swallow the bulk of the samples.
    CHECK(grazingSkipped < tested / 10);
    CHECK(boundarySkipped < tested / 100);
  }

  // --- Ring: analytic ray/annulus solver ------------------------------------
  // The inner hole is the ring's analogue of the rectangle's edge bounds: a ray
  // through it must MISS. A ring that renders as a full disk is the obvious
  // failure, and it is exactly what a missing inner-radius test produces.
  {
    RingLightGPUData ring{};
    ring.position = vec3(0.0f);
    ring.direction = vec3(0.0f, -1.0f, 0.0f);
    ring.cosOuterAngle = 0.0f;
    ring.cosInnerAngle = 1.0f;
    ring.radius = 2.0f;
    ring.innerRadius = 1.0f;
    ring.intensity = 1.0f;
    ring.oneOverArea = 1.0f / (kPi * (4.0f - 1.0f));

    const mat4 identity(1.0f);
    const vec3 down(0.0f, -1.0f, 0.0f);
    const vec3 up(0.0f, 1.0f, 0.0f);

    // On the annulus: r = 1.5, between inner 1 and outer 2.
    const RingIntersection onBand =
        intersectRing(ring, identity, vec3(1.5f, 3.0f, 0.0f), down);
    CHECK(onBand.hit);
    CHECK(std::fabs(onBand.t - 3.0f) < 1e-5f);
    CHECK(std::fabs(onBand.radius - 1.5f) < 1e-5f);

    // Through the inner hole: must miss.
    CHECK(!intersectRing(ring, identity, vec3(0.5f, 3.0f, 0.0f), down).hit);
    CHECK(!intersectRing(ring, identity, vec3(0.0f, 3.0f, 0.0f), down).hit);
    // Outside the outer radius: must miss.
    CHECK(!intersectRing(ring, identity, vec3(2.5f, 3.0f, 0.0f), down).hit);

    // Radially symmetric: the same radius hits from any azimuth.
    for (int i = 0; i < 16; ++i) {
      const float phi = kTwoPi * float(i) / 16.0f;
      const vec3 o(1.5f * std::cos(phi), 3.0f, 1.5f * std::sin(phi));
      CHECK(intersectRing(ring, identity, o, down).hit);
      const vec3 inner(0.5f * std::cos(phi), 3.0f, 0.5f * std::sin(phi));
      CHECK(!intersectRing(ring, identity, inner, down).hit);
    }

    // Boundary radii are consistent and finite, whichever way they classify.
    for (float r : {1.0f, 2.0f}) {
      const RingIntersection h =
          intersectRing(ring, identity, vec3(r, 3.0f, 0.0f), down);
      if (h.hit) {
        CHECK(std::isfinite(h.t));
        CHECK(h.radius >= ring.innerRadius - 1e-4f);
        CHECK(h.radius <= ring.radius + 1e-4f);
      }
    }

    // Parallel, in-plane, and behind-the-origin rays.
    CHECK(!intersectRing(
        ring, identity, vec3(1.5f, 3.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f))
            .hit);
    CHECK(!intersectRing(
        ring, identity, vec3(-3.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f))
            .hit);
    CHECK(!intersectRing(ring, identity, vec3(1.5f, 3.0f, 0.0f), up).hit);

    // Backfacing still hits: culling is the caller's job, not the solver's.
    CHECK(intersectRing(ring, identity, vec3(1.5f, -3.0f, 0.0f), up).hit);

    // A full disk (innerRadius 0) has no hole.
    RingLightGPUData disk = ring;
    disk.innerRadius = 0.0f;
    CHECK(intersectRing(disk, identity, vec3(0.0f, 3.0f, 0.0f), down).hit);

    // Degenerate rings never hit: a ring with no area emits nothing, so a hit
    // would deposit radiance NEE never sampled. The aimed-dead-centre ray below
    // would otherwise land exactly on r == inner == outer, which the inclusive
    // radial band test accepts.
    RingLightGPUData zeroRadius = ring;
    zeroRadius.radius = 0.0f;
    zeroRadius.innerRadius = 0.0f;
    CHECK(
        !intersectRing(zeroRadius, identity, vec3(0.0f, 3.0f, 0.0f), down).hit);

    RingLightGPUData empty = ring;
    empty.innerRadius = empty.radius;
    // Aimed straight at the collapsed radius, the one place a hit could sneak
    // through.
    CHECK(!intersectRing(empty, identity, vec3(2.0f, 3.0f, 0.0f), down).hit);
    CHECK(!intersectRing(empty, identity, vec3(1.5f, 3.0f, 0.0f), down).hit);

    // An inner radius that OVERSHOOTS the outer one is degenerate the same way.
    RingLightGPUData inverted = ring;
    inverted.innerRadius = inverted.radius * 2.0f;
    CHECK(!intersectRing(inverted, identity, vec3(1.5f, 3.0f, 0.0f), down).hit);

    // A real annulus is still inclusive at its rim: the degenerate guard must
    // reject empty rings, not trim valid ones.
    CHECK(intersectRing(ring, identity, vec3(2.0f, 3.0f, 0.0f), down).hit);
    CHECK(intersectRing(ring, identity, vec3(1.0f, 3.0f, 0.0f), down).hit);
  }

  // --- Ring: exact under any affine instance transform ----------------------
  // The solver works in the light's OBJECT frame, so the hittable set is the
  // image of the set NEE samples -- for a tilted axis under a non-uniform scale
  // included. That is the case a world-space circle of averaged radius gets
  // wrong (the disk is an ellipse), and the case a forward-transformed normal
  // gets wrong (it is not perpendicular to the transformed disk).
  {
    RingLightGPUData ring{};
    ring.position = vec3(0.3f, -0.2f, 0.1f);
    ring.direction = normalize(vec3(1.0f, 1.0f, 0.3f));
    ring.cosOuterAngle = -1.0f;
    ring.cosInnerAngle = 1.0f;
    ring.radius = 2.0f;
    ring.innerRadius = 0.5f;
    ring.intensity = 1.0f;
    ring.oneOverArea = 1.0f / (kPi * (4.0f - 0.25f));

    const mat4 xfm = glm::translate(mat4(1.0f), vec3(1.0f, 2.0f, -3.0f))
        * glm::rotate(mat4(1.0f), glm::radians(31.0f), vec3(0.2f, 1.0f, 0.4f))
        * glm::scale(mat4(1.0f), vec3(1.0f, 2.5f, 0.4f));
    const mat4 worldToObject = glm::inverse(xfm);

    const vec3 dObj = normalize(ring.direction);
    const vec3 ref = std::fabs(dObj.x) < 0.9f ? vec3(1.0f, 0.0f, 0.0f)
                                              : vec3(0.0f, 1.0f, 0.0f);
    const vec3 b0 = normalize(cross(dObj, ref));
    const vec3 b1 = cross(dObj, b0);

    // The world axis is the unit normal of the TRANSFORMED disk, oriented
    // with the forward-transformed direction.
    const vec3 axis = ringWorldAxis(ring, xfm);
    CHECK(std::fabs(length(axis) - 1.0f) < 1e-5f);
    CHECK(std::fabs(dot(axis, normalize(xfmVec(xfm, b0)))) < 1e-4f);
    CHECK(std::fabs(dot(axis, normalize(xfmVec(xfm, b1)))) < 1e-4f);
    CHECK(dot(axis, xfmVec(xfm, dObj)) > 0.0f);
    // ...and the forward-transformed direction is measurably NOT that normal
    // for this transform, so a regression to xfmVec(xfm, direction) fails here.
    const vec3 naive = normalize(xfmVec(xfm, dObj));
    CHECK(std::fabs(dot(naive, normalize(xfmVec(xfm, b0)))) > 1e-2f
        || std::fabs(dot(naive, normalize(xfmVec(xfm, b1)))) > 1e-2f);

    // Round trip: every point NEE can sample is hit, at that point, with the
    // object-space radius it was sampled at.
    std::mt19937 g(4242);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    int roundTrips = 0;
    for (int i = 0; i < 256; ++i) {
      const float rr = std::sqrt(unit(g)
              * (ring.radius * ring.radius
                  - ring.innerRadius * ring.innerRadius)
          + ring.innerRadius * ring.innerRadius);
      const float phi = kTwoPi * unit(g);
      const vec3 sampledObj =
          ring.position + b0 * (rr * std::cos(phi)) + b1 * (rr * std::sin(phi));
      const vec3 sampledWorld = xfmPoint(xfm, sampledObj);
      const vec3 origin = sampledWorld + axis * 5.0f;
      const vec3 dir = normalize(sampledWorld - origin);
      const RingIntersection h =
          intersectRing(ring, worldToObject, origin, dir);
      CHECK(h.hit);
      if (!h.hit)
        continue;
      const vec3 hitPoint = origin + h.t * dir;
      CHECK(length(hitPoint - sampledWorld) < 1e-3f);
      CHECK(std::fabs(h.radius - rr) < 1e-3f);
      ++roundTrips;
    }
    CHECK(roundTrips == 256);

    // Just outside the annulus in OBJECT units, along either in-plane basis
    // direction: a miss, however much the instance stretches it in world.
    for (const vec3 &b : {b0, b1}) {
      for (const float rr : {ring.radius * 1.05f, ring.innerRadius * 0.95f}) {
        const vec3 pWorld = xfmPoint(xfm, ring.position + b * rr);
        const vec3 origin = pWorld + axis * 5.0f;
        CHECK(!intersectRing(
            ring, worldToObject, origin, normalize(pWorld - origin))
                .hit);
      }
    }
  }

  // --- Sphere: exact under any affine instance transform --------------------
  // The same round trip as the ring, on the ellipsoid the sampler actually
  // scatters over. A ray from outside along the line through the centre meets
  // a convex body first at the near boundary point, i.e. exactly at the
  // sampled point.
  {
    const mat4 xfm = glm::translate(mat4(1.0f), vec3(1.0f, 2.0f, -3.0f))
        * glm::rotate(mat4(1.0f), glm::radians(31.0f), vec3(0.2f, 1.0f, 0.4f))
        * glm::scale(mat4(1.0f), vec3(1.0f, 2.5f, 0.4f));
    const mat4 worldToObject = glm::inverse(xfm);
    std::mt19937 g(4243);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    SphereLightGPUData sphere{};
    sphere.position = vec3(-0.4f, 0.1f, 0.25f);
    sphere.radius = 1.5f;
    sphere.intensity = 1.0f;
    const vec3 centreWorld = xfmPoint(xfm, sphere.position);
    int sphereRoundTrips = 0;
    for (int i = 0; i < 256; ++i) {
      const float z = 1.0f - 2.0f * unit(g);
      const float rxy = std::sqrt(std::max(0.0f, 1.0f - z * z));
      const float phi = kTwoPi * unit(g);
      const vec3 onUnit(rxy * std::cos(phi), rxy * std::sin(phi), z);
      const vec3 sampledWorld =
          xfmPoint(xfm, sphere.position + onUnit * sphere.radius);
      const vec3 outward = sampledWorld - centreWorld;
      const vec3 origin = sampledWorld + outward * 4.0f;
      const vec3 dir = normalize(sampledWorld - origin);
      const SphereIntersection h =
          intersectSphereLight(sphere, worldToObject, origin, dir);
      CHECK(h.hit);
      if (!h.hit)
        continue;
      CHECK(length((origin + h.t * dir) - sampledWorld) < 1e-3f);
      ++sphereRoundTrips;
    }
    CHECK(sphereRoundTrips == 256);
  }

  // --- Ring: the MIS identity -----------------------------------------------
  // Same round trip as the rect case: sample a point on the annulus, shoot at
  // it, intersect, and reconstruct the density from the hit.
  {
    const mat4 identity(1.0f);
    std::mt19937 g(770009);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    constexpr float kWellConditionedCos = 1e-3f;
    int tested = 0, mismatches = 0, grazingSkipped = 0;
    double worstRel = 0.0;

    for (int i = 0; i < 100000; ++i) {
      RingLightGPUData ring{};
      ring.position = vec3(uni(g), uni(g), uni(g));
      ring.direction = vec3(uni(g), uni(g), uni(g));
      if (length(ring.direction) < 1e-2f)
        continue;
      ring.innerRadius = unit(g) * 0.8f;
      ring.radius = ring.innerRadius + 0.2f + unit(g);
      ring.intensity = 1.0f;
      ring.oneOverArea = 1.0f
          / (kPi
              * (ring.radius * ring.radius
                  - ring.innerRadius * ring.innerRadius));
      // Wide cone so most samples are inside it; the falloff itself is covered
      // by the dedicated attenuation test.
      ring.cosOuterAngle = 0.0f;
      ring.cosInnerAngle = 1.0f;

      const vec3 axis = normalize(ring.direction);
      const vec3 origin(uni(g) * 3.0f, uni(g) * 3.0f, uni(g) * 3.0f);

      // Sample a point on the annulus, area-uniformly, as the sampler does.
      const float phi = kTwoPi * unit(g);
      const float rr = std::sqrt(unit(g)
              * (ring.radius * ring.radius
                  - ring.innerRadius * ring.innerRadius)
          + ring.innerRadius * ring.innerRadius);
      // Any orthonormal basis of the disk plane works: the density is radially
      // symmetric, so the basis choice cannot change it.
      const vec3 ref = std::fabs(axis.x) < 0.9f ? vec3(1.0f, 0.0f, 0.0f)
                                                : vec3(0.0f, 1.0f, 0.0f);
      const vec3 b0 = normalize(cross(axis, ref));
      const vec3 b1 = cross(axis, b0);
      const vec3 sampled =
          ring.position + b0 * (rr * std::cos(phi)) + b1 * (rr * std::sin(phi));

      const RingPointRelation nee =
          ringRelateToPoint(ring, axis, origin, sampled);
      if (!(nee.solidAnglePdf > 0.0f))
        continue;

      const RingIntersection isect =
          intersectRing(ring, identity, origin, nee.dir);
      if (!isect.hit) {
        // Only a fault away from the annulus edges, where an ulp can flip the
        // radial classification.
        const float edgeSlack = 1e-3f;
        if (rr > ring.innerRadius + edgeSlack && rr < ring.radius - edgeSlack)
          ++mismatches;
        continue;
      }

      const vec3 hitPoint = origin + isect.t * nee.dir;
      const RingPointRelation hit =
          ringRelateToPoint(ring, axis, origin, hitPoint);

      if (nee.cosTheta < kWellConditionedCos) {
        ++grazingSkipped;
        continue;
      }

      ++tested;
      const double rel =
          std::fabs(double(hit.solidAnglePdf) - double(nee.solidAnglePdf))
          / std::fmax(1e-30, double(nee.solidAnglePdf));
      worstRel = std::fmax(worstRel, rel);
      if (rel > 1e-3)
        ++mismatches;
    }

    std::printf(
        "  ring MIS identity: %d tested, %d mismatches, worst rel %.3e"
        " (skipped %d grazing)\n",
        tested,
        mismatches,
        worstRel,
        grazingSkipped);
    CHECK(tested > 5000);
    CHECK(mismatches == 0);
    CHECK(worstRel < 1e-3);
  }

  if (g_failures == 0)
    std::printf("test_LightGeometryOracle: all checks passed\n");
  else
    std::printf("test_LightGeometryOracle: %d failure(s)\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
