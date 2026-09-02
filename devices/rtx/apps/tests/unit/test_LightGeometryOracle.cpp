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

  if (g_failures == 0)
    std::printf("test_LightGeometryOracle: all checks passed\n");
  else
    std::printf("test_LightGeometryOracle: %d failure(s)\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
