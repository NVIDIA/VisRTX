/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Host unit tests for the shared rect/ring area-light leaves (ADR 0009). These
// are the functions BOTH next-event estimation and the hit-side deposit call,
// so a bug here desyncs MIS and biases the image in a way that is very hard to
// see in a render. Testing them directly is the point of keeping
// lightGeometry.h CUDA-free.
//
// Radiance is asserted to be Lambertian (independent of distance and viewing
// angle) because the cosine belongs to the pdf, not the radiance — getting that
// split wrong double-applies the cosine.

#include "gpu/lightGeometry.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdio>
#include <random>

using namespace visrtx;

static int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);              \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

static bool nearf(float a, float b, float eps = 1e-5f)
{
  return std::fabs(a - b) <= eps * std::fmax(1.0f, std::fabs(b));
}

static bool isFinite(float v)
{
  return std::isfinite(v);
}

static bool isFinite(const vec3 &v)
{
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// A unit quad in the XZ plane at the origin, emitting upward (+Y) on its
// front side. Per the ANARI spec the front is edge2 x edge1 =
// (0,0,1) x (1,0,0) = (0,1,0).
static RectLightGPUData unitRect(bool front, bool back, float intensity = 1.0f)
{
  RectLightGPUData r{};
  r.position = vec3(0.0f);
  r.edge1 = vec3(1.0f, 0.0f, 0.0f);
  r.edge2 = vec3(0.0f, 0.0f, 1.0f);
  r.intensity = intensity;
  r.side.front = front ? 1 : 0;
  r.side.back = back ? 1 : 0;
  r.oneOverArea = 1.0f;
  return r;
}

static RingLightGPUData unitRing(
    float innerRadius = 0.0f, float outerRadius = 1.0f, float intensity = 1.0f)
{
  RingLightGPUData r{};
  r.position = vec3(0.0f);
  r.direction = vec3(0.0f, -1.0f, 0.0f);
  r.cosOuterAngle = 0.0f; // 90 degrees: full hemisphere
  r.cosInnerAngle = 1.0f;
  r.radius = outerRadius;
  r.innerRadius = innerRadius;
  r.intensity = intensity;
  const float area =
      kPi * (outerRadius * outerRadius - innerRadius * innerRadius);
  r.oneOverArea = area > 0.0f ? 1.0f / area : 1.0f;
  return r;
}

int main()
{
  const mat4 identity(1.0f);

  // --- rectFrame: normal orientation and object-space area ------------------
  {
    const RectFrame f = rectFrame(unitRect(true, false), identity);
    CHECK(nearf(f.area, 1.0f));
    // edge2 x edge1 points along +Y for this winding.
    CHECK(nearf(f.worldNormal.y, 1.0f));
    CHECK(nearf(length(f.worldNormal), 1.0f));

    // Area is the cross-product magnitude, so a 2x3 rect has area 6.
    RectLightGPUData big = unitRect(true, false);
    big.edge1 = vec3(2.0f, 0.0f, 0.0f);
    big.edge2 = vec3(0.0f, 0.0f, 3.0f);
    CHECK(nearf(rectFrame(big, identity).area, 6.0f));

    // A non-perpendicular (sheared) parallelogram: area is |e1||e2|sin(theta),
    // NOT |e1||e2| — a dot-product-based area would pass the axis-aligned cases
    // above and fail here.
    RectLightGPUData sheared = unitRect(true, false);
    sheared.edge1 = vec3(1.0f, 0.0f, 0.0f);
    sheared.edge2 = vec3(1.0f, 0.0f, 1.0f);
    CHECK(nearf(rectFrame(sheared, identity).area, 1.0f));
  }

  // Object-space area is intentionally NOT scaled by the instance transform.
  // This mirrors the sampler's pre-existing behavior (see lightPickPower.h);
  // the hit side must reproduce it, not "fix" one side and desync the pair.
  {
    const mat4 scaled = glm::scale(mat4(1.0f), vec3(5.0f, 1.0f, 5.0f));
    const RectFrame f = rectFrame(unitRect(true, false), scaled);
    CHECK(nearf(f.area, 1.0f));
    CHECK(nearf(length(f.worldNormal), 1.0f));
  }

  // The world normal follows the instance transform.
  {
    const mat4 rot =
        glm::rotate(mat4(1.0f), glm::radians(90.0f), vec3(1.0f, 0.0f, 0.0f));
    const RectFrame f = rectFrame(unitRect(true, false), rot);
    // +Y rotated +90 deg about X maps to +Z.
    CHECK(nearf(f.worldNormal.z, 1.0f, 1e-4f));
  }

  // The world normal is the normal of the TRANSFORMED rectangle, for EVERY
  // affine transform including mirroring ones.
  //
  // Regression: deriving it as xfmVec(M, cross(e2,e1)) instead of
  // cross(M*e2, M*e1) differs by det(M). Under a mirroring transform (negative
  // determinant) the two point into opposite hemispheres, so the shared side
  // predicate calls the front face the back one and a single-sided light
  // illuminates the wrong half-space. Shadow rays never test light proxies, so
  // nothing downstream catches it -- this assertion is the guard.
  {
    // Mirror across X: det < 0.
    const mat4 mirror = glm::scale(mat4(1.0f), vec3(-1.0f, 1.0f, 1.0f));
    const RectLightGPUData r = unitRect(true, false);
    const RectFrame f = rectFrame(r, mirror);
    const vec3 fromEdges =
        normalize(cross(xfmVec(mirror, r.edge2), xfmVec(mirror, r.edge1)));
    CHECK(nearf(dot(f.worldNormal, fromEdges), 1.0f, 1e-5f));
    // And it is genuinely the opposite of the naive forward transform, so this
    // test would fail against the old formulation rather than pass vacuously.
    const vec3 naive = normalize(xfmVec(mirror, cross(r.edge2, r.edge1)));
    CHECK(dot(f.worldNormal, naive) < -0.9f);
  }

  // Same property over randomized affine transforms, mirroring included.
  {
    std::mt19937 g(20260009);
    std::uniform_real_distribution<float> uni(-2.0f, 2.0f);
    int checked = 0, negativeDet = 0;
    for (int i = 0; i < 20000; ++i) {
      RectLightGPUData r = unitRect(true, false);
      r.edge1 = vec3(uni(g), uni(g), uni(g));
      r.edge2 = vec3(uni(g), uni(g), uni(g));
      if (length(cross(r.edge1, r.edge2)) < 1e-2f)
        continue;

      mat4 m(1.0f);
      for (int c = 0; c < 3; ++c)
        for (int rw = 0; rw < 3; ++rw)
          m[c][rw] = uni(g);
      const float det = glm::determinant(mat3(m));
      if (std::fabs(det) < 1e-2f)
        continue;
      if (det < 0.0f)
        ++negativeDet;

      const vec3 we1 = xfmVec(m, r.edge1);
      const vec3 we2 = xfmVec(m, r.edge2);
      if (length(cross(we1, we2)) < 1e-4f)
        continue;

      const RectFrame f = rectFrame(r, m);
      CHECK(dot(f.worldNormal, normalize(cross(we2, we1))) > 0.999f);
      ++checked;
    }
    // The mirroring case must actually be exercised, or this proves nothing.
    CHECK(checked > 5000);
    CHECK(negativeDet > 1000);
  }

  // The ring's world axis must be UNIT length under any transform: cosTheta is
  // a dot against a unit direction, so a non-unit axis silently rescales the
  // cone thresholds. Regression against normalizing only the object-space
  // direction and then scaling it by the transform.
  {
    const mat4 scaled = glm::scale(mat4(1.0f), vec3(7.0f, 0.2f, 3.0f));
    RingLightGPUData r = unitRing();
    r.direction = vec3(0.0f, -3.0f, 0.0f); // deliberately non-unit
    CHECK(nearf(length(ringWorldAxis(r, scaled)), 1.0f, 1e-5f));
    CHECK(nearf(length(ringWorldAxis(r, mat4(1.0f))), 1.0f, 1e-5f));

    const mat4 rot =
        glm::rotate(mat4(1.0f), glm::radians(37.0f), vec3(0.3f, 1.0f, 0.2f));
    CHECK(nearf(length(ringWorldAxis(r, rot)), 1.0f, 1e-5f));
  }

  // --- rectEmissionCosTheta: the full side table ----------------------------
  // The predicate takes the normal explicitly; use -Y here so a point BELOW the
  // light sees the front face.
  // dirToLight points from the shaded point toward the light: +Y from below, -Y
  // from above.
  {
    const vec3 n(0.0f, -1.0f, 0.0f);
    const vec3 fromBelow(0.0f, 1.0f, 0.0f);
    const vec3 fromAbove(0.0f, -1.0f, 0.0f);

    // front only: lit from below, dark from above
    CHECK(rectEmissionCosTheta(unitRect(true, false), n, fromBelow) > 0.0f);
    CHECK(rectEmissionCosTheta(unitRect(true, false), n, fromAbove) <= 0.0f);

    // back only: mirrored
    CHECK(rectEmissionCosTheta(unitRect(false, true), n, fromBelow) <= 0.0f);
    CHECK(rectEmissionCosTheta(unitRect(false, true), n, fromAbove) > 0.0f);

    // both: lit from either side, and the magnitude matches
    CHECK(rectEmissionCosTheta(unitRect(true, true), n, fromBelow) > 0.0f);
    CHECK(rectEmissionCosTheta(unitRect(true, true), n, fromAbove) > 0.0f);
    CHECK(nearf(rectEmissionCosTheta(unitRect(true, true), n, fromBelow),
        rectEmissionCosTheta(unitRect(true, true), n, fromAbove)));

    // Magnitude is the true cosine: a 60-degree direction gives 0.5.
    const vec3 slanted = normalize(vec3(std::sqrt(3.0f) / 2.0f, 0.5f, 0.0f));
    CHECK(nearf(
        rectEmissionCosTheta(unitRect(true, false), n, slanted), 0.5f, 1e-4f));

    // Edge-on is exactly zero -> not emitting, and never negative-zero trouble.
    const vec3 edgeOn(1.0f, 0.0f, 0.0f);
    CHECK(!(rectEmissionCosTheta(unitRect(true, false), n, edgeOn) > 0.0f));
    CHECK(!(rectEmissionCosTheta(unitRect(true, true), n, edgeOn) > 0.0f));

    // front=0/back=0 is UNREACHABLE through the ANARI API: Rect's host-side
    // enum only has FRONT/BACK/BOTH and an unrecognized `side` string warns and
    // falls back to FRONT. The predicate consequently treats it as front-only
    // rather than as "emits nothing". Asserted here to pin the actual behavior
    // (this extraction must not change it), not to endorse it as a design.
    CHECK(rectEmissionCosTheta(unitRect(false, false), n, fromBelow) > 0.0f);
    CHECK(!(rectEmissionCosTheta(unitRect(false, false), n, fromAbove) > 0.0f));
  }

  // --- rectRadiance: Lambertian ---------------------------------------------
  {
    const RectLightGPUData r = unitRect(true, false, 3.0f);
    const vec3 color(0.25f, 0.5f, 1.0f);
    const vec3 rad = rectRadiance(r, color);
    CHECK(nearf(rad.x, 0.75f));
    CHECK(nearf(rad.y, 1.5f));
    CHECK(nearf(rad.z, 3.0f));

    // Radiance takes no distance and no direction argument at all, so it is
    // structurally independent of both. Proportionality in intensity is the
    // remaining claim.
    const vec3 twice = rectRadiance(unitRect(true, false, 6.0f), color);
    CHECK(nearf(twice.x, 2.0f * rad.x));

    // Zero intensity emits nothing.
    CHECK(nearf(rectRadiance(unitRect(true, false, 0.0f), color).x, 0.0f));
  }

  // --- rectSolidAnglePdf ----------------------------------------------------
  {
    // Head-on at unit distance from a unit-area rect: pdf == 1.
    CHECK(nearf(rectSolidAnglePdf(1.0f, 1.0f, 1.0f), 1.0f));
    // Inverse-square in distance.
    CHECK(nearf(rectSolidAnglePdf(1.0f, 2.0f, 1.0f), 4.0f));
    CHECK(nearf(rectSolidAnglePdf(1.0f, 3.0f, 1.0f), 9.0f));
    // Inversely proportional to area.
    CHECK(nearf(rectSolidAnglePdf(4.0f, 1.0f, 1.0f), 0.25f));
    // Grazing angles raise the density.
    CHECK(nearf(rectSolidAnglePdf(1.0f, 1.0f, 0.5f), 2.0f));
    CHECK(rectSolidAnglePdf(1.0f, 1.0f, 0.1f)
        > rectSolidAnglePdf(1.0f, 1.0f, 0.9f));
    // Always positive and finite for valid inputs.
    CHECK(isFinite(rectSolidAnglePdf(1e-3f, 1e3f, 1e-3f)));
  }

  // --- ringSpotAttenuation: cone falloff ------------------------------------
  {
    RingLightGPUData r = unitRing();
    r.cosOuterAngle = 0.5f; // 60 degrees
    r.cosInnerAngle = 0.8f; // ~37 degrees

    // Inside the inner cone: full.
    CHECK(nearf(ringSpotAttenuation(r, 1.0f), 1.0f));
    CHECK(nearf(ringSpotAttenuation(r, 0.9f), 1.0f));
    // Outside the outer cone: zero.
    CHECK(nearf(ringSpotAttenuation(r, 0.4f), 0.0f));
    CHECK(nearf(ringSpotAttenuation(r, -1.0f), 0.0f));
    // Continuous at both boundaries — a discontinuity here shows as a hard
    // ring.
    CHECK(nearf(ringSpotAttenuation(r, 0.5f), 0.0f, 1e-4f));
    CHECK(nearf(ringSpotAttenuation(r, 0.8f), 1.0f, 1e-4f));
    // Monotonically increasing across the falloff band.
    float prev = -1.0f;
    for (int i = 0; i <= 20; ++i) {
      const float c = 0.5f + (0.8f - 0.5f) * (float(i) / 20.0f);
      const float s = ringSpotAttenuation(r, c);
      CHECK(s >= prev - 1e-6f);
      CHECK(s >= 0.0f && s <= 1.0f);
      prev = s;
    }
    // Midpoint of a smoothstep is exactly 0.5.
    CHECK(nearf(ringSpotAttenuation(r, 0.65f), 0.5f, 1e-4f));
  }

  // --- ringRadiance ---------------------------------------------------------
  {
    const RingLightGPUData r = unitRing(0.0f, 1.0f, 2.0f);
    const vec3 color(1.0f, 0.5f, 0.25f);
    const vec3 full = ringRadiance(r, color, 1.0f);
    CHECK(nearf(full.x, 2.0f));
    CHECK(nearf(full.y, 1.0f));
    // The cone falloff scales radiance linearly.
    const vec3 half = ringRadiance(r, color, 0.5f);
    CHECK(nearf(half.x, 1.0f));
    CHECK(nearf(ringRadiance(r, color, 0.0f).x, 0.0f));
  }

  // --- ringSolidAnglePdf ----------------------------------------------------
  {
    // Unit disk: area pi, so head-on at unit distance pdf == 1/pi.
    const RingLightGPUData r = unitRing(0.0f, 1.0f);
    CHECK(nearf(ringSolidAnglePdf(r, 1.0f, 1.0f), 1.0f / kPi));
    CHECK(nearf(ringSolidAnglePdf(r, 2.0f, 1.0f), 4.0f / kPi));
    CHECK(nearf(ringSolidAnglePdf(r, 1.0f, 0.5f), 2.0f / kPi));

    // An annulus has strictly less area than the full disk, hence a higher pdf.
    const RingLightGPUData annulus = unitRing(0.5f, 1.0f);
    CHECK(ringSolidAnglePdf(annulus, 1.0f, 1.0f)
        > ringSolidAnglePdf(r, 1.0f, 1.0f));
    // pi*(1 - 0.25) = 0.75pi
    CHECK(nearf(ringSolidAnglePdf(annulus, 1.0f, 1.0f), 1.0f / (0.75f * kPi)));
  }

  // --- Degenerate configurations: no NaN, no infinity -----------------------
  {
    // Zero-area rect: the frame reports zero area, and the caller must gate on
    // it. What must NOT happen is a NaN normal silently poisoning downstream
    // math.
    RectLightGPUData degenerate = unitRect(true, false);
    degenerate.edge2 = degenerate.edge1; // parallel edges -> zero cross product
    const RectFrame f = rectFrame(degenerate, identity);
    CHECK(nearf(f.area, 0.0f));
    // The normal is zero, not NaN: rectFrame must not normalize a zero cross
    // product. Downstream this makes cosTheta zero for every direction, so the
    // `cosTheta > 0` gates reject the light instead of depending on NaN
    // comparisons that fast math is free to reorder.
    CHECK(isFinite(f.worldNormal));
    CHECK(nearf(length(f.worldNormal), 0.0f));
    CHECK(nearf(rectEmissionCosTheta(
                    degenerate, f.worldNormal, vec3(0.0f, -1.0f, 0.0f)),
        0.0f));

    // Same for a rect the INSTANCE TRANSFORM collapses: the object-space area
    // is nonzero, so only the transformed cross product degenerates.
    const mat4 flatten = glm::scale(mat4(1.0f), vec3(1.0f, 1.0f, 0.0f));
    const RectFrame flattened = rectFrame(unitRect(true, false), flatten);
    CHECK(isFinite(flattened.worldNormal));
    CHECK(nearf(length(flattened.worldNormal), 0.0f));

    RectLightGPUData zeroEdge = unitRect(true, false);
    zeroEdge.edge1 = vec3(0.0f);
    CHECK(nearf(rectFrame(zeroEdge, identity).area, 0.0f));
    CHECK(isFinite(rectFrame(zeroEdge, identity).worldNormal));

    // A `both`-sided degenerate rect must not escape through the fabsf() in
    // the side predicate either: zero stays zero under an absolute value,
    // whereas a NaN would survive it.
    RectLightGPUData degenerateBoth = degenerate;
    degenerateBoth.side.front = 1;
    degenerateBoth.side.back = 1;
    CHECK(nearf(rectEmissionCosTheta(degenerateBoth,
                    rectFrame(degenerateBoth, identity).worldNormal,
                    vec3(0.0f, -1.0f, 0.0f)),
        0.0f));

    // Radiance stays finite regardless of geometry degeneracy.
    CHECK(isFinite(rectRadiance(degenerate, vec3(1.0f)).x));

    // Zero-radius ring and inner==outer ring: the host-side oneOverArea guard
    // keeps the pdf finite rather than producing an infinity.
    const RingLightGPUData zeroRadius = unitRing(0.0f, 0.0f);
    CHECK(isFinite(ringSolidAnglePdf(zeroRadius, 1.0f, 1.0f)));
    const RingLightGPUData emptyAnnulus = unitRing(1.0f, 1.0f);
    CHECK(isFinite(ringSolidAnglePdf(emptyAnnulus, 1.0f, 1.0f)));
    CHECK(isFinite(ringRadiance(zeroRadius, vec3(1.0f), 1.0f).x));
  }

  if (g_failures == 0)
    std::printf("test_LightGeometry: all checks passed\n");
  else
    std::printf("test_LightGeometry: %d failure(s)\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
