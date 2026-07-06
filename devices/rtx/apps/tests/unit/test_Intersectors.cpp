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

// Host unit tests for the pure analytic primitive solvers. No CUDA/OptiX/GPU —
// exercises the geometry math directly and returns nonzero on any failure
// (assert() is compiled out in Release, the repo default).

#include "gpu/intersectPrimitives.h"

#include <cmath>
#include <cstdio>
#include <string>

using namespace visrtx;
using glm::vec3;

static int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);              \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

static bool nearf(float a, float b, float eps = 1e-3f)
{
  return std::fabs(a - b) <= eps;
}

static bool isUnit(const vec3 &v)
{
  return nearf(glm::length(v), 1.f, 1e-3f);
}

// Return the crossing with smallest t among those with t >= tmin (forward hit),
// or nullptr. Mirrors what OptiX closest-hit would keep.
static const PrimHit *nearestForward(const PrimHit *h, int n, float tmin = 0.f)
{
  const PrimHit *best = nullptr;
  for (int i = 0; i < n; ++i)
    if (h[i].t >= tmin && (!best || h[i].t < best->t))
      best = &h[i];
  return best;
}

static void testSphere()
{
  const vec3 c(0.f);
  PrimHit h[2];

  // Direct hit from outside: entry + exit.
  int n = solveSphere(vec3(0, 0, -5), vec3(0, 0, 1), c, 1.f, h);
  CHECK(n == 2);
  const PrimHit *near = nearestForward(h, n);
  CHECK(near && nearf(near->t, 4.f));
  CHECK(near && isUnit(near->Ng) && glm::dot(near->Ng, vec3(0, 0, 1)) < 0.f);

  // Miss.
  CHECK(solveSphere(vec3(0, 3, -5), vec3(0, 0, 1), c, 1.f, h) == 0);

  // Tangent -> single unique crossing (coincident roots de-duped).
  n = solveSphere(vec3(0, 1, -5), vec3(0, 0, 1), c, 1.f, h);
  CHECK(n == 1);
  CHECK(n == 1 && nearf(h[0].t, 5.f) && isUnit(h[0].Ng));

  // Origin inside: 2 algebraic roots, 1 forward-accepted.
  n = solveSphere(vec3(0, 0, 0), vec3(0, 0, 1), c, 1.f, h);
  CHECK(n == 2);
  int forward = 0;
  for (int i = 0; i < n; ++i)
    if (h[i].t >= 0.f)
      ++forward;
  CHECK(forward == 1);

  // Non-unit direction (instance-scale analogue): t rescales, hitpoint
  // invariant.
  n = solveSphere(vec3(0, 0, -5), vec3(0, 0, 2), c, 1.f, h);
  const PrimHit *nf = nearestForward(h, n);
  CHECK(nf && nearf(nf->t, 2.f)); // 2*(0,0,2) from -5 => z=-1 (entry)
}

static void testCylinder()
{
  const vec3 p0(0, 0, 0), p1(0, 0, 2); // axis +z, length 2
  const float r = 1.f;
  PrimHit h[4];

  // Side hit: entry + exit on the curved wall.
  int n = solveCylinder(vec3(-5, 0, 1), vec3(1, 0, 0), p0, p1, r, 0, h);
  CHECK(n == 2);
  const PrimHit *near = nearestForward(h, n);
  CHECK(near && nearf(near->t, 4.f) && nearf(near->u, 0.5f));
  CHECK(near && isUnit(near->Ng) && nearf(near->Ng.x, -1.f));

  // Axis-parallel into caps=both: THE regression. Was 0 hits; must be 2.
  n = solveCylinder(
      vec3(0, 0, -5), vec3(0, 0, 1), p0, p1, r, CAP_FIRST | CAP_SECOND, h);
  CHECK(n == 2);
  const PrimHit *cnear = nearestForward(h, n);
  CHECK(cnear && nearf(cnear->t, 5.f) && nearf(cnear->u, 0.f));
  CHECK(cnear && isUnit(cnear->Ng) && nearf(cnear->Ng.z, -1.f));

  // caps=none end-on -> nothing.
  CHECK(solveCylinder(vec3(0, 0, -5), vec3(0, 0, 1), p0, p1, r, 0, h) == 0);

  // caps=first only -> just the p0 cap.
  n = solveCylinder(vec3(0, 0, -5), vec3(0, 0, 1), p0, p1, r, CAP_FIRST, h);
  CHECK(n == 1 && nearf(h[0].u, 0.f));

  // caps=second only -> ray enters open p0 end, hits p1 cap (u=1).
  n = solveCylinder(vec3(0, 0, -5), vec3(0, 0, 1), p0, p1, r, CAP_SECOND, h);
  CHECK(n == 1 && nearf(h[0].u, 1.f));

  // Perpendicular ray (sd==0): caps can't be hit; wall still gives 2.
  n = solveCylinder(
      vec3(-5, 0, 1), vec3(1, 0, 0), p0, p1, r, CAP_FIRST | CAP_SECOND, h);
  CHECK(n == 2);

  // Cap-edge clip: end-on but outside radius -> miss even with caps.
  CHECK(solveCylinder(
            vec3(3, 0, -5), vec3(0, 0, 1), p0, p1, r, CAP_FIRST | CAP_SECOND, h)
      == 0);

  // Degenerate p0==p1 -> no hit, no crash/NaN.
  CHECK(solveCylinder(
            vec3(-5, 0, 0), vec3(1, 0, 0), p0, p0, r, CAP_FIRST | CAP_SECOND, h)
      == 0);

  // Every reported normal is unit.
  n = solveCylinder(
      vec3(0, 0, -5), vec3(0, 0, 1), p0, p1, r, CAP_FIRST | CAP_SECOND, h);
  for (int i = 0; i < n; ++i)
    CHECK(isUnit(h[i].Ng));
}

static void testCone()
{
  PrimHit h[4];

  // Truncated cone: p0 r=1 at z=0, apex r=0 at z=4. Side ray at z=1 where the
  // cone radius is 0.75 -> entry x=-0.75 (t=4.25), exit x=0.75 (t=5.75).
  const vec3 p0(0, 0, 0), apex(0, 0, 4);
  int n = solveCone(vec3(-5, 0, 1), vec3(1, 0, 0), p0, apex, 1.f, 0.f, 0, h);
  CHECK(n == 2);
  const PrimHit *near = nearestForward(h, n);
  CHECK(near && nearf(near->t, 4.25f) && nearf(near->u, 0.25f));
  CHECK(near && isUnit(near->Ng) && glm::dot(near->Ng, vec3(1, 0, 0)) < 0.f);

  // Equal-radius cone (== cylinder) hit axis-parallel: total-degeneracy body
  // MISS. caps=none -> 0 crossings; caps=both -> 2 cap crossings.
  const vec3 q0(0, 0, 0), q1(0, 0, 2);
  CHECK(solveCone(vec3(0, 0, -5), vec3(0, 0, 1), q0, q1, 1.f, 1.f, 0, h) == 0);
  n = solveCone(vec3(0, 0, -5),
      vec3(0, 0, 1),
      q0,
      q1,
      1.f,
      1.f,
      CAP_FIRST | CAP_SECOND,
      h);
  CHECK(n == 2);

  // caps=none on the truncated cone with a cap-crossing axis ray still shows
  // no cap; caps=both adds the base disk.
  n = solveCone(vec3(0, 0, -5), vec3(0, 0, 1), p0, apex, 1.f, 0.f, 0, h);
  const int bodyOnly = n;
  n = solveCone(
      vec3(0, 0, -5), vec3(0, 0, 1), p0, apex, 1.f, 0.f, CAP_FIRST, h);
  CHECK(n == bodyOnly + 1); // base cap at p0 added

  // Zero-radius apex end requests a cap -> degenerate disk, no cap hit.
  n = solveCone(
      vec3(0, 0, 9), vec3(0, 0, -1), p0, apex, 1.f, 0.f, CAP_SECOND, h);
  for (int i = 0; i < n; ++i)
    CHECK(!nearf(h[i].u, 1.f)); // no p1 (apex) cap emitted

  // Non-unit direction: hitpoint invariant.
  n = solveCone(vec3(-5, 0, 1), vec3(2, 0, 0), p0, apex, 1.f, 0.f, 0, h);
  const PrimHit *nf = nearestForward(h, n);
  CHECK(nf && nearf((vec3(-5, 0, 1) + nf->t * vec3(2, 0, 0)).x, -0.75f));

  // Normals unit.
  n = solveCone(vec3(-5, 0, 1), vec3(1, 0, 0), p0, apex, 1.f, 0.f, 0, h);
  for (int i = 0; i < n; ++i)
    CHECK(isUnit(h[i].Ng));
}

// Regression for phantom self-shadowing exits at grazing incidence (acne rings
// on large ground spheres): shadow rays start ~1e-6*|P| above their origin
// surface (epsilonFrom's lift) and run nearly tangent to it toward the light.
// Near tangency the quadratic discriminant is fp noise, and before the
// kGrazeRelEps gate a false-positive root pair emitted a back-facing exit at
// t up to ~1e-3*r — far past tmin, shadowing the surface in rings. Scans many
// surface points and above-horizon directions on RTOW-scale primitives and
// requires that no forward crossing survives.
static void testGrazeSelfShadow()
{
  constexpr float r = 1000.f; // RTOW ground-sphere scale
  // epsilonFrom's lift for |P| ~ r: max(|P|,t*|dir|) * 0x1.fp-21
  const float lift = r * 0x1.fp-21f;
  const float tmin = lift; // renderers also set tmin = hit.epsilon

  int phantoms = 0;
  const auto countForward = [&](const PrimHit *h, int n) {
    for (int i = 0; i < n; ++i)
      if (h[i].t >= tmin)
        ++phantoms;
  };

  // Sphere: surface points around the top of a giant ground sphere, shadow
  // directions above the local horizon (any hit is geometrically impossible).
  {
    const vec3 c(0.f, -r, 0.f);
    PrimHit h[2];
    for (int ia = 0; ia < 100; ++ia) {
      const float alpha = 0.0007f * float(ia); // polar angle from +y
      const vec3 N(std::sin(alpha), std::cos(alpha), 0.f);
      const vec3 T1(std::cos(alpha), -std::sin(alpha), 0.f);
      const vec3 T2(0.f, 0.f, 1.f);
      const vec3 org = c + (r + lift) * N;
      for (int it = 0; it < 20; ++it) {
        const float theta = 0.0005f + 0.0025f * float(it); // elevation
        for (int ip = 0; ip < 8; ++ip) {
          const float phi = 0.7853982f * float(ip);
          const vec3 dir =
              std::cos(theta) * (std::cos(phi) * T1 + std::sin(phi) * T2)
              + std::sin(theta) * N;
          countForward(h, solveSphere(org, dir, c, r, h));
        }
      }
    }
  }

  // Cylinder wall: same scan on a giant cylinder, tangent-plane directions.
  {
    const vec3 p0(0.f, -r, -r), p1(0.f, -r, r);
    PrimHit h[4];
    for (int ia = 0; ia < 100; ++ia) {
      const float alpha = 0.0007f * float(ia);
      const vec3 N(std::sin(alpha), std::cos(alpha), 0.f);
      const vec3 T1(std::cos(alpha), -std::sin(alpha), 0.f);
      const vec3 T2(0.f, 0.f, 1.f);
      const vec3 org = vec3(0.f, -r, 0.f) + (r + lift) * N;
      for (int it = 0; it < 20; ++it) {
        const float theta = 0.0005f + 0.0025f * float(it);
        for (int ip = 0; ip < 8; ++ip) {
          const float phi = 0.7853982f * float(ip);
          const vec3 dir =
              std::cos(theta) * (std::cos(phi) * T1 + std::sin(phi) * T2)
              + std::sin(theta) * N;
          countForward(h, solveCylinder(org, dir, p0, p1, r, 0, h));
        }
      }
    }
  }

  // Cone slant: giant near-cylindrical truncated cone, directions above the
  // slant surface's tangent plane.
  {
    const vec3 p0(0.f, -r, -r), p1(0.f, -r, r);
    const float r0 = r, r1 = 0.9f * r;
    // Slant makes angle beta with the axis plane; outward normal tilts by beta.
    const float beta = std::atan((r0 - r1) / (2.f * r));
    PrimHit h[4];
    for (int ia = 0; ia < 100; ++ia) {
      const float alpha = 0.0007f * float(ia);
      const vec3 radial(std::sin(alpha), std::cos(alpha), 0.f);
      const vec3 T1(std::cos(alpha), -std::sin(alpha), 0.f);
      const vec3 axis(0.f, 0.f, 1.f);
      const vec3 N = std::cos(beta) * radial + std::sin(beta) * axis;
      const vec3 T2 = std::cos(beta) * axis - std::sin(beta) * radial;
      const float rz = 0.5f * (r0 + r1); // radius at the axis midpoint (z=0)
      const vec3 org = vec3(0.f, -r, 0.f) + rz * radial + lift * N;
      for (int it = 0; it < 20; ++it) {
        const float theta = 0.0005f + 0.0025f * float(it);
        for (int ip = 0; ip < 8; ++ip) {
          const float phi = 0.7853982f * float(ip);
          const vec3 dir =
              std::cos(theta) * (std::cos(phi) * T1 + std::sin(phi) * T2)
              + std::sin(theta) * N;
          countForward(h, solveCone(org, dir, p0, p1, r0, r1, 0, h));
        }
      }
    }
  }

  if (phantoms) {
    std::printf("FAIL %s:%d  %d phantom grazing self-shadow crossing(s)\n",
        __FILE__,
        __LINE__,
        phantoms);
    ++g_failures;
  }
}

int main()
{
  testSphere();
  testCylinder();
  testCone();
  testGrazeSelfShadow();
  if (g_failures) {
    std::printf("%d CHECK(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all intersector host tests passed\n");
  return 0;
}
