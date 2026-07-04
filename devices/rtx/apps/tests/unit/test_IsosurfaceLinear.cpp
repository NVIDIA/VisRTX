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

// Host unit tests for the pure analytic isosurface core math. No CUDA/OptiX/GPU
// — exercises the cubic solver + trilinear crossing directly and returns
// nonzero on any failure (assert() is compiled out in Release, the repo
// default).

#include "geometry/IsosurfaceLinear.h"

#include <cmath> // std::sqrt / std::cbrt / std::fabs used directly below

#include <cstdint>
#include <cstdio>

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

// Test-local mock field state types. `SpatialFieldGPUData` is only forward-
// declared in the header; this TU links nothing else that defines it (only
// glm), so completing it here is legal. sampleValue/sampleNormal are resolved
// by ADL on the state type (which lives in `visrtx`) when linearCrossing
// instantiates.
namespace visrtx {

struct SpatialFieldGPUData
{
}; // test-local completion of the fwd decl

// Affine field f(p) = dot(grad, p) + offset; constant gradient.
struct LinearFieldState
{
  vec3 grad;
  float offset;
};
inline float sampleValue(
    const LinearFieldState &st, const SpatialFieldGPUData &, const vec3 &p)
{
  return glm::dot(st.grad, p) + st.offset;
}
inline vec3 sampleNormal(
    const LinearFieldState &st, const SpatialFieldGPUData &, const vec3 &)
{
  return st.grad;
}

// Quadratic along x: f = (p.x - x0)^2 (for the hybrid-bracket miss case).
struct QuadFieldState
{
  float x0;
};
inline float sampleValue(
    const QuadFieldState &st, const SpatialFieldGPUData &, const vec3 &p)
{
  const float d = p.x - st.x0;
  return d * d;
}
inline vec3 sampleNormal(
    const QuadFieldState &st, const SpatialFieldGPUData &, const vec3 &p)
{
  return vec3(2.f * (p.x - st.x0), 0.f, 0.f);
}

} // namespace visrtx

using namespace visrtx;

// g(s) = ((k3 s + k2) s + k1) s + k0 — reference monomial evaluation.
static float evalCubic(float k0, float k1, float k2, float k3, float s)
{
  return ((k3 * s + k2) * s + k1) * s + k0;
}

// Sample a cubic at the four fixed nodes {0,1/3,2/3,1}.
static void nodesOf(float k0, float k1, float k2, float k3, float n[4])
{
  n[0] = evalCubic(k0, k1, k2, k3, 0.f);
  n[1] = evalCubic(k0, k1, k2, k3, 1.f / 3.f);
  n[2] = evalCubic(k0, k1, k2, k3, 2.f / 3.f);
  n[3] = evalCubic(k0, k1, k2, k3, 1.f);
}

static void testTrilinear()
{
  float c[8];
  for (int m = 0; m < 8; ++m)
    c[m] = float(m);

  // Each corner c[i + 2j + 4k] reproduced at u = (i,j,k) in {0,1}^3.
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i) {
        const vec3 u = vec3(float(i), float(j), float(k));
        CHECK(nearf(trilinearValue(c, u), c[i + 2 * j + 4 * k]));
      }

  // Center equals the mean of all 8 corners.
  float mean = 0.f;
  for (int m = 0; m < 8; ++m)
    mean += c[m];
  mean /= 8.f;
  CHECK(nearf(trilinearValue(c, vec3(0.5f)), mean));

  // Edge midpoints equal the mean of the two edge corners (per-axis linearity).
  CHECK(nearf(trilinearValue(c, vec3(0.5f, 0.f, 0.f)), 0.5f * (c[0] + c[1])));
  CHECK(nearf(trilinearValue(c, vec3(0.f, 0.5f, 0.f)), 0.5f * (c[0] + c[2])));
  CHECK(nearf(trilinearValue(c, vec3(0.f, 0.f, 0.5f)), 0.5f * (c[0] + c[4])));
}

static void testPrepareCubic()
{
  // 1. Exact fit: recover arbitrary monomial coefficients from node values.
  {
    const float k0 = 0.5f, k1 = -2.f, k2 = 3.f, k3 = 1.25f;
    float n[4];
    nodesOf(k0, k1, k2, k3, n);
    const CubicPrep p = prepareCubic(n[0], n[1], n[2], n[3]);
    CHECK(nearf(p.k0, k0, 1e-4f));
    CHECK(nearf(p.k1, k1, 1e-4f));
    CHECK(nearf(p.k2, k2, 1e-4f));
    CHECK(nearf(p.k3, k3, 1e-4f));
  }

  // 2. Linear g = s - 0.5 -> k = (-0.5, 1, 0, 0), nb == 2, bounds {0,1}.
  {
    const CubicPrep p = prepareCubic(-0.5f, -1.f / 6.f, 1.f / 6.f, 0.5f);
    CHECK(nearf(p.k0, -0.5f));
    CHECK(nearf(p.k1, 1.f));
    CHECK(nearf(p.k2, 0.f));
    CHECK(nearf(p.k3, 0.f));
    CHECK(p.nb == 2);
    CHECK(nearf(p.bound[0], 0.f));
    CHECK(nearf(p.bound[1], 1.f));
  }

  // 3. Interior extrema: g = (s-0.2)(s-0.5)(s-0.8). nb == 4 and the two
  // interior bounds equal the roots of g'(s) = 3s^2 - 3s + 0.66.
  {
    // g = s^3 - 1.5 s^2 + 0.66 s - 0.08
    const float k0 = -0.08f, k1 = 0.66f, k2 = -1.5f, k3 = 1.f;
    float n[4];
    nodesOf(k0, k1, k2, k3, n);
    const CubicPrep p = prepareCubic(n[0], n[1], n[2], n[3]);
    const float a = 3.f, b = -3.f, c = 0.66f;
    const float sq = std::sqrt(b * b - 4.f * a * c);
    const float r0 = (-b - sq) / (2.f * a);
    const float r1 = (-b + sq) / (2.f * a);
    CHECK(p.nb == 4);
    CHECK(nearf(p.bound[0], 0.f));
    CHECK(nearf(p.bound[1], r0));
    CHECK(nearf(p.bound[2], r1));
    CHECK(nearf(p.bound[3], 1.f));
  }

  // 4. Quadratic branch (k3 == 0): g = (s-0.5)^2 -> nb == 3, bound[1] == 0.5.
  {
    const CubicPrep p = prepareCubic(0.25f, 1.f / 36.f, 1.f / 36.f, 0.25f);
    CHECK(nearf(p.k3, 0.f));
    CHECK(p.nb == 3);
    CHECK(nearf(p.bound[1], 0.5f));
  }

  // 5. Extremum within kExtremumMargin of an endpoint is dropped: g = s^2 has
  // its g' root at s = 0 -> nb stays 2.
  {
    const CubicPrep p = prepareCubic(0.f, 1.f / 9.f, 4.f / 9.f, 1.f);
    CHECK(p.nb == 2);
  }
}

static void testSolveCubic()
{
  // 1. Three real roots in [0,1] (g = (s-0.2)(s-0.5)(s-0.8)) -> smallest ~0.2.
  {
    float n[4];
    nodesOf(-0.08f, 0.66f, -1.5f, 1.f, n);
    const CubicPrep p = prepareCubic(n[0], n[1], n[2], n[3]);
    float s = 0.f;
    CHECK(solveCubic(p, 0.f, s));
    CHECK(nearf(s, 0.2f));
  }

  // 2. Multi-iso reuse: one prep for g = s solves several isovalues.
  {
    const CubicPrep p = prepareCubic(0.f, 1.f / 3.f, 2.f / 3.f, 1.f);
    float s = 0.f;
    CHECK(solveCubic(p, 0.25f, s) && nearf(s, 0.25f));
    CHECK(solveCubic(p, 0.5f, s) && nearf(s, 0.5f));
    CHECK(solveCubic(p, 0.9f, s) && nearf(s, 0.9f));
  }

  // 3. Boundary asymmetry + watertight carry.
  {
    // g = s - 1 (root at upper end, approached from below): found, s ~ 1.
    float s = 0.f;
    const CubicPrep pInc = prepareCubic(-1.f, -2.f / 3.f, -1.f / 3.f, 0.f);
    CHECK(solveCubic(pInc, 0.f, s));
    CHECK(nearf(s, 1.f));

    // g = 1 - s (root at upper end, approached from above): miss.
    const CubicPrep pDec = prepareCubic(1.f, 2.f / 3.f, 1.f / 3.f, 0.f);
    CHECK(!solveCubic(pDec, 0.f, s));

    // Watertight carry: next voxel's entry node value is exactly 0 -> hit s==0.
    const CubicPrep pCarry = prepareCubic(0.f, -1.f / 3.f, -2.f / 3.f, -1.f);
    CHECK(solveCubic(pCarry, 0.f, s));
    CHECK(s == 0.f);
  }

  // 4. Root exactly at s=0 from either side -> s == 0.
  {
    float s = 1.f;
    CHECK(solveCubic(prepareCubic(0.f, 1.f / 3.f, 2.f / 3.f, 1.f), 0.f, s));
    CHECK(s == 0.f);
    s = 1.f;
    CHECK(solveCubic(prepareCubic(0.f, -1.f / 3.f, -2.f / 3.f, -1.f), 0.f, s));
    CHECK(s == 0.f);
  }

  // 5. Tangent, exact vs lifted: g = (s-1/2)^2 at nodes {1/4,1/36,1/36,1/4}.
  {
    float s = 0.f;
    const CubicPrep pTan = prepareCubic(0.25f, 1.f / 36.f, 1.f / 36.f, 0.25f);
    CHECK(solveCubic(pTan, 0.f, s));
    CHECK(s == 0.5f);

    const float d = 1e-3f;
    const CubicPrep pLift =
        prepareCubic(0.25f + d, 1.f / 36.f + d, 1.f / 36.f + d, 0.25f + d);
    CHECK(!solveCubic(pLift, 0.f, s));
  }

  // 6. No crossing (g strictly positive) -> false.
  {
    float s = 0.f;
    const CubicPrep p = prepareCubic(
        0.75f, 0.5f + 1.f / 36.f, 0.5f + 1.f / 36.f, 0.75f); // (s-1/2)^2 + 1/2
    CHECK(!solveCubic(p, 0.f, s));
  }

  // 7. Scale independence: g = A (s^3 - 0.2) has root cbrt(0.2) for any A, but
  // at A~1e4 the value residual (~1e4 * step) never drops below kRootTolerance
  // (1e-6, absolute in value units), so the solver runs the full 8 Illinois
  // iterations. The root estimate must not degrade vs the small-scale case that
  // could early-exit -- regula-falsi converges in the s domain, independent of
  // A.
  {
    const float root = std::cbrt(0.2f);
    float n[4];
    nodesOf(-0.2f, 0.f, 0.f, 1.f, n);
    float sSmall = 0.f;
    CHECK(solveCubic(prepareCubic(n[0], n[1], n[2], n[3]), 0.f, sSmall));
    nodesOf(-2000.f, 0.f, 0.f, 1e4f, n);
    float sLarge = 0.f;
    CHECK(solveCubic(prepareCubic(n[0], n[1], n[2], n[3]), 0.f, sLarge));
    // 8-iter regula-falsi on this cubic lands ~2e-3 from the true root; the key
    // property is that the 1e4 scaling leaves that unchanged.
    CHECK(nearf(sLarge, root, 5e-3f));
    CHECK(nearf(sLarge, sSmall, 1e-5f));
  }

  // 8. Constant zero field (all nodes == iso) -> true, s == 0.
  {
    float s = 1.f;
    CHECK(solveCubic(prepareCubic(0.f, 0.f, 0.f, 0.f), 0.f, s));
    CHECK(s == 0.f);
  }
}

static void testFirstTrilinearRootProperty()
{
  const float iso = 0.f;
  const int kScanSteps = 2048;
  const int kBisectIters = 40;

  uint32_t x = 12345u;
  auto nextUnit = [&]() {
    x = x * 1664525u + 1013904223u;
    return float(x) / 4294967296.f; // [0,1)
  };
  auto nextSigned = [&]() { return 2.f * nextUnit() - 1.f; }; // [-1,1)

  for (int iter = 0; iter < 200; ++iter) {
    float c[8];
    for (int m = 0; m < 8; ++m)
      c[m] = nextSigned();
    const vec3 aEntry(nextUnit(), nextUnit(), nextUnit());
    const vec3 aExit(nextUnit(), nextUnit(), nextUnit());
    const vec3 d = aExit - aEntry;

    auto v = [&](float s) { return trilinearValue(c, aEntry + s * d) - iso; };

    const float v0 = v(0.f);
    const float v1 = v(1.f);
    // Contract is only defined when the endpoints straddle.
    if ((v0 < 0.f) == (v1 < 0.f))
      continue;

    // Reference: dense scan for the first sign-change bracket, then bisect.
    float lo = 0.f, hi = 1.f;
    float prevS = 0.f, prevV = v0;
    bool found = false;
    for (int i = 1; i <= kScanSteps; ++i) {
      const float s = float(i) / float(kScanSteps);
      const float vs = v(s);
      if ((prevV < 0.f) != (vs < 0.f)) {
        lo = prevS;
        hi = s;
        found = true;
        break;
      }
      prevS = s;
      prevV = vs;
    }
    CHECK(found);
    float sRef = 0.5f * (lo + hi);
    {
      float a = lo, b = hi, va = v(a);
      for (int i = 0; i < kBisectIters; ++i) {
        const float m = 0.5f * (a + b);
        const float vm = v(m);
        if ((vm < 0.f) == (va < 0.f)) {
          a = m;
          va = vm;
        } else
          b = m;
      }
      sRef = 0.5f * (a + b);
    }

    float s = 0.f;
    CHECK(firstTrilinearRoot(c, aEntry, aExit, iso, s));
    CHECK(nearf(v(s), 0.f, 1e-3f));
    // Never returns a later root than the true first crossing (one-sided:
    // it may legitimately return an earlier one the coarse scan missed).
    CHECK(s <= sRef + 1e-3f);
  }
}

static void testLinearCrossing()
{
  SpatialFieldGPUData field; // dummy; mocks ignore it
  const vec3 ro(0.f), rd(1.f, 0.f, 0.f);

  // 1. f(p) = p.x, iso 0.25: hit at t=0.25, gradient flipped to face the ray.
  {
    const LinearFieldState st{vec3(1.f, 0.f, 0.f), 0.f};
    const float iso[] = {0.25f};
    const float vE = sampleValue(st, field, ro),
                vX = sampleValue(st, field, ro + rd);
    float outT = 0.f;
    vec3 outN(0.f);
    uint32_t outIdx = 99u;
    CHECK(linearCrossing(
        st, field, ro, rd, 0.f, 1.f, vE, vX, iso, 1, outT, outN, outIdx));
    CHECK(nearf(outT, 0.25f));
    CHECK(outIdx == 0u);
    CHECK(nearf(outN.x, -1.f));
    CHECK(isUnit(outN));
    CHECK(glm::dot(outN, rd) <= 0.f);
  }

  // 2. Multi-isovalue nearest-wins.
  {
    const LinearFieldState st{vec3(1.f, 0.f, 0.f), 0.f};
    const float iso[] = {0.75f, 0.25f, 0.5f};
    const float vE = sampleValue(st, field, ro),
                vX = sampleValue(st, field, ro + rd);
    float outT = 0.f;
    vec3 outN(0.f);
    uint32_t outIdx = 99u;
    CHECK(linearCrossing(
        st, field, ro, rd, 0.f, 1.f, vE, vX, iso, 3, outT, outN, outIdx));
    CHECK(nearf(outT, 0.25f));
    CHECK(outIdx == 1u);
  }

  // 3. No straddle -> false.
  {
    const LinearFieldState st{vec3(1.f, 0.f, 0.f), 0.f};
    const float iso[] = {1.5f};
    const float vE = sampleValue(st, field, ro),
                vX = sampleValue(st, field, ro + rd);
    float outT = 0.f;
    vec3 outN(0.f);
    uint32_t outIdx = 0u;
    CHECK(!linearCrossing(
        st, field, ro, rd, 0.f, 1.f, vE, vX, iso, 1, outT, outN, outIdx));
  }

  // 4. Non-unit rd with matching tExit: hit point invariant, t rescales.
  {
    const LinearFieldState st{vec3(1.f, 0.f, 0.f), 0.f};
    const vec3 rd2(2.f, 0.f, 0.f);
    const float tExit = 0.5f;
    const float iso[] = {0.25f};
    const float vE = sampleValue(st, field, ro);
    const float vX = sampleValue(st, field, ro + tExit * rd2);
    float outT = 0.f;
    vec3 outN(0.f);
    uint32_t outIdx = 0u;
    CHECK(linearCrossing(
        st, field, ro, rd2, 0.f, tExit, vE, vX, iso, 1, outT, outN, outIdx));
    CHECK(nearf(outT, 0.125f));
    CHECK(isUnit(outN));
    CHECK(glm::dot(outN, rd2) <= 0.f);
  }

  // 5. Decreasing field f = 1 - p.x, iso 0.75: gradient already faces the ray.
  {
    const LinearFieldState st{vec3(-1.f, 0.f, 0.f), 1.f};
    const float iso[] = {0.75f};
    const float vE = sampleValue(st, field, ro),
                vX = sampleValue(st, field, ro + rd);
    float outT = 0.f;
    vec3 outN(0.f);
    uint32_t outIdx = 99u;
    CHECK(linearCrossing(
        st, field, ro, rd, 0.f, 1.f, vE, vX, iso, 1, outT, outN, outIdx));
    CHECK(nearf(outT, 0.25f));
    CHECK(nearf(outN.x, -1.f));
    CHECK(isUnit(outN));
    CHECK(glm::dot(outN, rd) <= 0.f);
  }

  // 6. Zero gradient constant field == iso: hit at tEntry, normal = -rd.
  {
    const LinearFieldState st{vec3(0.f), 0.5f};
    const float iso[] = {0.5f};
    const float vE = sampleValue(st, field, ro),
                vX = sampleValue(st, field, ro + rd);
    float outT = 99.f;
    vec3 outN(0.f);
    uint32_t outIdx = 99u;
    CHECK(linearCrossing(
        st, field, ro, rd, 0.f, 1.f, vE, vX, iso, 1, outT, outN, outIdx));
    CHECK(nearf(outT, 0.f));
    CHECK(nearf(outN.x, glm::normalize(-rd).x) && isUnit(outN));
    CHECK(glm::dot(outN, rd) <= 0.f);
  }

  // 7. Hybrid-bracket miss: f dips to 0 inside but both endpoints read 0.25, so
  // iso 0.1 (crossed twice strictly inside) is skipped by design.
  {
    const QuadFieldState st{0.5f};
    const float iso[] = {0.1f};
    const float vE = sampleValue(st, field, ro),
                vX = sampleValue(st, field, ro + rd);
    float outT = 0.f;
    vec3 outN(0.f);
    uint32_t outIdx = 0u;
    CHECK(!linearCrossing(
        st, field, ro, rd, 0.f, 1.f, vE, vX, iso, 1, outT, outN, outIdx));
  }
}

int main()
{
  testTrilinear();
  testPrepareCubic();
  testSolveCubic();
  testFirstTrilinearRootProperty();
  testLinearCrossing();
  if (g_failures) {
    std::printf("%d CHECK(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all isosurface linear host tests passed\n");
  return 0;
}
