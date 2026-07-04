// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu/gpu_decl.h"  // VISRTX_DEVICE (== inline on host)
#include <glm/glm.hpp>
#include <cmath>  // fabsf/sqrtf/fminf/fmaxf — self-contained, not via glm's transitive include
#include <cstdint>

namespace visrtx {

using glm::vec3;

// Incomplete is sufficient: linearCrossing only forwards `field` by reference to
// the per-sampler sampleValue()/sampleNormal() hooks (resolved at instantiation
// in device code). Forward-declaring keeps this header host-parseable without
// the device-only gpu_objects.h (which pulls in OptiX/CUDA).
struct SpatialFieldGPUData;

// Trilinear interpolant of the 8 cell corners at local coord u in [0,1]^3.
// Corner index: c[i + 2*j + 4*k] is the value at local corner (i,j,k).
VISRTX_DEVICE float trilinearValue(const float c[8], const vec3 &u)
{
  const float x = u.x, y = u.y, z = u.z, X = 1.f - x, Y = 1.f - y, Z = 1.f - z;
  return c[0] * X * Y * Z + c[1] * x * Y * Z + c[2] * X * y * Z + c[3] * x * y * Z
      + c[4] * X * Y * z + c[5] * x * Y * z + c[6] * X * y * z + c[7] * x * y * z;
}

namespace detail {
constexpr float kCoeffDegenerateEps = 1e-12f; // |leading coeff| below this drops a degree
constexpr float kExtremumMargin = 1e-5f;       // ignore g' roots within this of an endpoint
constexpr float kRootTolerance = 1e-6f;        // |g| below this counts as a root
constexpr int kRootRefineIters = 8;            // false-position refinement steps
} // namespace detail

// Monomial form of the cubic through the fixed nodes {0,1/3,2/3,1} plus the
// monotone-span boundaries (roots of g' partitioning [0,1]). Splitting prepare
// from solve lets a voxel with N isovalues fit the cubic + extrema ONCE: across
// isovalues the cubic only shifts by the constant -iso, so k1,k2,k3 — and hence
// g' and its roots — are isovalue-independent; only k0 moves.
struct CubicPrep
{
  float k0, k1, k2, k3; // g(s) = ((k3 s + k2) s + k1) s + k0
  float bound[4]; // monotone-span boundaries; `nb` of them used, in [0,1]
  int nb;
};

// Fit the cubic through the four node VALUES (constant inverse Vandermonde) and
// partition [0,1] at g's extrema. Isovalue-independent — call once per voxel.
VISRTX_DEVICE CubicPrep prepareCubic(float n0, float n1, float n2, float n3)
{
  CubicPrep p;
  p.k0 = n0;
  p.k1 = -5.5f * n0 + 9.f * n1 - 4.5f * n2 + n3;
  p.k2 = 9.f * n0 - 22.5f * n1 + 18.f * n2 - 4.5f * n3;
  p.k3 = -4.5f * n0 + 13.5f * n1 - 13.5f * n2 + 4.5f * n3;

  // Build up to 4 monotone-span boundaries [0, e0, e1, 1] from g' roots.
  p.bound[0] = 0.f; p.bound[1] = 1.f; p.bound[2] = 1.f; p.bound[3] = 1.f;
  p.nb = 2;
  const float a = 3.f * p.k3, b = 2.f * p.k2, cc = p.k1; // g'(s)=a s^2+b s+c
  if (fabsf(a) > detail::kCoeffDegenerateEps) {
    const float disc = b * b - 4.f * a * cc;
    if (disc > 0.f) {
      const float sq = sqrtf(disc);
      float r0 = (-b - sq) / (2.f * a), r1 = (-b + sq) / (2.f * a);
      if (r0 > r1) { const float t = r0; r0 = r1; r1 = t; }
      float tmp[4]; int n = 0;
      tmp[n++] = 0.f;
      if (r0 > detail::kExtremumMargin && r0 < 1.f - detail::kExtremumMargin) tmp[n++] = r0;
      if (r1 > detail::kExtremumMargin && r1 < 1.f - detail::kExtremumMargin) tmp[n++] = r1;
      tmp[n++] = 1.f;
      for (int i = 0; i < n; ++i) p.bound[i] = tmp[i];
      p.nb = n;
    }
  } else if (fabsf(b) > detail::kCoeffDegenerateEps) {
    const float r = -cc / b;
    if (r > detail::kExtremumMargin && r < 1.f - detail::kExtremumMargin) { p.bound[1] = r; p.bound[2] = 1.f; p.nb = 3; }
  }
  return p;
}

// Smallest root in [0,1] of (prepared cubic) - iso. Reuses the prepared spans;
// only the constant term shifts by -iso. Returns false on no crossing.
VISRTX_DEVICE bool solveCubic(const CubicPrep &p, float iso, float &outS)
{
  const float k0 = p.k0 - iso;
  auto gc = [&](float s) { return ((p.k3 * s + p.k2) * s + p.k1) * s + k0; };

  // First monotone span with a sign change -> regula-falsi to the root.
  for (int i = 0; i + 1 < p.nb; ++i) {
    float lo = p.bound[i], hi = p.bound[i + 1];
    float glo = gc(lo), ghi = gc(hi);
    if (glo == 0.f) { outS = lo; return true; }
    if ((glo < 0.f) == (ghi < 0.f)) continue;
    // Illinois-modified false position: halving the retained endpoint's value
    // breaks one-sided stagnation so the bracket shrinks from both ends and m
    // converges superlinearly. glo,ghi have opposite signs here, so
    // ghi-glo = |glo|+|ghi| > 0 and m is strictly inside [lo,hi].
    float m; // assigned by the false-position step below before any use
    for (int it = 0; it < detail::kRootRefineIters; ++it) {
      m = (lo * ghi - hi * glo) / (ghi - glo);
      const float gm = gc(m);
      if (fabsf(gm) < detail::kRootTolerance) { outS = m; return true; }
      if ((gm < 0.f) == (glo < 0.f)) { // gm on lo's side: advance lo, retain hi
        lo = m; glo = gm; ghi *= 0.5f;
      } else { // gm on hi's side: advance hi, retain lo
        hi = m; ghi = gm; glo *= 0.5f;
      }
    }
    // The bracket already proved a sign change, so a root exists; after the
    // bounded Illinois iterations the latest iterate is the best estimate (no
    // residual gate needed -- the return is intentional).
    outS = m; // converged estimate (two-sided shrink), not the bracket midpoint
    return true;
  }
  return false;
}

// Smallest root in [0,1] of the cubic sampled at {0,1/3,2/3,1} as g0..g3 (already
// iso-subtracted). Thin prepare+solve wrapper, kept for the host unit test.
VISRTX_DEVICE bool firstCubicRoot(
    float g0, float g1, float g2, float g3, float &outS)
{
  return solveCubic(prepareCubic(g0, g1, g2, g3), 0.f, outS);
}

// Convenience wrapper: smallest root in [0,1] of trilinear(lerp(aEntry,aExit,s))
// - iso, sampling the cubic at the four fixed nodes. (Exercised by the host
// unit test; the device path samples the field directly via linearCrossing.)
VISRTX_DEVICE bool firstTrilinearRoot(
    const float c[8], const vec3 &aEntry, const vec3 &aExit, float iso, float &outS)
{
  const vec3 d = aExit - aEntry;
  return firstCubicRoot(trilinearValue(c, aEntry) - iso,
      trilinearValue(c, aEntry + (1.f / 3.f) * d) - iso,
      trilinearValue(c, aEntry + (2.f / 3.f) * d) - iso,
      trilinearValue(c, aExit) - iso,
      outS);
}

// Object-space field gradient at a linear-isosurface hit. The hit value is
// already known (≈ the matched isovalue, vHit), so NanoVDB grids override this
// with a forward difference reusing vHit -- 3 taps vs central difference's 6,
// still spanning into the neighbour cell so it stays smooth (unlike the faceted
// per-cell analytic gradient). The generic fallback is the shared central-
// difference sampleNormal (structured/raw is hardware tex3D, already cheap; the
// volume integrator keeps its own sampleNormal). Resolved by ADL at instantiation.
template <typename S>
VISRTX_DEVICE vec3 isosurfaceHitGradient(
    const S &state, const SpatialFieldGPUData &field, const vec3 &p, float /*vHit*/)
{
  return sampleNormal(state, field, p);
}

// Shared narrow-phase for one voxel under linear filtering. tEntry/tExit bound
// the voxel along the ray; vEntry/vExit are the field values there (hybrid
// bracket, carried by the caller). For each isovalue that straddles the bracket
// it fits the field's cubic from hardware samples and solves; the nearest
// crossing across isovalues wins. Writes the hit t, oriented+normalized object
// normal, and isovalue index; returns true on a hit. sampleValue/sampleNormal
// are resolved per-sampler by ADL at instantiation.
template <typename S>
VISRTX_DEVICE bool linearCrossing(const S &state,
    const SpatialFieldGPUData &field,
    const vec3 &ro,
    const vec3 &rd,
    float tEntry,
    float tExit,
    float vEntry,
    float vExit,
    const float *isovals,
    uint32_t numIsovalues,
    float &outT,
    vec3 &outNormal,
    uint32_t &outIdx)
{
  const float lo = fminf(vEntry, vExit), hi = fmaxf(vEntry, vExit);

  // Hybrid bracket: only sample/solve when an isovalue straddles the segment
  // ends. The carried endpoint values (vEntry/vExit) avoid re-fetching them.
  bool bracketed = false;
  for (uint32_t i = 0; i < numIsovalues; ++i)
    if (isovals[i] >= lo && isovals[i] <= hi) {
      bracketed = true;
      break;
    }
  if (!bracketed)
    return false;

  // Fit the field along the ray from 4 hardware samples: trilinear-along-a-ray
  // is cubic, and sampling the field directly (like the old march) avoids the
  // per-cell corner reconstruction whose half-texel cell mismatch faceted the
  // surface (AO crease lattice). Endpoints reuse vEntry/vExit; sample the two
  // interior nodes.
  const float dt = tExit - tEntry;
  const float f0 = vEntry;
  const float f1 = sampleValue(state, field, ro + (tEntry + dt * (1.f / 3.f)) * rd);
  const float f2 = sampleValue(state, field, ro + (tEntry + dt * (2.f / 3.f)) * rd);
  const float f3 = vExit;

  // Fit the cubic + extrema once; only k0 shifts per isovalue (solveCubic).
  const CubicPrep prep = prepareCubic(f0, f1, f2, f3);

  float bestS = 2.f;
  uint32_t bestIdx = 0;
  for (uint32_t i = 0; i < numIsovalues; ++i) {
    const float iv = isovals[i];
    if (iv < lo || iv > hi)
      continue;
    float s;
    if (solveCubic(prep, iv, s) && s < bestS) {
      bestS = s;
      bestIdx = i;
    }
  }

  if (bestS > 1.f)
    return false;

  outT = tEntry + bestS * dt;
  // Forward-difference gradient reusing the hit value (isovals[bestIdx]); see
  // isosurfaceHitGradient. NanoVDB: 3 taps; others fall back to central diff.
  const vec3 pHit = ro + outT * rd;
  vec3 n = isosurfaceHitGradient(state, field, pHit, isovals[bestIdx]);
  // A (near-)zero gradient at a field extremum/saddle makes normalize(n) NaN,
  // which poisons shading. Fall back to facing the incoming ray.
  if (dot(n, n) < 1e-12f) {
    outNormal = normalize(-rd);
  } else {
    if (dot(n, rd) > 0.f)
      n = -n;
    outNormal = normalize(n);
  }
  outIdx = bestIdx;
  return true;
}

} // namespace visrtx
