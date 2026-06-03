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

#include "cameraCreateRay.h"
#include "gpu/gpu_debug.h"
#include "gpu_objects.h"
#include "gpu_tonemap.h"
#include "shadingState.h"
// optix
#include <optix_device.h>
// std
#include <cstdint>
// glm
#include <glm/gtc/color_space.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/packing.hpp>
// cuda
#include <vector_types.h>

#ifndef __CUDACC__
#error "gpu_util.h can only be included in device code"
#endif

namespace visrtx {

//
template <typename T_OUT, typename T_IN>
VISRTX_DEVICE T_OUT bit_cast(T_IN v)
{
  static_assert(sizeof(T_OUT) <= sizeof(T_IN),
      "bit_cast<> should only be used to cast to types equal "
      "or smaller than the input value");
  return *reinterpret_cast<T_OUT *>(&v);
}

///////////////////////////////////////////////////////////////////////////////
// Conversion functions
///////////////////////////////////////////////////////////////////////////////

// Make sure to bring global make_float* so we can access there global set of
// overload despite the definitions below
using ::make_float1, ::make_float2, ::make_float3, ::make_float4;
using ::make_int1, ::make_int2, ::make_int3, ::make_int4;
using ::make_uint1, ::make_uint2, ::make_uint3, ::make_uint4;

// clang-format off

VISRTX_DEVICE glm::vec1 make_vec1(const float1& v) { return bit_cast<glm::vec1>(v); }
VISRTX_DEVICE glm::vec2 make_vec2(const float2& v) { return bit_cast<glm::vec2>(v); }
VISRTX_DEVICE glm::vec3 make_vec3(const float3& v) { return bit_cast<glm::vec3>(v); }
VISRTX_DEVICE glm::vec4 make_vec4(const float4& v) { return bit_cast<glm::vec4>(v); }
VISRTX_DEVICE glm::ivec1 make_ivec1(const int1& v) { return bit_cast<glm::ivec1>(v); }
VISRTX_DEVICE glm::ivec2 make_ivec2(const int2& v) { return bit_cast<glm::ivec2>(v); }
VISRTX_DEVICE glm::ivec3 make_ivec3(const int3& v) { return bit_cast<glm::ivec3>(v); }
VISRTX_DEVICE glm::ivec4 make_ivec4(const int4& v) { return bit_cast<glm::ivec4>(v); }
VISRTX_DEVICE glm::uvec1 make_uvec2(const uint1& v) { return bit_cast<glm::uvec1>(v); }
VISRTX_DEVICE glm::uvec2 make_uvec2(const uint2& v) { return bit_cast<glm::uvec2>(v); }
VISRTX_DEVICE glm::uvec3 make_uvec3(const uint3& v) { return bit_cast<glm::uvec3>(v); }
VISRTX_DEVICE glm::uvec4 make_uvec4(const uint4& v) { return bit_cast<glm::uvec4>(v); }

VISRTX_DEVICE float1 make_float1(const glm::vec2& v) { return bit_cast<float1>(v); }
VISRTX_DEVICE float2 make_float2(const glm::vec2& v) { return bit_cast<float2>(v); }
VISRTX_DEVICE float3 make_float3(const glm::vec3& v) { return bit_cast<float3>(v); }
VISRTX_DEVICE float4 make_float4(const glm::vec4& v) { return bit_cast<float4>(v); }
VISRTX_DEVICE int1 make_int1(const glm::ivec1& v) { return bit_cast<int1>(v); }
VISRTX_DEVICE int2 make_int2(const glm::ivec2& v) { return bit_cast<int2>(v); }
VISRTX_DEVICE int3 make_int3(const glm::ivec3& v) { return bit_cast<int3>(v); }
VISRTX_DEVICE int4 make_int4(const glm::ivec4& v) { return bit_cast<int4>(v); }
VISRTX_DEVICE uint1 make_uint1(const glm::uvec1& v) { return bit_cast<uint1>(v); }
VISRTX_DEVICE uint2 make_uint2(const glm::uvec2& v) { return bit_cast<uint2>(v); }
VISRTX_DEVICE uint3 make_uint3(const glm::uvec3& v) { return bit_cast<uint3>(v); }
VISRTX_DEVICE uint4 make_uint4(const glm::uvec4& v) { return bit_cast<uint4>(v); }

// clang-format on

///////////////////////////////////////////////////////////////////////////////
// Utility functions //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

VISRTX_DEVICE float atomicMinf(float *address, float val)
{
  int ret = __float_as_int(*address);
  while (val < __int_as_float(ret)) {
    int old = ret;
    if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
      break;
  }
  return __int_as_float(ret);
}

VISRTX_DEVICE float atomicMaxf(float *address, float val)
{
  int ret = __float_as_int(*address);
  while (val > __int_as_float(ret)) {
    int old = ret;
    if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
      break;
  }
  return __int_as_float(ret);
}

template <typename T>
VISRTX_DEVICE void accumulateValue(T &a, const T &b, float interp)
{
  a += b * (1.f - interp);
}

template <typename T>
VISRTX_DEVICE void accumulateNormal(T &a, const T &b, float interp)
{
  accumulateValue(a, b, interp);
}

// Rec.709 luminance.
VISRTX_DEVICE float luminance(const vec3 &c)
{
  return glm::dot(c, vec3(0.2126f, 0.7152f, 0.0722f));
}

namespace detail {

VISRTX_DEVICE void packPointer(void *ptr, uint32_t &i0, uint32_t &i1)
{
  const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

VISRTX_DEVICE void *unpackPointer(uint32_t i0, uint32_t i1)
{
  const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
  void *ptr = reinterpret_cast<void *>(uptr);
  return ptr;
}

enum class PRDSelector
{
  SCREEN_SAMPLE,
  RAY_DATA
};

template <typename T>
VISRTX_DEVICE T *getPRD(PRDSelector s)
{
  if (s == PRDSelector::SCREEN_SAMPLE) {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(unpackPointer(u0, u1));
  } else {
    const uint32_t u0 = optixGetPayload_2();
    const uint32_t u1 = optixGetPayload_3();
    return reinterpret_cast<T *>(unpackPointer(u0, u1));
  }
}

} // namespace detail

VISRTX_DEVICE vec3 makeRandomColor(uint32_t i)
{
  const uint32_t mx = 13 * 17 * 43;
  const uint32_t my = 11 * 29;
  const uint32_t mz = 7 * 23 * 63;
  const uint32_t g = (i * (3 * 5 * 127) + 12312314);
  return vec3((g % mx) * (1.f / (mx - 1)),
      (g % my) * (1.f / (my - 1)),
      (g % mz) * (1.f / (mz - 1)));
}

VISRTX_DEVICE vec3 boolColor(bool pred)
{
  return pred ? vec3(0.f, 1.f, 0.f) : vec3(1.f, 0.f, 0.f);
}

// Uniform on the unit sphere via Marsaglia (1972); pdf = 1/(4*pi).
// Downstream uses: isotropic volume scatter, AO/bounce hemisphere base.
VISRTX_DEVICE vec3 randomDir(RandState &rs)
{
  const float cosTheta = 1.f - 2.f * pcg_uniform(&rs);
  const float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
  const float phi = kTwoPi * pcg_uniform(&rs);
  return vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}

VISRTX_DEVICE vec3 randomDir(RandState &rs, const vec3 &normal)
{
  const auto dir = randomDir(rs);
  return dot(dir, normal) > 0.f ? dir : -dir;
}

VISRTX_DEVICE mat3 computeOrthonormalBasis(const vec3 &normal)
{
  // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
  auto sign = normal.z >= 0.0f ? 1.0f : -1.0f;
  auto a = -1.0f / (sign + normal.z);
  auto b = normal.x * normal.y * a;
  auto u =
      vec3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
  auto v = vec3(b, sign + normal.y * normal.y * a, -normal.y);

  return mat3(u, v, normal);
}

// Cosine-weighted hemisphere sample (Malley's method); pdf = cos(theta)/pi.
VISRTX_DEVICE vec3 sampleHemisphere(RandState &rs, const vec3 &normal)
{
  const float u1 = pcg_uniform(&rs);
  const float u2 = pcg_uniform(&rs);
  const float r = sqrtf(u1);
  const float z = sqrtf(fmaxf(0.f, 1.f - r * r));
  const float phi = kTwoPi * u2;
  const vec3 sample(r * cosf(phi), r * sinf(phi), z);
  return computeOrthonormalBasis(normal) * sample;
}

VISRTX_DEVICE vec3 sampleUnitSphere(RandState &rs, const vec3 &normal)
{
  // sample unit sphere
  const float cost = 1.f - 2.f * pcg_uniform(&rs);
  const float sint = sqrtf(fmaxf(0.f, 1.f - cost * cost));
  const float phi = kTwoPi * pcg_uniform(&rs);

  return computeOrthonormalBasis(normal)
      * vec3(sint * cosf(phi), sint * sinf(phi), -cost);
}

VISRTX_DEVICE float epsilonFrom(const vec3 &P, const vec3 &dir, float t)
{
  constexpr float hitEpsilonScale = 0x1.fp-21f;
  constexpr float minHitEpsilon = 1e-8f;

  const float pMag = glm::compMax(glm::abs(P));
  const float dMag = glm::compMax(glm::abs(dir)) * t;

  return fmaxf(glm::max(pMag, dMag) * hitEpsilonScale, minHitEpsilon);
}

// Hanika's shadow-terminator fix (Ray Tracing Gems II, ch. 4): lifts a
// triangle hit point onto the smooth surface implied by per-vertex normals.
// Without this, grazing-angle shadow rays self-occlude on the planar facet
// and produce dark bands shaped like the underlying tessellation. All inputs
// must share a coordinate space.
VISRTX_DEVICE vec3 shadowTerminatorOffset(const vec3 &P,
    const vec3 &v0,
    const vec3 &v1,
    const vec3 &v2,
    const vec3 &n0,
    const vec3 &n1,
    const vec3 &n2,
    const vec3 &bary)
{
  const float du = glm::dot(P - v0, n0);
  const float dv = glm::dot(P - v1, n1);
  const float dw = glm::dot(P - v2, n2);
  const vec3 lu = du < 0.f ? -du * n0 : vec3(0.f);
  const vec3 lv = dv < 0.f ? -dv * n1 : vec3(0.f);
  const vec3 lw = dw < 0.f ? -dw * n2 : vec3(0.f);
  return P + bary.x * lu + bary.y * lv + bary.z * lw;
}

// World-space hit position lifted onto the smooth surface implied by
// per-vertex normals (Hanika shadow-terminator fix). Use this as the origin
// for direct-light/AO shadow rays so grazing-angle queries do not self-shadow
// the planar facet. Do NOT use it for path-continuation rays — transmission
// especially needs the original facet point, since the smoothed point can sit
// far enough above the facet that an "into-the-surface" offset still ends up
// outside the volume.
VISRTX_DEVICE vec3 shadingHitpoint(const SurfaceHit &hit)
{
  if (hit.geometry == nullptr || hit.geometry->type != GeometryType::TRIANGLE)
    return hit.hitpoint;

  const auto &tri = hit.geometry->tri;
  if (tri.vertexNormalsFV == nullptr && tri.vertexNormals == nullptr)
    return hit.hitpoint;

  const uvec3 idx =
      tri.indices ? tri.indices[hit.primID] : uvec3(0, 1, 2) + hit.primID * 3;
  const vec3 v0 = tri.vertices[idx.x];
  const vec3 v1 = tri.vertices[idx.y];
  const vec3 v2 = tri.vertices[idx.z];

  vec3 n0, n1, n2;
  if (tri.vertexNormalsFV != nullptr) {
    const uvec3 nidx = uvec3(0, 1, 2) + hit.primID * 3;
    n0 = tri.vertexNormalsFV[nidx.x];
    n1 = tri.vertexNormalsFV[nidx.y];
    n2 = tri.vertexNormalsFV[nidx.z];
  } else {
    n0 = tri.vertexNormals[idx.x];
    n1 = tri.vertexNormals[idx.y];
    n2 = tri.vertexNormals[idx.z];
  }

  // Hanika's tangent-plane projection assumes unit normals; user data is
  // not guaranteed to be normalized.
  n0 = normalize(n0);
  n1 = normalize(n1);
  n2 = normalize(n2);

  // populateHit.h flips hit.Ng/Ns for back-face hits so they point toward
  // the ray origin. The per-vertex normals here are still in the original
  // outward orientation; flip them too so the smooth surface bulges onto
  // the ray-origin side of the facet (otherwise Hanika lifts P away from
  // the ray origin and the trailing `+ Ng * epsilon` can land below the
  // facet).
  if (!hit.isFrontFace) {
    n0 = -n0;
    n1 = -n1;
    n2 = -n2;
  }

  const vec3 Plocal = xfmPoint(hit.instance->worldToObject, hit.hitpoint);
  const vec3 Psmooth =
      shadowTerminatorOffset(Plocal, v0, v1, v2, n0, n1, n2, hit.uvw);
  return xfmPoint(hit.instance->objectToWorld, Psmooth);
}

VISRTX_DEVICE bool pixelOutOfFrame(
    const uvec2 &pixel, const FramebufferGPUData &fb)
{
  return pixel.x >= fb.size.x || pixel.y >= fb.size.y;
}

VISRTX_DEVICE bool isFirstPixel(
    const uvec2 &pixel, const FramebufferGPUData &fb)
{
  return pixel.x == 0 && pixel.y == 0;
}

VISRTX_DEVICE bool isMiddelPixel(
    const uvec2 &pixel, const FramebufferGPUData &fb)
{
  return pixel.x == (fb.size.x / 2) && pixel.y == (fb.size.y / 2);
}

VISRTX_DEVICE bool continuesThroughSurface(const NextRay &nextRay)
{
  return (nextRay.flags & NEXT_RAY_CONTINUES_THROUGH_SURFACE) != 0u;
}

VISRTX_DEVICE vec3 sampleHDRI(const LightGPUData &ld, const vec2 &uv)
{
  return vec3(make_vec4(tex2D<::float4>(ld.hdri.radiance, uv.x, uv.y)));
}

VISRTX_DEVICE vec3 sampleHDRI(const LightGPUData &ld, const vec3 &rayDir)
{
  if (ld.type != LightType::HDRI)
    return vec3(0.f);

  constexpr float invPi = 1.f / kPi;
  constexpr float inv2Pi = 1.f / (kTwoPi);
  const vec3 d = ld.hdri.xfm * rayDir;
  const vec2 thetaPhi = sphericalCoordsFromDirection(d);
  const float u = thetaPhi.y * inv2Pi;
  const float v = thetaPhi.x * invPi;

  return sampleHDRI(ld, vec2(u, v)) * ld.hdri.scale;
}

VISRTX_DEVICE bool getBackgroundLight(
    const FrameGPUData &fd, const vec3 &rayDir, vec3 &outRadiance)
{
  // Accumulate contributions from all visible HDRI lights
  outRadiance = vec3(0.f);
  bool hasVisibleHDRI = false;

  for (size_t i = 0; i < fd.world.numHdriLightInstances; i++) {
    const auto &hdriLight = fd.world.hdriLightInstances[i];
    const auto &light = fd.registry.lights[hdriLight.lightIndex];
    if (light.hdri.visible) {
      // Transform ray direction from world space to HDRI local space
      // For orthonormal matrices, inverse = transpose
      const mat3 xfmInv = glm::transpose(mat3(hdriLight.xfm));
      const vec3 localRayDir = xfmInv * rayDir;
      // sampleHDRI applies hdri.scale; tint by light.color to match the NEE
      // radiance in sampleHDRILight (raw * hdri.scale * color), so env MIS
      // deposits identical radiance on the NEE and BSDF-escape sides.
      outRadiance += sampleHDRI(light, localRayDir) * light.color;
      hasVisibleHDRI = true;
    }
  }

  return hasVisibleHDRI;
}

// Solid-angle sampling pdf of the visible HDRI environment(s) at `rayDir`, used
// as the light-sampling density on the escape side of environment MIS. It must
// match the NEE importance pdf in sampleHDRILight exactly: raw-texel luminance
// (NO scale/color) times pdfWeight, with the same instance/HDRI transform chain
// as getBackgroundLight. Summed over visible HDRIs — exact for the single-HDRI
// case, a mixture-pdf approximation when several are visible.
VISRTX_DEVICE float envPdf(const FrameGPUData &fd, const vec3 &rayDir)
{
  float pdf = 0.0f;
  for (size_t i = 0; i < fd.world.numHdriLightInstances; i++) {
    const auto &hdriLight = fd.world.hdriLightInstances[i];
    const auto &light = fd.registry.lights[hdriLight.lightIndex];
    if (!light.hdri.visible)
      continue;
    const vec3 localRayDir = glm::transpose(mat3(hdriLight.xfm)) * rayDir;
    const vec3 d = light.hdri.xfm * localRayDir;
    const vec2 thetaPhi = sphericalCoordsFromDirection(d);
    const vec2 uv = vec2(thetaPhi.y / kTwoPi, thetaPhi.x / kPi);
    pdf += dot(sampleHDRI(light, uv), vec3(0.2126f, 0.7152f, 0.0722f))
        * light.hdri.pdfWeight;
  }
  return pdf;
}

VISRTX_DEVICE uint32_t computeGeometryPrimId(const SurfaceHit &hit)
{
  if (!hit.foundHit)
    return ~0u;
  return hit.geometry->primitiveId ? hit.geometry->primitiveId[hit.primID]
                                   : hit.primID;
}

///////////////////////////////////////////////////////////////////////////////
// Outputs ////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename T>
VISRTX_DEVICE void accumValue(T *arr, size_t idx, const T &v)
{
  if (!arr)
    return;

  arr[idx] += v;
}

VISRTX_DEVICE bool accumDepth(float *arr, size_t idx, const float &v)
{
  if (!arr)
    return true; // no previous depth to compare with

  if (v < arr[idx]) {
    arr[idx] = v;
    return true;
  } else {
    return false;
  }
}

VISRTX_DEVICE uint32_t pixelIndex(
    const FramebufferGPUData &fb, const uvec2 &pixel)
{
  return pixel.x + pixel.y * fb.size.x;
}

} // namespace detail

VISRTX_DEVICE void setPixelIds(const FramebufferGPUData &fb,
    const uvec2 &pixel,
    uint32_t primID,
    uint32_t objID,
    uint32_t instID)
{
  const uint32_t idx = detail::pixelIndex(fb, pixel);

  if (fb.buffers.primID)
    fb.buffers.primID[idx] = primID;
  if (fb.buffers.objID)
    fb.buffers.objID[idx] = objID;
  if (fb.buffers.instID)
    fb.buffers.instID[idx] = instID;
}

VISRTX_DEVICE void setPixelIds(const FramebufferGPUData &fb,
    const uvec2 &pixel,
    const float depth,
    uint32_t primID,
    uint32_t objID,
    uint32_t instID)
{
  const uint32_t idx = detail::pixelIndex(fb, pixel);
  if (detail::accumDepth(fb.buffers.depth, idx, depth)) {
    if (fb.buffers.primID)
      fb.buffers.primID[idx] = primID;
    if (fb.buffers.objID)
      fb.buffers.objID[idx] = objID;
    if (fb.buffers.instID)
      fb.buffers.instID[idx] = instID;
  }
}

// Per-pixel, per-channel Welford soft-clamp for firefly suppression.
//
// Two regimes keyed on the pixel's own sample count n:
//   * warmup (n < warmupSamples): per-channel stats are too sparse for a
//     variance-based cap, so clamp each channel to a generous multiple of its
//     running mean. This catches a firefly from the 2nd sample on (the 1st has
//     no prior).
//   * steady (n >= warmupSamples): clamp each channel to mean + k*stddev.
//
// In both regimes the Welford stats are updated from the *clamped* value: a
// sample below its cap contributes its true value (so σ tracks the well-behaved
// bulk), but a sample above the cap contributes only the capped value. This is
// the base-excluding threshold — letting raw outliers into the stats lets one
// firefly inflate σ enough to raise its own future cap, so a moderate k never
// fires (the σ-inflation trap). Feeding the clamped value bounds that inflation,
// which is what lets k drop to a value that actually bites. The cost is that a
// genuinely legitimate >kσ excursion on a high-variance pixel is clipped and
// cannot grow the cap — unavoidable for a per-pixel online clamp, which is why
// CLAMP is the deliberately-aggressive, biased mode.
//
// Each channel is clamped independently to its own cap, so a chromatic
// (single-channel) outlier is caught even when its luminance is unremarkable,
// without a near-zero channel dragging the whole (saturated-color) pixel dark.
VISRTX_DEVICE vec4 fireflyClamp(
    PixelLumStats *lumStatsBuf, uint32_t idx, vec4 color, float kSigma, int warmupSamples)
{
  constexpr float kWarmupCapFactor = 8.0f; // warmup cap = factor * running mean

  if (!lumStatsBuf)
    return color;

  PixelLumStats s = lumStatsBuf[idx];
  const vec3 orig = vec3(color);
  const bool warm = s.n < float(warmupSamples);

  vec3 clamped = orig;
  if (s.n >= 1.0f) {
    for (int k = 0; k < 3; ++k) {
      const float L = orig[k];
      if (!(L > 0.0f))
        continue;
      float cap;
      if (warm) {
        cap = kWarmupCapFactor * s.mean[k];
      } else {
        // Needs >=2 samples for a sample variance; with warmupSamples==1 the
        // steady branch is reachable at n==1, where m2/(n-1) is 0/0.
        const float variance =
            s.n > 1.0f ? fmaxf(s.m2[k] / (s.n - 1.0f), 0.0f) : 0.0f;
        cap = s.mean[k] + kSigma * sqrtf(variance);
      }
      if (cap > 0.0f && L > cap)
        clamped[k] = cap;
    }
  }

  // Welford update from the clamped value in both regimes: a within-cap sample
  // updates with its true value, an outlier only with the bounded cap value.
  const float n = s.n + 1.0f;
  for (int k = 0; k < 3; ++k) {
    const float delta = clamped[k] - s.mean[k];
    s.mean[k] += delta / n;
    s.m2[k] += delta * (clamped[k] - s.mean[k]);
  }
  s.n = n;
  lumStatsBuf[idx] = s;

  return vec4(clamped, color.a);
}

VISRTX_DEVICE void accumPixelSample(const FrameGPUData &frame,
    const uvec2 &pixel,
    const vec4 &color,
    const vec3 &albedo,
    const vec3 &normal)
{
  const auto &fb = frame.fb;
  const uint32_t idx = detail::pixelIndex(fb, pixel);

  vec4 c;
  switch (frame.renderer.fireflyFilterMode) {
  case FireflyFilterMode::TONEMAP:
    c = detail::tonemap(color);
    break;
  case FireflyFilterMode::CLAMP:
    c = fireflyClamp(fb.buffers.lumStats,
        idx,
        color,
        frame.renderer.fireflyFilterSigma,
        frame.renderer.fireflyFilterWarmup);
    break;
  case FireflyFilterMode::TRIM:
    // Accumulate the raw sample (colorAccumulation keeps the running sum) while
    // tracking the `trim` brightest samples this pixel has seen and a luminance
    // Welford. The trimmed mean at resolve removes only the tracked samples
    // that a base-excluding threshold flags as outliers, so clean pixels drop
    // nothing (exact mean) and the dropped fraction -> 0 with spp.
    if (fb.buffers.trimTopK) {
      const int trim = frame.renderer.fireflyFilterTrim;
      const float L = luminance(vec3(color));
      // A non-finite sample poisons the Welford mean/variance and drives the
      // resolve threshold to inf, so nothing ever trims and the pixel stays
      // non-finite forever. Drop it outright -- the very firefly TRIM exists to
      // suppress must not survive into colorAccumulation.
      if (glm::isnan(L) || glm::isinf(L))
        return;
      vec4 *slots = fb.buffers.trimTopK + size_t(idx) * trim;
      int minSlot = 0;
      float minL = slots[0].w;
      for (int i = 1; i < trim; ++i) {
        if (slots[i].w < minL) {
          minL = slots[i].w;
          minSlot = i;
        }
      }
      if (L > minL)
        slots[minSlot] = vec4(vec3(color), L);

      PixelLumStats &s = fb.buffers.lumStats[idx];
      const float n = s.n + 1.f;
      const float delta = L - s.mean.x;
      s.mean.x += delta / n;
      s.m2.x += delta * (L - s.mean.x);
      s.n = n;
    }
    c = color;
    break;
  default:
    c = color;
    break;
  }

  detail::accumValue(fb.buffers.colorAccumulation, idx, c);
  detail::accumValue(fb.buffers.albedo, idx, albedo);
  detail::accumValue(fb.buffers.normal, idx, normal);
}

} // namespace visrtx
