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

VISRTX_DEVICE vec3 randomDir(RandState &rs)
{
#if 0
  const float r1 = curand_uniform(&rs);
  const float r2 = curand_uniform(&rs);
  return normalize(vec3(cos(2 * float(M_PI) * r1) * sqrt(1 - (r2 * r2)),
      sin(2 * float(M_PI) * r1) * sqrt(1 - (r2 * r2)),
      r2 * r2));
#else
  const auto r = curand_uniform4(&rs);
  return normalize((2.f * vec3(r.x, r.y, r.z)) - vec3(1.f));
#endif
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

VISRTX_DEVICE vec3 sampleHemisphere(RandState &rs, const vec3 &normal)
{
  auto z = curand_uniform(&rs);
  auto r = sqrtf(1.f - sqrt(z));
  auto phi = 2.0f * float(M_PI) * curand_uniform(&rs);

  auto sample = vec3(r * cos(phi), r * sin(phi), z);

  return computeOrthonormalBasis(normal) * sample;
}

VISRTX_DEVICE vec3 sampleUnitSphere(RandState &rs, const vec3 &normal)
{
  // sample unit sphere
  const float cost = 1.f - 2.f * curand_uniform(&rs);
  const float sint = sqrtf(fmaxf(0.f, 1.f - cost * cost));
  const float phi = 2.f * float(M_PI) * curand_uniform(&rs);

  return computeOrthonormalBasis(normal)
      * vec3(sint * cosf(phi), sint * sinf(phi), -cost);
}

#define ulpEpsilon 0x1.fp-21

VISRTX_DEVICE float epsilonFrom(const vec3 &P, const vec3 &dir, float t)
{
  return glm::compMax(vec4(abs(P), glm::compMax(abs(dir)) * t)) * ulpEpsilon;
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

VISRTX_DEVICE vec3 sampleHDRI(const LightGPUData &ld, const vec2 &uv)
{
  return vec3(make_vec4(tex2D<::float4>(ld.hdri.radiance, uv.x, uv.y)));
}

VISRTX_DEVICE vec3 sampleHDRI(const LightGPUData &ld, const vec3 &rayDir)
{
  if (ld.type != LightType::HDRI)
    return vec3(0.f);

  constexpr float invPi = 1.f / float(M_PI);
  constexpr float inv2Pi = 1.f / (2.f * float(M_PI));
  const vec3 d = ld.hdri.xfm * rayDir;
  const vec2 thetaPhi = sphericalCoordsFromDirection(d);
  const float u = thetaPhi.y * inv2Pi;
  const float v = thetaPhi.x * invPi;

  return sampleHDRI(ld, vec2(u, v)) * ld.hdri.scale;
}

VISRTX_DEVICE vec4 getBackgroundImage(
    const RendererGPUData &rd, const vec2 &loc)
{
  return rd.backgroundMode == BackgroundMode::COLOR
      ? rd.background.color
      : make_vec4(tex2D<::float4>(rd.background.texobj, loc.x, loc.y));
}

VISRTX_DEVICE vec4 getBackground(
    const FrameGPUData &fd, const vec2 &loc, const vec3 &rayDir)
{
  // Accumulate contributions from all visible HDRI lights
  vec3 hdriContribution = vec3(0.f);
  bool hasVisibleHDRI = false;

  for (size_t i = 0; i < fd.world.numHdriLightInstances; i++) {
    const auto &hdriLight = fd.world.hdriLightInstances[i];
    const auto &light = fd.registry.lights[hdriLight.lightIndex];
    if (light.hdri.visible) {
      // Transform ray direction from world space to HDRI local space
      // For orthonormal matrices, inverse = transpose
      const mat3 xfmInv = glm::transpose(mat3(hdriLight.xfm));
      const vec3 localRayDir = xfmInv * rayDir;
      hdriContribution += sampleHDRI(light, localRayDir);
      hasVisibleHDRI = true;
    }
  }

  if (hasVisibleHDRI)
    return vec4(hdriContribution, 1.f);

  // No visible HDRI, use background image/color
  return getBackgroundImage(fd.renderer, loc);
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

VISRTX_DEVICE
vec3 tonemap(vec3 v)
{
  return v / (1.0f + max(0.0f, compMax(v)));
}

VISRTX_DEVICE
vec3 inverseTonemap(vec3 v)
{
  return v / max(1e-12f, 1.f - compMax(v));
}

VISRTX_DEVICE
vec4 tonemap(vec4 v)
{
  return vec4(tonemap(vec3(v)), v.w);
}

VISRTX_DEVICE
vec4 inverseTonemap(vec4 v)
{
  return vec4(inverseTonemap(vec3(v)), v.w);
}

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

VISRTX_DEVICE void writeOutputColor(
    const FramebufferGPUData &fb, const vec4 &color, const uint32_t idx)
{
  if (fb.format == FrameFormat::SRGB) {
    fb.buffers.outColorUint[idx] =
        glm::packUnorm4x8(glm::convertLinearToSRGB(color));
  } else if (fb.format == FrameFormat::UINT)
    fb.buffers.outColorUint[idx] = glm::packUnorm4x8(color);
  else
    fb.buffers.outColorVec4[idx] = color;
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

VISRTX_DEVICE void accumPixelSample(const FrameGPUData &frame,
    const uvec2 &pixel,
    const vec4 &color,
    const vec3 &albedo,
    const vec3 &normal,
    const int frameIDOffset = 0)
{
  const auto &fb = frame.fb;
  const uint32_t idx = detail::pixelIndex(fb, pixel);
  const auto frameID = fb.frameID + frameIDOffset;

  // Conditionally apply tonemapping during accumulation
  if (frame.renderer.tonemap)
    detail::accumValue(
        fb.buffers.colorAccumulation, idx, detail::tonemap(color));
  else
    detail::accumValue(fb.buffers.colorAccumulation, idx, color);
  detail::accumValue(fb.buffers.albedo, idx, albedo);
  detail::accumValue(fb.buffers.normal, idx, normal);

  const auto accumColor = fb.buffers.colorAccumulation[idx];
  // Conditionally apply inverse tonemapping on output
  const float frameDivisor = float(fb.frameID + frameIDOffset + 1);
  const auto normalizedColor = accumColor / frameDivisor;
  const auto outputColor = frame.renderer.tonemap
      ? detail::inverseTonemap(normalizedColor)
      : normalizedColor;

  detail::writeOutputColor(fb, outputColor, idx);

  if (fb.checkerboardID == 0 && frameID == 0) {
    auto adjPix = uvec2(pixel.x + 1, pixel.y + 0);
    if (!pixelOutOfFrame(adjPix, fb)) {
      detail::writeOutputColor(fb, outputColor, detail::pixelIndex(fb, adjPix));
    }

    adjPix = uvec2(pixel.x + 0, pixel.y + 1);
    if (!pixelOutOfFrame(adjPix, fb)) {
      detail::writeOutputColor(fb, outputColor, detail::pixelIndex(fb, adjPix));
    }

    adjPix = uvec2(pixel.x + 1, pixel.y + 1);
    if (!pixelOutOfFrame(adjPix, fb)) {
      detail::writeOutputColor(fb, outputColor, detail::pixelIndex(fb, adjPix));
    }
  }
}

} // namespace visrtx
