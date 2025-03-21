/*
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu/cameraCreateRay.h"
#include "gpu/gpu_objects.h"
// optix
#include <optix_device.h>
// std
#include <cstdint>

#ifndef __CUDACC__
#error "gpu_util.h can only be included in device code"
#endif

namespace visrtx {

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

template <typename T_OUT, typename T_IN>
VISRTX_DEVICE T_OUT bit_cast(T_IN v)
{
  static_assert(sizeof(T_OUT) <= sizeof(T_IN),
      "bit_cast<> should only be used to cast to types equal "
      "or smaller than the input value");
  return *reinterpret_cast<T_OUT *>(&v);
}

VISRTX_DEVICE vec3 make_vec3(const ::float3 &v)
{
  return vec3(v.x, v.y, v.z);
}

VISRTX_DEVICE vec4 make_vec4(const ::float4 &v)
{
  return vec4(v.x, v.y, v.z, v.w);
}

template <typename T>
VISRTX_DEVICE void accumulateValue(T &a, const T &b, float interp)
{
  a += b * (1.f - interp);
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

VISRTX_DEVICE vec3 sampleUnitSphere(RandState &rs, const vec3 &normal)
{
  // sample unit sphere
  const float cost = 1.f - 2.f * curand_uniform(&rs);
  const float sint = sqrtf(fmaxf(0.f, 1.f - cost * cost));
  const float phi = 2.f * float(M_PI) * curand_uniform(&rs);
  // make ortho basis and transform to ray-centric coordinates:
  const vec3 w = normal;
  const vec3 v = fabsf(w.x) > fabsf(w.y) ? normalize(vec3(-w.z, 0.f, w.x))
                                         : normalize(vec3(0.f, w.z, -w.y));
  const vec3 u = cross(v, w);
  return normalize(sint * cosf(phi) * u + sint * sinf(phi) * v + cost * -w);
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

VISRTX_DEVICE vec3 sampleHDRI(const LightGPUData &ld, const vec3 &rayDir)
{
  if (ld.type != LightType::HDRI)
    return vec3(0.f);

  constexpr float invPi = 1.f / float(M_PI);
  constexpr float inv2Pi = 1.f / (2.f * float(M_PI));
  const vec3 d = ld.hdri.xfm * rayDir;
  const float theta = pbrtSphericalTheta(d);
  const float phi = pbrtSphericalPhi(d);
  const float u = phi * inv2Pi;
  const float v = theta * invPi;

  return vec3(make_vec4(tex2D<::float4>(ld.hdri.radiance, u, v)))
      * ld.hdri.scale;
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
  const LightGPUData *hdri =
      fd.world.hdri != -1 ? &fd.registry.lights[fd.world.hdri] : nullptr;
  if (hdri && hdri->hdri.visible)
    return vec4(sampleHDRI(*hdri, rayDir), 1.f);
  else
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

template <typename T>
VISRTX_DEVICE void accumValue(T *arr, size_t idx, size_t fid, const T &v)
{
  if (!arr)
    return;

  if (fid == 0)
    arr[idx] = v;
  else
    arr[idx] += v;
}

VISRTX_DEVICE bool accumDepth(
    float *arr, size_t idx, size_t fid, const float &v)
{
  if (!arr)
    return true; // no previous depth to compare with

  const bool closerSample = fid == 0 || v < arr[idx];

  if (closerSample)
    arr[idx] = v;

  return closerSample;
}

VISRTX_DEVICE void writeOutputColor(const FramebufferGPUData &fb,
    const vec4 &color,
    const uint32_t idx,
    const int frameIDOffset)
{
  const auto c = color / float(fb.frameID + frameIDOffset + 1);
  if (fb.format == FrameFormat::SRGB) {
    fb.buffers.outColorUint[idx] =
        glm::packUnorm4x8(glm::convertLinearToSRGB(c));
  } else if (fb.format == FrameFormat::UINT)
    fb.buffers.outColorUint[idx] = glm::packUnorm4x8(c);
  else
    fb.buffers.outColorVec4[idx] = c;
}

VISRTX_DEVICE uint32_t pixelIndex(
    const FramebufferGPUData &fb, const uvec2 &pixel)
{
  return pixel.x + pixel.y * fb.size.x;
}

} // namespace detail

VISRTX_DEVICE void accumResults(const FramebufferGPUData &fb,
    const uvec2 &pixel,
    const vec4 &color,
    float depth,
    const vec3 &albedo,
    const vec3 &normal,
    uint32_t primID,
    uint32_t objID,
    uint32_t instID,
    const int frameIDOffset = 0)
{
  const uint32_t idx = detail::pixelIndex(fb, pixel);

  const auto frameID = fb.frameID + frameIDOffset;

  detail::accumValue(fb.buffers.colorAccumulation, idx, frameID, color);
  detail::accumValue(fb.buffers.albedo, idx, frameID, albedo);
  detail::accumValue(fb.buffers.normal, idx, frameID, normal);

  if (detail::accumDepth(fb.buffers.depth, idx, frameID, depth)) {
    if (fb.buffers.primID)
      fb.buffers.primID[idx] = primID;
    if (fb.buffers.objID)
      fb.buffers.objID[idx] = objID;
    if (fb.buffers.instID)
      fb.buffers.instID[idx] = instID;
  }

  const auto accumColor = fb.buffers.colorAccumulation[idx];
  detail::writeOutputColor(fb, accumColor, idx, frameIDOffset);

  if (fb.checkerboardID == 0 && frameID == 0) {
    auto adjPix = uvec2(pixel.x + 1, pixel.y + 0);
    if (!pixelOutOfFrame(adjPix, fb)) {
      detail::writeOutputColor(
          fb, accumColor, detail::pixelIndex(fb, adjPix), frameIDOffset);
    }

    adjPix = uvec2(pixel.x + 0, pixel.y + 1);
    if (!pixelOutOfFrame(adjPix, fb)) {
      detail::writeOutputColor(
          fb, accumColor, detail::pixelIndex(fb, adjPix), frameIDOffset);
    }

    adjPix = uvec2(pixel.x + 1, pixel.y + 1);
    if (!pixelOutOfFrame(adjPix, fb)) {
      detail::writeOutputColor(
          fb, accumColor, detail::pixelIndex(fb, adjPix), frameIDOffset);
    }
  }
}

} // namespace visrtx
