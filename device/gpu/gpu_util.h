/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

RT_FUNCTION float atomicMinf(float *address, float val)
{
  int ret = __float_as_int(*address);
  while (val < __int_as_float(ret)) {
    int old = ret;
    if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
      break;
  }
  return __int_as_float(ret);
}

RT_FUNCTION float atomicMaxf(float *address, float val)
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
RT_FUNCTION T_OUT bit_cast(T_IN v)
{
  static_assert(sizeof(T_OUT) <= sizeof(T_IN),
      "bit_cast<> should only be used to cast to types equal "
      "or smaller than the input value");
  return *reinterpret_cast<T_OUT *>(&v);
}

RT_FUNCTION vec3 make_vec3(const ::float3 &v)
{
  return vec3(v.x, v.y, v.z);
}

RT_FUNCTION vec4 make_vec4(const ::float4 &v)
{
  return vec4(v.x, v.y, v.z, v.w);
}

template <typename T>
RT_FUNCTION void accumulateValue(T &a, const T &b, float interp)
{
  a += b * (1.f - interp);
}

namespace detail {

RT_FUNCTION void packPointer(void *ptr, uint32_t &i0, uint32_t &i1)
{
  const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

RT_FUNCTION void *unpackPointer(uint32_t i0, uint32_t i1)
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
RT_FUNCTION T *getPRD(PRDSelector s)
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

RT_FUNCTION vec3 makeRandomColor(uint32_t i)
{
  const uint32_t mx = 13 * 17 * 43;
  const uint32_t my = 11 * 29;
  const uint32_t mz = 7 * 23 * 63;
  const uint32_t g = (i * (3 * 5 * 127) + 12312314);
  return vec3((g % mx) * (1.f / (mx - 1)),
      (g % my) * (1.f / (my - 1)),
      (g % mz) * (1.f / (mz - 1)));
}

RT_FUNCTION vec3 boolColor(bool pred)
{
  return pred ? vec3(0.f, 1.f, 0.f) : vec3(1.f, 0.f, 0.f);
}

RT_FUNCTION vec3 randomDir(RandState &rs)
{
#if 1
  const auto r = vec2(curand_uniform(&rs), curand_uniform(&rs));
  return normalize(vec3(cos(2 * M_PI * r.x) * sqrt(1 - (r.y * r.y)),
      sin(2 * M_PI * r.x) * sqrt(1 - (r.y * r.y)),
      r.y * r.y));
#else
  const auto r = curand_uniform4(&rs);
  return normalize((2.f * vec3(r.x, r.y, r.z)) - vec3(1.f));
#endif
}

RT_FUNCTION vec3 randomDir(RandState &rs, const vec3 &normal)
{
  const auto dir = randomDir(rs);
  return dot(dir, normal) > 0.f ? dir : -dir;
}

RT_FUNCTION vec3 sampleUnitSphere(RandState &rs, const vec3 &normal)
{
  // sample unit sphere
  const float cost = 1.f - 2.f * curand_uniform(&rs);
  const float sint = sqrtf(fmaxf(0.f, 1.f - cost * cost));
  const float phi = 2.f * M_PI * curand_uniform(&rs);
  // make ortho basis and transform to ray-centric coordinates:
  const vec3 w = normal;
  const vec3 v = fabsf(w.x) > fabsf(w.y) ? normalize(vec3(-w.z, 0.f, w.x))
                                         : normalize(vec3(0.f, w.z, -w.y));
  const vec3 u = cross(v, w);
  return normalize(sint * cosf(phi) * u + sint * sinf(phi) * v + cost * -w);
}

#define ulpEpsilon 0x1.fp-21

RT_FUNCTION float epsilonFrom(const vec3 &P, const vec3 &dir, float t)
{
  return glm::compMax(vec4(abs(P), glm::compMax(abs(dir)) * t)) * ulpEpsilon;
}

RT_FUNCTION bool pixelOutOfFrame(
    const uvec2 &pixel, const FramebufferGPUData &fb)
{
  return pixel.x >= fb.size.x || pixel.y >= fb.size.y;
}

RT_FUNCTION bool isFirstPixel(const uvec2 &pixel, const FramebufferGPUData &fb)
{
  return pixel.x == 0 && pixel.y == 0;
}

RT_FUNCTION bool isMiddelPixel(const uvec2 &pixel, const FramebufferGPUData &fb)
{
  return pixel.x == (fb.size.x / 2) && pixel.y == (fb.size.y / 2);
}

///////////////////////////////////////////////////////////////////////////////
// Outputs ////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename T>
RT_FUNCTION void accumValue(T *arr, size_t idx, size_t fid, const T &v)
{
  if (!arr)
    return;

  if (fid == 0)
    arr[idx] = v;
  else
    arr[idx] += v;
}

RT_FUNCTION void accumDepth(float *arr, size_t idx, size_t fid, const float &v)
{
  if (!arr)
    return;

  if (fid == 0)
    arr[idx] = v;
  else
    arr[idx] = min(arr[idx], v);
}

RT_FUNCTION void writeOutputColor(
    const FramebufferGPUData &fb, const vec4 &color, uint32_t idx)
{
  const auto c = color * fb.invFrameID;
  if (fb.format == FrameFormat::SRGB) {
    fb.buffers.outColorUint[idx] =
        glm::packUnorm4x8(glm::convertLinearToSRGB(c));
  } else if (fb.format == FrameFormat::UINT)
    fb.buffers.outColorUint[idx] = glm::packUnorm4x8(c);
  else
    fb.buffers.outColorVec4[idx] = c;
}

RT_FUNCTION uint32_t pixelIndex(
    const FramebufferGPUData &fb, const uvec2 &pixel)
{
  return pixel.x + pixel.y * fb.size.x;
}

} // namespace detail

RT_FUNCTION void accumResults(const FramebufferGPUData &fb,
    const uvec2 &pixel,
    const vec4 &color,
    float depth,
    const vec3 &albedo,
    const vec3 &normal)
{
  const uint32_t idx = detail::pixelIndex(fb, pixel);

  detail::accumValue(fb.buffers.colorAccumulation, idx, fb.frameID, color);
  detail::accumDepth(fb.buffers.depth, idx, fb.frameID, depth);
  detail::accumValue(fb.buffers.albedo, idx, fb.frameID, albedo);
  detail::accumValue(fb.buffers.normal, idx, fb.frameID, normal);

  const auto accumColor = fb.buffers.colorAccumulation[idx];
  detail::writeOutputColor(fb, accumColor, idx);

  if (fb.checkerboardID == 0 && fb.frameID == 0) {
    auto adjPix = uvec2(pixel.x + 1, pixel.y + 0);
    if (!pixelOutOfFrame(adjPix, fb))
      detail::writeOutputColor(fb, accumColor, detail::pixelIndex(fb, adjPix));

    adjPix = uvec2(pixel.x + 0, pixel.y + 1);
    if (!pixelOutOfFrame(adjPix, fb))
      detail::writeOutputColor(fb, accumColor, detail::pixelIndex(fb, adjPix));

    adjPix = uvec2(pixel.x + 1, pixel.y + 1);
    if (!pixelOutOfFrame(adjPix, fb))
      detail::writeOutputColor(fb, accumColor, detail::pixelIndex(fb, adjPix));
  }
}

} // namespace visrtx
