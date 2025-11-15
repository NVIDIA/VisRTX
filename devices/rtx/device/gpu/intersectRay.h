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

#include "gpu/gpu_util.h"

namespace visrtx {
namespace detail {

template <typename T>
VISRTX_DEVICE void launchRay(ScreenSample &ss,
    Ray r,
    T rayType,
    bool tracingSurfaces,
    void *dataPtr,
    uint32_t optixFlags)
{
  uint32_t bvhSelection = tracingSurfaces;

  uint32_t u0, u1;
  packPointer(&ss, u0, u1);

  uint32_t u2, u3;
  packPointer(dataPtr, u2, u3);

  OptixTraversableHandle traversable = tracingSurfaces
      ? ss.frameData->world.surfacesTraversable
      : ss.frameData->world.volumesTraversable;

  optixTrace(traversable,
      (::float3 &)r.org,
      (::float3 &)r.dir,
      r.t.lower,
      r.t.upper,
      0.0f,
      OptixVisibilityMask(255),
      optixFlags,
      static_cast<uint32_t>(rayType) * NUM_SBT_PRIMITIVE_INTERSECTOR_ENTRIES,
      0u,
      static_cast<uint32_t>(rayType),
      u0,
      u1,
      u2,
      u3,
      bvhSelection);
}

} // namespace detail

VISRTX_DEVICE uint32_t primaryRayOptiXFlags(const RendererGPUData &rd)
{
  return rd.cullTriangleBF ? OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES
                           : OPTIX_RAY_FLAG_NONE;
}

template <typename T>
VISRTX_DEVICE void intersectSurface(ScreenSample &ss,
    Ray r,
    T rayType,
    void *dataPtr = nullptr,
    uint32_t optixFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT)
{
  detail::launchRay(ss, r, rayType, true, dataPtr, optixFlags);
}

template <typename T>
VISRTX_DEVICE void intersectVolume(ScreenSample &ss,
    Ray r,
    T rayType,
    void *dataPtr = nullptr,
    uint32_t optixFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT)
{
  detail::launchRay(ss, r, rayType, false, dataPtr, optixFlags);
}

template <typename T>
VISRTX_DEVICE float surfaceAttenuation(ScreenSample &ss, Ray r, T rayType)
{
  float a = 0.f;
  intersectSurface(ss, r, rayType, &a, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return a;
}

} // namespace visrtx
