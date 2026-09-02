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
    uint32_t optixFlags,
    uint32_t visibilityMask)
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
      OptixVisibilityMask(visibilityMask),
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

// Ray-class visibility masks (ADR 0009). Which light proxies a ray may hit is a
// property of the trace, so a renderer that does not deposit proxy hits simply
// never names them and its rays cannot reach one.
//
// The default is geometry-only: every existing call site keeps its exact current
// behavior, and opting into proxies is explicit.

// Real scene geometry only. Shadow/occlusion rays use this, so a light neither
// shadows the scene nor self-occludes.
VISRTX_DEVICE constexpr uint32_t geometryOnlyMask()
{
  return VISRTX_MASK_GEOMETRY;
}

// Camera/primary rays in a renderer that shows lights: geometry plus proxies of
// lights whose `visible` is true.
VISRTX_DEVICE constexpr uint32_t primaryWithVisibleLightsMask()
{
  return VISRTX_MASK_GEOMETRY | VISRTX_MASK_LIGHT_PROXY_VISIBLE;
}

// Reflection/GI continuation rays: geometry plus ALL light proxies, including
// visible=false ones — hiding a light from the camera must not remove it from
// indirect illumination.
VISRTX_DEVICE constexpr uint32_t secondaryWithAllLightsMask()
{
  return VISRTX_MASK_GEOMETRY | VISRTX_MASK_LIGHT_PROXY_VISIBLE
      | VISRTX_MASK_LIGHT_PROXY_HIDDEN;
}

template <typename T>
VISRTX_DEVICE void intersectSurface(ScreenSample &ss,
    Ray r,
    T rayType,
    void *dataPtr = nullptr,
    uint32_t optixFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT,
    uint32_t visibilityMask = VISRTX_MASK_GEOMETRY)
{
  detail::launchRay(ss, r, rayType, true, dataPtr, optixFlags, visibilityMask);
}

template <typename T>
VISRTX_DEVICE void intersectVolume(ScreenSample &ss,
    Ray r,
    T rayType,
    void *dataPtr = nullptr,
    uint32_t optixFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT)
{
  // Volumes live in their own traversable with no light proxies in it.
  detail::launchRay(
      ss, r, rayType, false, dataPtr, optixFlags, VISRTX_MASK_GEOMETRY);
}

// Apply a cutting plane to a ray.
// The plane is encoded as vec4(N.x, N.y, N.z, d) where the visible half-space
// is { p | dot(N,p)+d >= 0 }.  Disabled when cp == vec4(0) (GPU default).
// Modifies ray.org and ray.t.upper in place.
VISRTX_DEVICE void applyCuttingPlane(const vec4 &cp, Ray &ray)
{
  if (cp == vec4(0.f))
    return;
  const vec3 N(cp.x, cp.y, cp.z);
  const float dist_org = glm::dot(N, ray.org) + cp.w;
  const float denom = glm::dot(N, ray.dir);
  if (dist_org >= 0.f) {
    // camera on visible side: clip where ray exits the half-space
    if (denom < 0.f) {
      float t_cut = -dist_org / denom;
      ray.t.upper = glm::min(ray.t.upper, t_cut);
    }
  } else {
    // camera on invisible side: advance origin to where ray enters
    if (denom > 0.f) {
      float t_entry = -dist_org / denom;
      ray.org += t_entry * ray.dir;
    } else {
      ray.t.upper = 0.f; // ray never enters visible half-space
    }
  }
}

} // namespace visrtx
