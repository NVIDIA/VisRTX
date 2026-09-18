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

#include "gpu/gpu_objects.h"
#include "gpu/lightGeometry.h"

namespace visrtx {

// Radiance an analytic area-light proxy deposits when a ray hits it (ADR
// 0009). One function for every renderer that shows lights, so the camera sees
// the same radiance from each of them, and the ring's cone falloff comes from
// the same leaf the sampler uses.
//
// `origin` is the SHADED POINT the light is being seen from -- the vertex that
// produced this ray, not necessarily the ray's current origin. A ring's cone
// falloff depends on the direction from that point to the emitting point, and a
// path tracer can re-origin a ray mid-flight for a coverage pass-through
// without that being a scattering event. Callers that MIS-weight the deposit
// must pass the same point they measured the NEE density from, or the weight
// and the radiance it scales describe different geometry.
VISRTX_DEVICE vec3 lightProxyRadiance(
    const FrameGPUData &frameData, const SurfaceHit &hit, const vec3 &origin)
{
  const auto &proxy = frameData.world.lightProxies[hit.lightProxyIndex];
  const auto &ld = frameData.registry.lights[proxy.lightIndex];
  switch (ld.type) {
  case LightType::RECT:
    return rectRadiance(ld.rect, ld.color);
  case LightType::RING: {
    const vec3 axis = ringWorldAxis(ld.ring, proxy.xfm);
    const RingPointRelation rel =
        ringRelateToPoint(ld.ring, axis, origin, hit.hitpoint);
    return ringRadiance(ld.ring, ld.color, rel.spot);
  }
  case LightType::SPHERE:
    return sphereRadiance(ld.sphere, ld.color);
  default:
    return vec3(0.0f);
  }
}

} // namespace visrtx
