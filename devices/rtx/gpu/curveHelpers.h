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

#include "gpu/gpu_util.h"

#include <glm/vec4.hpp>

namespace visrtx {

struct LinearBSplineSegment
{
  VISRTX_DEVICE LinearBSplineSegment(const vec4 *q)
  {
    p[0] = q[0];
    p[1] = q[1] - q[0];
  }

  VISRTX_DEVICE float radius(const float &u) const
  {
    return p[0].w + p[1].w * u;
  }

  VISRTX_DEVICE vec3 position3(float u) const
  {
    return (vec3 &)p[0] + u * (vec3 &)p[1];
  }
  VISRTX_DEVICE vec4 position4(float u) const
  {
    return p[0] + u * p[1];
  }

  VISRTX_DEVICE float min_radius(float u1, float u2) const
  {
    return fminf(radius(u1), radius(u2));
  }

  VISRTX_DEVICE float max_radius(float u1, float u2) const
  {
    if (!p[1].w)
      return p[0].w;
    return fmaxf(radius(u1), radius(u2));
  }

  VISRTX_DEVICE vec3 velocity3(float u) const
  {
    return (vec3 &)p[1];
  }
  VISRTX_DEVICE vec4 velocity4(float u) const
  {
    return p[1];
  }

  VISRTX_DEVICE vec3 acceleration3(float u) const
  {
    return vec3(0.f);
  }

  vec4 p[2];
};

VISRTX_DEVICE vec3 curveSurfaceNormal(
    const LinearBSplineSegment &bc, float u, const vec3 &ps)
{
  const vec4 p4 = bc.position4(u);
  const vec3 p = p4;
  const float r = p4.w;
  const vec4 d4 = bc.velocity4(u);
  const vec3 d = d4;

  float dd = dot(d, d);

  vec3 o1 = ps - p;
  o1 -= (dot(o1, d) / dd) * d;
  o1 *= r / length(o1);
  return normalize(o1);
}

} // namespace visrtx