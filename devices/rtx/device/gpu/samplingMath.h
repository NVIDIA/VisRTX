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

#include "gpu_math.h"
#include "pcg.h"

#include <glm/geometric.hpp>

namespace visrtx {

// Rec.709 luminance.
VISRTX_DEVICE float luminance(const vec3 &c)
{
  return glm::dot(c, vec3(0.2126f, 0.7152f, 0.0722f));
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
// u1/u2 in [0, 1). The RandState overload draws them from PCG in that order.
// Quality env/ambient NEE intentionally shares Owen-scrambled Sobol dims 0–1,
// with per-surface-bounce/transparency reseeding.
VISRTX_DEVICE vec3 sampleHemisphere(float u1, float u2, const vec3 &normal)
{
  const float r = sqrtf(u1);
  const float z = sqrtf(fmaxf(0.f, 1.f - r * r));
  const float phi = kTwoPi * u2;
  const vec3 sample(r * cosf(phi), r * sinf(phi), z);
  return computeOrthonormalBasis(normal) * sample;
}

VISRTX_DEVICE vec3 sampleHemisphere(PCGState &rs, const vec3 &normal)
{
  const float u = pcg_uniform(&rs);
  const float v = pcg_uniform(&rs);
  return sampleHemisphere(u, v, normal);
}

} // namespace visrtx
