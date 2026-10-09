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

// View-adjusted shading normal: a mapped or interpolated normal that leans
// away from the viewer is raised toward the geometric normal until the view's
// mirror direction clears the geometric surface. Port of Cycles'
// ensure_valid_specular_reflection, as halcyon's PathPbm_validSpecularNormal.
// A renderer policy, like the dot(Ng, Ns) < 0 flip in populateHit: applied
// once per hit at material init by the CUDA physicallyBased shader and, for
// every MDL material, through MDL's adapt_normal hook.

#include "gpu/gpu_math.h"

namespace visrtx {

// Ng: geometric normal facing the viewer. V: unit direction from the hit back
// toward the viewer. N: unit shading normal. Returns N itself when it is Ng,
// when it is already valid, or when there is no (N, Ng) plane to rotate it in.
VISRTX_DEVICE vec3 adjustNormalForView(
    const vec3 &Ng, const vec3 &V, const vec3 &N)
{
  if (N == Ng)
    return N;
  const vec3 R = 2.f * dot(N, V) * N - V;
  const float VdotNg = fmaxf(dot(V, Ng), 0.f);
  const float threshold = fminf(0.9f * VdotNg, 0.01f);
  if (dot(Ng, R) >= threshold)
    return N;

  // Solve for the normal in the (N, Ng) plane whose mirror direction of V
  // makes exactly `threshold` with the surface: X is N's component
  // orthogonal to Ng, and the quadratic is in the squared Ng component.
  vec3 X = N - dot(N, Ng) * Ng;
  const float lengthX = length(X);
  if (!isfinite(lengthX) || lengthX <= 1e-7f)
    return N;
  X /= lengthX;
  const float VdotX = dot(V, X);
  const float a = VdotX * VdotX + VdotNg * VdotNg;
  if (a <= 1e-7f)
    return N;
  const float b = 2.f * (a + VdotNg * threshold);
  const float c = pow2(threshold + VdotNg);
  const float root = sqrtf(fmaxf(b * b - 4.f * a * c, 0.f));
  const float Nz2 = 0.25f * (VdotX < 0.f ? b + root : b - root) / a;
  return X * sqrtf(fmaxf(1.f - Nz2, 0.f)) + Ng * sqrtf(fmaxf(Nz2, 0.f));
}

} // namespace visrtx
