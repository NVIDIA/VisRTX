/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
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

#include "gpu/gpu_decl.h"
#include "gpu/samplerContracts.h"

#include <vector_types.h> // ::float2, ::float4
#include <cstdint>

// Sobol with pbrt-style FastOwenScrambler: a hash-based fast Owen
// approximation, not pbrt's separate OwenScrambler or a guarantee of
// independent random permutations/dimensions. See
// apps/tests/unit/sobol/README.md. Pattern: QMC sample points and sequential
// material draws, PCG fallback.
//
// Transport dimension map (camera jitter and lens use separate Halton):
//   0..1   ambient hemisphere / HDRI cosine strategy (intentionally shared)
//   2..3   analytic-light coordinates OR HDRI CDF row/column
//   4..5   HDRI within-texel jitter
//   6..7   light category / instance selection
//   8..11  first four sequential material draws (then PCG)
// The raw table supports 16 dimensions; none are reserved for the camera.
// Surface dimensions are reused with bounce/transparency reseeding.
// Material consumption includes categorical lobe decisions and remains
// branch-dependent, not fixed semantic roles. Draws on each chosen branch are
// sequenced explicitly; they are not eagerly consumed across early returns.
//
// sampleIndex is a uint32 binary ordinal (period 2^32), not a Gray-code
// index. Renderer addressing and the first-camera-sample center override
// are separate from this pure sampler.

namespace visrtx {

inline constexpr uint32_t kSobolDimHemiU = 0;
inline constexpr uint32_t kSobolDimHemiV = 1;
inline constexpr uint32_t kSobolDimCdfY = 2;
inline constexpr uint32_t kSobolDimCdfX = 3;
inline constexpr uint32_t kSobolDimCdfJitterU = 4;
inline constexpr uint32_t kSobolDimCdfJitterV = 5;
inline constexpr uint32_t kSobolDimLightPick0 = 6;
inline constexpr uint32_t kSobolDimLightPick1 = 7;
inline constexpr uint32_t kSobolDimNextRay0 = 8;
inline constexpr uint32_t kSobolNumDimensions = 16;

// Attributed, immutable pbrt/Joe-Kuo table; intentionally included in
// namespace.
#include "gpu/sobolDirections.h"

VISRTX_HOST_DEVICE uint32_t reverseBits32(uint32_t x)
{
#ifdef __CUDA_ARCH__
  return __brev(x);
#else
  x = (x << 16) | (x >> 16);
  x = ((x & 0x00ff00ffu) << 8) | ((x & 0xff00ff00u) >> 8);
  x = ((x & 0x0f0f0f0fu) << 4) | ((x & 0xf0f0f0f0u) >> 4);
  x = ((x & 0x33333333u) << 2) | ((x & 0xccccccccu) >> 2);
  x = ((x & 0x55555555u) << 1) | ((x & 0xaaaaaaaau) >> 1);
  return x;
#endif
}

// pbrt-v4 FastOwenScrambler operations, unchanged (not OwenScrambler).
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Licensed under Apache-2.0; see apps/tests/unit/sobol/LICENSE.pbrt and the
// pinned excerpt there. Modified here only for VisRTX naming/annotations.
VISRTX_HOST_DEVICE uint32_t owenScramble(uint32_t v, uint32_t seed)
{
  v = reverseBits32(v);
  v ^= v * 0x3d20adeau;
  v += seed;
  v *= (seed >> 16) | 1u;
  v ^= v * 0x05526c56u;
  v ^= v * 0x53a22864u;
  return reverseBits32(v);
}

VISRTX_HOST_DEVICE uint32_t sobolUInt(uint32_t index, uint32_t dimension)
{
  if (dimension >= kSobolNumDimensions)
    detail::samplerContractFailure();
  uint32_t x = 0;
  for (uint32_t bit = 0; index; ++bit, index >>= 1u) {
    if (index & 1u)
      x ^= kSobolDirection[dimension][bit];
  }
  return x;
}

VISRTX_HOST_DEVICE uint32_t owenPixelSeed(uint32_t x, uint32_t y)
{
  // Preserve the legacy Halton pixel hash, now seeding Owen rather than index.
  uint32_t v = x + y * 0x9E3779B9u + 0xB5297A4Du;
  v ^= v >> 16;
  v *= 0x7feb352du;
  v ^= v >> 15;
  v *= 0x846ca68bu;
  v ^= v >> 16;
  return v;
}

// Bounce 0 / no alpha pass-through keeps the transport pixel seed (XOR 0).
// Later bounces and coverage-continue vertices reseed so dims 0–11 can be
// reused without repeating an earlier vertex's sequence.
VISRTX_HOST_DEVICE uint32_t owenBounceSeed(
    uint32_t pixelSeed, int bounce, int transparency = 0)
{
  return pixelSeed ^ (0x9e3779b9u * uint32_t(bounce))
      ^ (0x85ebca6bu * uint32_t(transparency));
}

// Exact 24-bit half-open grid; zero is a legal sample, not an error.
VISRTX_HOST_DEVICE float sobolFloat(uint32_t scrambled)
{
  return float(scrambled >> 8) * 0x1p-24f;
}

// Uniform on the 24-bit grid in [0, 1). Dimension mixing does not prove
// independence. Zero is legal and the maximum is 1 - 2^-24; no clamp.
VISRTX_HOST_DEVICE float owenSobol(
    uint32_t sampleIndex, uint32_t dimension, uint32_t pixelSeed)
{
  const uint32_t dimSeed = pixelSeed ^ (0x9e3779b9u * (dimension + 1u));
  const uint32_t scrambled =
      owenScramble(sobolUInt(sampleIndex, dimension), dimSeed);
  return sobolFloat(scrambled);
}

VISRTX_HOST_DEVICE ::float2 owenSobolHemi(
    uint32_t sampleIndex, uint32_t pixelSeed)
{
  return ::float2{owenSobol(sampleIndex, kSobolDimHemiU, pixelSeed),
      owenSobol(sampleIndex, kSobolDimHemiV, pixelSeed)};
}

} // namespace visrtx
