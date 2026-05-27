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

#include "gpu/gpu_decl.h"

#include <vector_types.h> // ::float4
#include <cstdint>

// PCG-XSH-RR-32 (O'Neill 2014, "PCG: A Family of Simple Fast Space-Efficient
// Statistically Good Algorithms for Random Number Generation"). 64-bit
// internal state, 32-bit output, period 2^64. Passes TestU01 BigCrush.
//
// 8 B per-thread state, single-output, uniform convention [0, 1). `inc`
// selects one of 2^63 parallel streams — that independence is what
// makes this fit per-pixel seeds keyed on (pixel_id, frame_id).

namespace visrtx {

struct PCGState
{
  uint64_t state;
  uint64_t inc; // odd;
};

namespace detail {

VISRTX_DEVICE uint64_t pcg_mix64(uint64_t v)
{
  v += 0x9E3779B97F4A7C15ULL;
  v = (v ^ (v >> 30u)) * 0xBF58476D1CE4E5B9ULL;
  v = (v ^ (v >> 27u)) * 0x94D049BB133111EBULL;
  return v ^ (v >> 31u);
}

// One step of the PCG-XSH-RR-32 permutation. Internal helper; callers want
// pcg_uniform / pcg_uniform4 below.
VISRTX_DEVICE uint32_t pcg_advance(PCGState *s)
{
  // LCG multiplier constants from PCG canonical implementation.
  constexpr uint64_t LCG_MULT = 6364136223846793005ULL;
  const uint64_t oldState = s->state;
  s->state = oldState * LCG_MULT + s->inc;
  // XSH-RR-32 output function: xorshift, then rotate the high bits.
  const uint32_t xorshifted = uint32_t(((oldState >> 18u) ^ oldState) >> 27u);
  const uint32_t rot = uint32_t(oldState >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
}

} // namespace detail

// Seed the stream. `seed` becomes the initial counter; `streamId` selects
// one of 2^63 statistically-independent streams (must be made odd; the
// `| 1u` ensures this). Callers should hash structured keys before passing
// them here; adjacent stream IDs with tiny seeds can show visible structure
// when a pixel consumes only the first few outputs.
VISRTX_DEVICE void pcg_init(PCGState *s, uint64_t seed, uint64_t streamId)
{
  s->state = 0u;
  s->inc = (streamId << 1u) | 1u;
  (void)detail::pcg_advance(s);
  s->state += seed;
  (void)detail::pcg_advance(s);
}

// Float in [0, 1). 24-bit mantissa precision — matches the natural
// single-precision resolution; the half-open convention matches
// std::uniform_real_distribution and most modern RNGs.
VISRTX_DEVICE float pcg_uniform(PCGState *s)
{
  // Upper 24 bits of the 32-bit PCG output → integer in [0, 2^24).
  // Multiplied by 2^-24 gives a float in [0, 1 - 2^-24] ⊂ [0, 1).
  return float(detail::pcg_advance(s) >> 8) * (1.0f / 16777216.0f);
}

VISRTX_DEVICE ::float4 pcg_uniform4(PCGState *s)
{
  // Four independent draws. Compiler should pipeline the four PCG
  // advances since each only depends on the previous state.
  const float x = pcg_uniform(s);
  const float y = pcg_uniform(s);
  const float z = pcg_uniform(s);
  const float w = pcg_uniform(s);
  return ::float4{x, y, z, w};
}

} // namespace visrtx
