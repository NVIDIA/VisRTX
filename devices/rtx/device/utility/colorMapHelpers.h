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

#include "gpu/gpu_math.h"
// std
#include <algorithm>
#include <vector>
// anari
#include "utility/Span.h"

namespace visrtx {

inline std::vector<float> generateLinearPositions(size_t n, box1 range)
{
  std::vector<float> positions(n);
  positions.front() = 0.f;
  positions.back() = 1.f;

  float w = 1.f / (n - 1);
  for (int i = 1; i < positions.size() - 1; i++)
    positions[i] = positions[i - 1] + w;

  std::transform(
      positions.begin(), positions.end(), positions.begin(), [&](float p) {
        return p * size(range) + range.lower;
      });

  return positions;
}

template <typename T>
inline T getInterpolatedValue(
    const T *values, const Span<float> &positions, box1 range, float pos)
{
  for (size_t i = 0; i < positions.size() - 1; i++) {
    box1 r(position(positions[i], range), position(positions[i + 1], range));
    if (contains(pos, r))
      return glm::mix(values[i], values[i + 1], position(pos, r));
  }

  return pos <= position(positions[0], range) ? values[0]
                                              : values[positions.size() - 1];
}

} // namespace visrtx
