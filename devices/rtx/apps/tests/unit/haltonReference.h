// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace halton_reference {
// Reverse integer digits into an exact rational before ONE double division.
// Unlike production this does not accumulate rounded float powers of 1/base.
inline double inverse(uint32_t index, uint32_t base)
{
  uint64_t numerator = 0, denominator = 1;
  while (index) {
    numerator = numerator * base + index % base;
    denominator *= base;
    index /= base;
  }
  return std::min(
      double(numerator) / double(denominator), double(0x1.fffffep-1f));
}
inline std::array<double, 4> point(uint32_t index)
{
  return {inverse(index, 2),
      inverse(index, 3),
      inverse(index, 5),
      inverse(index, 7)};
}
} // namespace halton_reference
