// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Per-channel texel reduction of a bound texture, in sampler-output space. The
// emission classifier's value source hands these to the fold. SDK-free so both
// libmdl and the device sampler produce/consume it without the MDL SDK.

#include <array>

namespace visrtx::libmdl {

struct ResourceStats
{
  bool valid{false}; // false ⇒ unbound/invalid ⇒ a lookup folds to 0
  std::array<float, 3>
      maxAbs{}; // maxAbs==0 (with transferPreservesZero) ⇒ zero
  std::array<float, 3> meanPositive{}; // mean of max(texel,0) ⇒ magnitude proxy
  std::array<float, 3> minValue{}; // minValue>=0 ⇒ sign ProvablyNonnegative
  bool transferPreservesZero{
      true}; // T(0)==0; else the stored-zero bound breaks
  bool finite{true};
};

} // namespace visrtx::libmdl
