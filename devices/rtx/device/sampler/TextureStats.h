// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Device-side texel reduction for Image2D: one thrust pass over the resident
// texels yields the raw per-source-channel accumulators the emissive Pick Power
// and the emission classifier (maxAbs / meanPositive / minValue) are built
// from. SDK-free and MDL-agnostic — averageValue() feeds the non-MDL PBR path.

#include <array>
#include <cfloat>
#include <cstddef>

namespace visrtx {

enum class TexelFormat
{
  Unsupported,
  Float32, // raw float channels
  Fixed8, // uint8/255, linear
  Srgb8 // uint8/255, sRGB->linear on color channels
};

// Raw per-source-channel reduction (up to 4 channels), decoded to linear. The
// caller applies the magnitude/broadcast and sign policy.
struct TexelAccum
{
  std::array<double, 4> posSum{}; // Σ max(v,0)      → meanPositive
  std::array<float, 4> maxAbs{}; // max |v|         → exact zero proof
  std::array<float, 4> minValue{
      {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX}}; // min v → sign proof
  bool finite{true};
};

// Reduce `count` texels of `nc` channels resident at `deviceData` (a device
// pointer from Array::data(AddressSpace::GPU)). sRGB color channels (source
// index < colorChannels) are linearized to match the hardware sampler.
TexelAccum reduceTexelsDevice(const void *deviceData,
    TexelFormat fmt,
    int nc,
    int colorChannels,
    std::size_t count);

} // namespace visrtx
