// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// The emission descriptor — the classifier's output and the seam between
// MDL-pure analysis and the renderer. It is SDK-free (no mi:: types) so the
// device/renderer consume it without the MDL SDK. It DESCRIBES; a thin
// renderer-side policy decides registration. See ADR 0007.

#include <array>
#include <cstdint>

namespace visrtx::libmdl {

// Three-valued emission verdict for one material side.
enum class EmissionVerdict : std::uint8_t
{
  ProvablyNull, // identically zero at the current snapshot — never an emitter
  ProvablyEmissive, // provably nonzero somewhere
  Unknown, // could be either (register-eligible if faithful; worst case perf)
};

// EDF kinds present in a slot, as a bitmask. `Unknown` marks an unmodeled EDF
// leaf — described, never registered.
enum class EdfKind : std::uint8_t
{
  None = 0,
  Diffuse = 1 << 0,
  Spot = 1 << 1,
  Directional = 1 << 2,
  Measured = 1 << 3,
  Unknown = 1 << 4,
};

inline EdfKind operator|(EdfKind a, EdfKind b)
{
  return EdfKind(std::uint8_t(a) | std::uint8_t(b));
}
inline EdfKind &operator|=(EdfKind &a, EdfKind b)
{
  a = a | b;
  return a;
}
inline bool hasKind(EdfKind set, EdfKind k)
{
  return (std::uint8_t(set) & std::uint8_t(k)) != 0;
}
// True iff every kind in `set` is contained in `allowed` (the ⊆ test), and the
// set is non-empty.
inline bool isSubsetOf(EdfKind set, EdfKind allowed)
{
  return set != EdfKind::None
      && (std::uint8_t(set) & ~std::uint8_t(allowed)) == 0;
}

enum class IntensityMode : std::uint8_t
{
  RadiantExitance, // radiance = intensity / PI (the faithful, default mode)
  Power, // total power over area — not faithfully handled yet
};

enum class EmissionSign : std::uint8_t
{
  ProvablyNonnegative, // no reachable emission value can be negative
  Unknown, // possibly negative — register would drop the NEE term, so
           // forward-only
};

struct SlotDescriptor
{
  EmissionVerdict verdict{EmissionVerdict::ProvablyNull};
  EdfKind edfKinds{EdfKind::None};
  // Non-negative mean-radiance proxy (meanPositive · folded scale), per
  // channel; feeds Pick Power only, never gates zero. Zero is a valid value.
  std::array<float, 3> magnitude{};
  IntensityMode mode{IntensityMode::RadiantExitance};
  // The emission (EDF or intensity) reads a geometric-state quantity the
  // synthetic next-event hit fabricates (normal/tangent/object-id/position);
  // registering it would evaluate a different integrand than the forward hit.
  bool dependsOnGeometricState{false};
  EmissionSign sign{EmissionSign::Unknown};
};

struct EmissionDescriptor
{
  SlotDescriptor surface;
  SlotDescriptor backface;
};

} // namespace visrtx::libmdl
