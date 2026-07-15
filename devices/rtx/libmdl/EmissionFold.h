// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Folds an emission IR against a value source into an EmissionDescriptor, using
// the three-valued abstract interpretation of ADR 0007. Pure and SDK-free: it
// walks the IR (which holds no SDK pointers) plus an abstract value source, so
// it runs device-free in tests and against live argument bytes on the device.

#include "EmissionDescriptor.h"
#include "EmissionIR.h"

#include <array>
#include <string>

namespace visrtx::libmdl {

// Per-channel texel reduction of a bound texture, in sampler-output space.
struct ResourceStats
{
  bool valid{false}; // false ⇒ unbound/invalid ⇒ lookup folds to 0
  std::array<float, 3> maxAbs{}; // maxAbs==0 ⇒ ProvablyZero (exact)
  std::array<float, 3> meanPositive{}; // mean of max(texel,0) ⇒ magnitude proxy
  std::array<float, 3> minValue{}; // minValue>=0 ⇒ sign ProvablyNonnegative
  bool transferPreservesZero{true}; // T(0)==0; else the zero bound breaks
  bool finite{true};
};

// Supplies current parameter values and resource stats to the fold. The device
// backs it with live argument bytes + the resource table; tests back it with
// fakes or the compiled material's own arguments. All indices are IR parameter
// indices; resource names are IR resource DB names.
class EmissionValueSource
{
 public:
  virtual ~EmissionValueSource() = default;

  // Current value of a parameter, if known. A float scalar is broadcast to rgb.
  // Returning false makes the parameter symbolic (Unknown), never zero.
  virtual bool color(int parameterIndex, std::array<float, 3> &out) const = 0;
  virtual bool boolean(int parameterIndex, bool &out) const = 0;

  // Stats for a body-literal bound texture (by DB name) or an argument-bound
  // texture (by parameter index; name empty). Returning false ⇒ Unknown.
  virtual bool resourceByName(
      const std::string &name, ResourceStats &out) const = 0;
  virtual bool resourceByParam(
      int parameterIndex, ResourceStats &out) const = 0;
};

// A value source that knows nothing — every parameter/resource is Unknown. Used
// for the topology-only path and as a base for partial fakes.
class NullValueSource : public EmissionValueSource
{
 public:
  bool color(int, std::array<float, 3> &) const override
  {
    return false;
  }
  bool boolean(int, bool &) const override
  {
    return false;
  }
  bool resourceByName(const std::string &, ResourceStats &) const override
  {
    return false;
  }
  bool resourceByParam(int, ResourceStats &) const override
  {
    return false;
  }
};

EmissionDescriptor foldEmissionDescriptor(
    const EmissionIR &ir, const EmissionValueSource &values);

} // namespace visrtx::libmdl
