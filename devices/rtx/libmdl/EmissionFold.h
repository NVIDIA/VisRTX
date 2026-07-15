// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Folds an emission IR against a value source into an EmissionDescriptor, using
// the three-valued abstract interpretation of ADR 0007. Pure and SDK-free: it
// walks the IR (which holds no SDK pointers) plus an abstract value source, so
// it runs device-free in tests and against live argument bytes on the device.

#include "EmissionDescriptor.h"
#include "EmissionIR.h"
#include "ResourceStats.h"

#include <array>
#include <string>

namespace visrtx::libmdl {

// Supplies current parameter values and resource stats to the fold, keyed by
// the class-compilation argument NAME (the device's argument block and samplers
// are name-keyed). The device backs it with live argument bytes + sampler
// reductions; tests back it with fakes.
class EmissionValueSource
{
 public:
  virtual ~EmissionValueSource() = default;

  // Current value of a parameter, if known. A float scalar is broadcast to rgb.
  // Returning false makes the parameter symbolic (Unknown), never zero.
  virtual bool color(
      const std::string &parameterName, std::array<float, 3> &out) const = 0;
  virtual bool boolean(const std::string &parameterName, bool &out) const = 0;

  // Stats for a body-literal bound texture (by DB name) or an argument-bound
  // texture (by parameter name). Returning false ⇒ Unknown.
  virtual bool resourceByName(
      const std::string &name, ResourceStats &out) const = 0;
  virtual bool resourceByParam(
      const std::string &parameterName, ResourceStats &out) const = 0;
};

// A value source that knows nothing — every parameter/resource is Unknown. Used
// for the topology-only path and as a base for partial fakes.
class NullValueSource : public EmissionValueSource
{
 public:
  bool color(const std::string &, std::array<float, 3> &) const override
  {
    return false;
  }
  bool boolean(const std::string &, bool &) const override
  {
    return false;
  }
  bool resourceByName(const std::string &, ResourceStats &) const override
  {
    return false;
  }
  bool resourceByParam(const std::string &, ResourceStats &) const override
  {
    return false;
  }
};

EmissionDescriptor foldEmissionDescriptor(
    const EmissionIR &ir, const EmissionValueSource &values);

} // namespace visrtx::libmdl
