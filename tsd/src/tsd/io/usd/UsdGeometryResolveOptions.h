// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/TSDMath.hpp"
// std
#include <map>
#include <string>

namespace tsd::io::usd {

/*
 * What resolving a gprim needs that it cannot read off the resolved prim --
 * which is to say, everything an Import decided that does not change over
 * time. An animation binding carries one of these and replays it, so a scrub
 * reproduces the Import's conversion instead of guessing at it again.
 *
 * `uvNamesByPart` replays the attribute-slot assignment: which primvar a Part's
 * material reads as texture coordinates decides which slot every other primvar
 * falls into, and a scrub must not re-resolve materials to find that out. An
 * absent entry means the conventional `st`.
 *
 * Deliberately free of OpenUSD types, so the animation bindings that carry one
 * still declare themselves in builds without USD.
 */
struct GeometryResolveOptions
{
  // Baked into the emitted vertex data; identity for everything but
  // Prototype-internal geometry (ADR 0016).
  tsd::math::mat4 bakeXform{tsd::math::IDENTITY_MAT4};
  bool refine{false};
  int refinementLevel{2};
  std::map<std::string, std::string> uvNamesByPart;
};

} // namespace tsd::io::usd
