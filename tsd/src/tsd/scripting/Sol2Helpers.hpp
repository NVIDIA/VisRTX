// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header defines comparison operators for TSD types that don't have them
// in their core headers. These operators are only needed for Sol2's template
// machinery and are intentionally defined here to avoid polluting the core
// library public headers.
//
// Include this header AFTER the TSD headers and BEFORE <sol/sol.hpp>.

#include "tsd/scene/Layer.hpp"
#include "tsd/scene/objects/Array.hpp"
#include "tsd/scene/objects/Camera.hpp"
#include "tsd/scene/objects/Geometry.hpp"
#include "tsd/scene/objects/Light.hpp"
#include "tsd/scene/objects/Material.hpp"
#include "tsd/scene/objects/Sampler.hpp"
#include "tsd/scene/objects/SpatialField.hpp"
#include "tsd/scene/objects/Surface.hpp"
#include "tsd/scene/objects/Volume.hpp"

#include <functional>

namespace tsd::scene {

// Macro to generate all 6 comparison operators for a type.
// These compare by pointer identity, which is what Sol2 needs for usertype
// objects that don't have semantic comparison operators.
// Ordering uses std::less to guarantee a total order across unrelated pointers.
#define TSD_SOL2_COMPARISON_OPS(Type)                                          \
  inline bool operator==(const Type &a, const Type &b)                         \
  {                                                                            \
    return &a == &b;                                                           \
  }                                                                            \
  inline bool operator!=(const Type &a, const Type &b)                         \
  {                                                                            \
    return &a != &b;                                                           \
  }                                                                            \
  inline bool operator<(const Type &a, const Type &b)                          \
  {                                                                            \
    return std::less<const Type *>{}(&a, &b);                                  \
  }                                                                            \
  inline bool operator<=(const Type &a, const Type &b)                         \
  {                                                                            \
    return &a == &b || std::less<const Type *>{}(&a, &b);                      \
  }                                                                            \
  inline bool operator>(const Type &a, const Type &b)                          \
  {                                                                            \
    return std::less<const Type *>{}(&b, &a);                                  \
  }                                                                            \
  inline bool operator>=(const Type &a, const Type &b)                         \
  {                                                                            \
    return &a == &b || std::less<const Type *>{}(&b, &a);                      \
  }

TSD_SOL2_COMPARISON_OPS(Array)
TSD_SOL2_COMPARISON_OPS(Surface)
TSD_SOL2_COMPARISON_OPS(SpatialField)
TSD_SOL2_COMPARISON_OPS(Camera)
TSD_SOL2_COMPARISON_OPS(Geometry)
TSD_SOL2_COMPARISON_OPS(Light)
TSD_SOL2_COMPARISON_OPS(Material)
TSD_SOL2_COMPARISON_OPS(Sampler)
TSD_SOL2_COMPARISON_OPS(Volume)
TSD_SOL2_COMPARISON_OPS(LayerNodeData)

#undef TSD_SOL2_COMPARISON_OPS

} // namespace tsd::scene
