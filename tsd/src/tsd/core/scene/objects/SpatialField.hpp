// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/scene/Object.hpp"

namespace tsd::core {

struct Scene;

struct SpatialField : public Object
{
  DECLARE_OBJECT_DEFAULT_LIFETIME(SpatialField);

  SpatialField(Token subtype = tokens::unknown);
  virtual ~SpatialField() = default;

  ObjectPoolRef<SpatialField> self() const;

  anari::Object makeANARIObject(anari::Device d) const override;

  tsd::math::float2 computeValueRange();
};

using SpatialFieldRef = ObjectPoolRef<SpatialField>;

namespace tokens::spatial_field {

extern const Token structuredRegular;
extern const Token structuredRectilinear;
extern const Token unstructured;
extern const Token amr;
extern const Token nanovdb;
extern const Token nanovdbRectilinear;

} // namespace tokens::spatial_field

} // namespace tsd::core
