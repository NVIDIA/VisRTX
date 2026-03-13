// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/scene/ObjectUsePtr.hpp"

namespace tsd::scene {

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
using SpatialFieldAppRef = ObjectUsePtr<SpatialField, Object::UseKind::APP>;

namespace tokens::spatial_field {

extern const Token structuredRegular;
extern const Token structuredRectilinear;
extern const Token unstructured;
extern const Token amr;
extern const Token nanovdb;
extern const Token nanovdbRectilinear;

} // namespace tokens::spatial_field

} // namespace tsd::scene
