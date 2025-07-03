// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Object.hpp"

namespace tsd {

struct Context;

struct SpatialField : public Object
{
  static constexpr anari::DataType ANARI_TYPE = ANARI_SPATIAL_FIELD;

  DECLARE_OBJECT_DEFAULT_LIFETIME(SpatialField);

  SpatialField(Token subtype = tokens::unknown);
  virtual ~SpatialField() = default;

  anari::Object makeANARIObject(anari::Device d) const override;

  tsd::float2 computeValueRange();
};

using SpatialFieldRef = IndexedVectorRef<SpatialField>;

namespace tokens::spatial_field {

extern const Token structuredRegular;
extern const Token unstructured;
extern const Token amr;
extern const Token nanovdb;

} // namespace tokens::spatial_field

} // namespace tsd
