// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Object.hpp"

namespace tsd {

struct SpatialField : public Object
{
  static constexpr anari::DataType ANARI_TYPE = ANARI_SPATIAL_FIELD;

  DECLARE_OBJECT_DEFAULT_LIFETIME(SpatialField);

  SpatialField(Token subtype = tokens::unknown);
  virtual ~SpatialField() = default;

  anari::Object makeANARIObject(anari::Device d) const override;
};

using SpatialFieldRef = IndexedVectorRef<SpatialField>;

namespace tokens::spatial_field {

extern const Token structuredRegular;
extern const Token amr;

} // namespace tokens::spatial_field

} // namespace tsd
