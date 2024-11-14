// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/objects/SpatialField.hpp"

namespace tsd {

SpatialField::SpatialField(Token stype) : Object(ANARI_SPATIAL_FIELD, stype) {}

anari::Object SpatialField::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::SpatialField>(d, subtype().c_str());
}

namespace tokens::spatial_field {

const Token structuredRegular = "structuredRegular";
const Token amr = "amr";

} // namespace tokens::spatial_field

} // namespace tsd
