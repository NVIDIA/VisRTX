// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/objects/SpatialField.hpp"

namespace tsd {

SpatialField::SpatialField(Token stype) : Object(ANARI_SPATIAL_FIELD, stype)
{
  if (stype == tokens::spatial_field::structuredRegular)
  {
    addParameter("origin")
        .setValue(float3(0.f, 0.f, 0.f))
        .setDescription("bottom-left corner of the field");
    addParameter("spacing")
        .setValue(float3(1.f, 1.f, 1.f))
        .setMin(float3(0.f, 0.f, 0.f))
        .setDescription("voxel size in object-space units");
  }
}

anari::Object SpatialField::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::SpatialField>(d, subtype().c_str());
}

namespace tokens::spatial_field {

const Token structuredRegular = "structuredRegular";
const Token amr = "amr";

} // namespace tokens::spatial_field

} // namespace tsd
