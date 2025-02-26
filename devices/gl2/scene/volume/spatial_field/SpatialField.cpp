// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "SpatialField.h"

namespace visgl2 {

SpatialField::SpatialField(VisGL2DeviceGlobalState *s)
    : Object(ANARI_SPATIAL_FIELD, s)
{}

SpatialField *SpatialField::createInstance(
    std::string_view subtype, VisGL2DeviceGlobalState *s)
{
  return (SpatialField *)new UnknownObject(ANARI_SPATIAL_FIELD, s);
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_DEFINITION(visgl2::SpatialField *);
