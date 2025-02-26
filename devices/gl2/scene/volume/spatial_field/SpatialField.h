// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "Object.h"

namespace visgl2 {

// Inherit from this, add your functionality, and add it to 'createInstance()'
struct SpatialField : public Object
{
  SpatialField(VisGL2DeviceGlobalState *d);
  virtual ~SpatialField() = default;
  static SpatialField *createInstance(
      std::string_view subtype, VisGL2DeviceGlobalState *d);
};

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_SPECIALIZATION(
    visgl2::SpatialField *, ANARI_SPATIAL_FIELD);
