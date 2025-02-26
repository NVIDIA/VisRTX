// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "Object.h"

namespace visgl2 {

// Inherit from this, add your functionality, and add it to 'createInstance()'
struct Geometry : public Object
{
  Geometry(VisGL2DeviceGlobalState *s);
  ~Geometry() override = default;
  static Geometry *createInstance(
      std::string_view subtype, VisGL2DeviceGlobalState *s);
};

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_SPECIALIZATION(visgl2::Geometry *, ANARI_GEOMETRY);
