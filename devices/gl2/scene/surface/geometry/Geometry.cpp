// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "Geometry.h"

namespace visgl2 {

Geometry::Geometry(VisGL2DeviceGlobalState *s) : Object(ANARI_GEOMETRY, s) {}

Geometry *Geometry::createInstance(
    std::string_view subtype, VisGL2DeviceGlobalState *s)
{
  return (Geometry *)new UnknownObject(ANARI_GEOMETRY, s);
}

} // namespace visgl2

VISGL2_ANARI_TYPEFOR_DEFINITION(visgl2::Geometry *);
